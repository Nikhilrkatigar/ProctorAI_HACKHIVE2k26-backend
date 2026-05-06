import asyncio
import base64
import json
import logging
import time
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import cv2

from detectors.face_detector import FaceDetector
from detectors.eye_tracker import EyeTracker
from detectors.head_pose import HeadPoseEstimator
from detectors.person_counter import PersonCounter
from detectors.alert_engine import AlertEngine
from detectors.shared_facemesh import SharedFaceMesh
from utils.snapshot import save_snapshot
from models.database import get_db, Violation
from models import schemas

logger = logging.getLogger("proctorAI.ws")


class CandidateSession:
    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self.websocket: WebSocket | None = None
        self.reference_encoding = None
        self.alert_engine = AlertEngine(candidate_id)
        self.last_frame_time = time.time()
        self.gaze_off_center_since: float | None = None
        self.tab_switch_count = 0
        self.exam_id: str | None = None
        self.candidate_name: str = candidate_id
        self.is_active = True
        self.latest_frame_b64: str | None = None  # For live video relay
        self._processing_lock = asyncio.Lock()     # Prevent concurrent frame processing

    def to_status_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "risk_score": self.alert_engine.risk_score,
            "status": self.alert_engine.get_status(),
            "tab_switches": self.tab_switch_count,
            "last_alert": self.alert_engine.last_alert,
            "score_history": self.alert_engine.score_history[-20:],
            "is_active": self.is_active,
            "exam_id": self.exam_id,
        }


class ProctorWebSocketManager:
    def __init__(self):
        self.student_sessions: Dict[str, CandidateSession] = {}
        self.proctor_sockets: Set[WebSocket] = set()

        # Shared detectors (loaded once)
        self.shared_facemesh = SharedFaceMesh()
        self.face_detector = FaceDetector()
        self.eye_tracker = EyeTracker(shared_facemesh=self.shared_facemesh)
        self.head_pose = HeadPoseEstimator(shared_facemesh=self.shared_facemesh)
        self.person_counter = PersonCounter()

    async def handle(self, websocket: WebSocket, candidate_id: str, role: str):
        await websocket.accept()

        if role == "proctor":
            await self._handle_proctor(websocket)
        else:
            await self._handle_student(websocket, candidate_id)

    async def _handle_proctor(self, websocket: WebSocket):
        self.proctor_sockets.add(websocket)
        logger.info("Proctor connected (total: %d)", len(self.proctor_sockets))
        try:
            # Send current state of all candidates immediately
            snapshot = {
                cid: sess.to_status_dict()
                for cid, sess in self.student_sessions.items()
            }
            await websocket.send_json({"type": "INIT", "candidates": snapshot})

            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("action") == "terminate":
                    cid = msg.get("candidate_id")
                    if cid in self.student_sessions:
                        logger.warning("Proctor terminated exam for %s", cid)
                        self.student_sessions[cid].is_active = False
                        if self.student_sessions[cid].websocket:
                            await self.student_sessions[cid].websocket.send_json(
                                {"type": "TERMINATED", "message": "Exam terminated by proctor"}
                            )
                elif msg.get("action") == "request_frame":
                    # Proctor requesting latest frame for a candidate
                    cid = msg.get("candidate_id")
                    if cid in self.student_sessions:
                        frame = self.student_sessions[cid].latest_frame_b64
                        if frame:
                            await websocket.send_json({
                                "type": "LIVE_FRAME",
                                "candidate_id": cid,
                                "frame": frame,
                            })
        except WebSocketDisconnect:
            logger.info("Proctor disconnected")
        except Exception:
            logger.exception("Proctor handler error")
        finally:
            self.proctor_sockets.discard(websocket)

    async def _handle_student(self, websocket: WebSocket, candidate_id: str):
        if candidate_id not in self.student_sessions:
            self.student_sessions[candidate_id] = CandidateSession(candidate_id)

        session = self.student_sessions[candidate_id]
        session.websocket = websocket
        session.is_active = True

        logger.info("Student connected: %s", candidate_id)
        await websocket.send_json({"type": "CONNECTED", "candidate_id": candidate_id})
        await self._broadcast_to_proctors({"type": "CANDIDATE_JOINED", "candidate_id": candidate_id})

        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)

                msg_type = msg.get("type", "FRAME")

                if msg_type == "FRAME":
                    frame_b64 = msg.get("frame", "")
                    if frame_b64:
                        if session._processing_lock.locked():
                            continue
                        asyncio.create_task(
                            self._process_frame(session, frame_b64, websocket)
                        )

                elif msg_type == "REFERENCE_FACE":
                    frame_b64 = msg.get("frame", "")
                    if frame_b64:
                        enc = self.face_detector.encode_reference(frame_b64)
                        if enc is not None:
                            session.reference_encoding = enc
                            logger.info("Reference face captured for %s", candidate_id)
                            await websocket.send_json(
                                {"type": "REFERENCE_CAPTURED", "success": True}
                            )
                        else:
                            logger.warning("No face in reference image for %s", candidate_id)
                            await websocket.send_json(
                                {"type": "REFERENCE_CAPTURED", "success": False,
                                 "message": "No face detected in reference image"}
                            )

                elif msg_type == "TAB_SWITCH":
                    session.tab_switch_count += 1
                    severity = "MEDIUM" if session.tab_switch_count < 3 else "HIGH"
                    alert = session.alert_engine.add_alert(
                        "TAB_SWITCH", severity,
                        extra={"count": session.tab_switch_count}
                    )
                    if alert:
                        self._persist_violation(session, alert)
                    await self._dispatch_alert(session, alert, websocket)

                elif msg_type == "CLIPBOARD_VIOLATION":
                    alert = session.alert_engine.add_alert("CLIPBOARD_VIOLATION", "LOW")
                    if alert:
                        self._persist_violation(session, alert)
                    await self._dispatch_alert(session, alert, websocket)

                elif msg_type == "KEYBOARD_ANOMALY":
                    detail = msg.get("detail", "unknown")
                    severity = "HIGH" if detail in ("alt_tab", "windows_key") else "MEDIUM"
                    alert = session.alert_engine.add_alert(
                        "KEYBOARD_ANOMALY", severity,
                        extra={"detail": detail, "rate": msg.get("rate", 0)}
                    )
                    if alert:
                        self._persist_violation(session, alert)
                    await self._dispatch_alert(session, alert, websocket)

                elif msg_type == "AUDIO_ANOMALY":
                    level = msg.get("level", 0)
                    alert = session.alert_engine.add_alert(
                        "AUDIO_ANOMALY", "MEDIUM",
                        extra={"level": level}
                    )
                    if alert:
                        self._persist_violation(session, alert)
                    await self._dispatch_alert(session, alert, websocket)

                elif msg_type == "CANDIDATE_INFO":
                    session.candidate_name = msg.get("name", candidate_id)
                    session.exam_id = msg.get("exam_id")
                    await self._broadcast_to_proctors({
                        "type": "SCORE_UPDATE",
                        **session.to_status_dict(),
                    })
                    logger.info("Candidate info: %s → exam %s", session.candidate_name, session.exam_id)

                elif msg_type == "HEARTBEAT":
                    await websocket.send_json({"type": "HEARTBEAT_ACK"})

        except WebSocketDisconnect:
            logger.info("Student disconnected: %s", candidate_id)
        except Exception:
            logger.exception("Student handler error for %s", candidate_id)
        finally:
            session.is_active = False
            await self._broadcast_to_proctors(
                {"type": "CANDIDATE_LEFT", "candidate_id": candidate_id}
            )

    def _persist_violation(self, session: CandidateSession, alert: dict):
        """Persist violation to MongoDB for report generation."""
        if not alert or not session.exam_id:
            return
        try:
            db = get_db()
            if db is not None:
                Violation.create(
                    db,
                    exam_id=session.exam_id,
                    type=alert.get("type", "UNKNOWN"),
                    severity=alert.get("severity", "LOW"),
                    risk_score=alert.get("score", 0),
                    snapshot_path=alert.get("snapshot_url"),
                )
        except Exception as e:
            logger.error("Failed to persist violation: %s", e)

    async def _process_frame(self, session: CandidateSession, frame_b64: str, websocket: WebSocket):
        # Use lock to prevent concurrent frame processing for same session
        async with session._processing_lock:
            try:
                img_data = base64.b64decode(frame_b64.split(",")[-1])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    return

                # Store downscaled frame for live relay to proctors
                h, w = frame.shape[:2]
                if w > 320:
                    scale = 320 / w
                    small = cv2.resize(frame, (320, int(h * scale)))
                else:
                    small = frame
                _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 50])
                session.latest_frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                # Relay frame to proctors (every frame for live feed)
                await self._broadcast_to_proctors({
                    "type": "LIVE_FRAME",
                    "candidate_id": session.candidate_id,
                    "frame": session.latest_frame_b64,
                })

                alerts = []

                # 1. Person + phone detection (YOLOv8)
                detection = self.person_counter.detect_objects(frame)
                person_count = detection["person_count"]
                phone_detected = detection["phone_detected"]

                if person_count == 0:
                    alerts.append(session.alert_engine.add_alert("NO_FACE", "HIGH"))
                elif person_count > 1:
                    alerts.append(session.alert_engine.add_alert("MULTIPLE_PERSONS", "HIGH"))

                if phone_detected:
                    alerts.append(session.alert_engine.add_alert("PHONE_DETECTED", "HIGH"))

                # 2. Face recognition
                if session.reference_encoding is not None and person_count == 1:
                    face_result = self.face_detector.verify(frame, session.reference_encoding)
                    if face_result == "MISMATCH":
                        alerts.append(session.alert_engine.add_alert("FACE_MISMATCH", "HIGH"))
                    elif face_result == "NO_FACE":
                        alerts.append(session.alert_engine.add_alert("NO_FACE", "HIGH"))

                # Process shared FaceMesh once for both eye tracking and head pose
                self.shared_facemesh.process(frame)

                # 3. Eye tracking
                gaze = self.eye_tracker.get_gaze(frame)
                if gaze and gaze != "CENTER":
                    now = time.time()
                    if session.gaze_off_center_since is None:
                        session.gaze_off_center_since = now
                    elapsed = now - session.gaze_off_center_since
                    if elapsed >= 3:
                        sev = "MEDIUM" if elapsed >= 5 else "LOW"
                        alerts.append(session.alert_engine.add_alert(f"GAZE_{gaze}", sev))
                else:
                    session.gaze_off_center_since = None

                # 4. Head pose
                head_result = self.head_pose.estimate(frame)
                if head_result:
                    yaw, pitch, _ = head_result
                    if abs(yaw) > 30 or abs(pitch) > 20:
                        alerts.append(session.alert_engine.add_alert("HEAD_POSE", "MEDIUM"))

                # Score decay
                session.alert_engine.decay(time.time() - session.last_frame_time)
                session.last_frame_time = time.time()

                # Dispatch non-None alerts
                for alert in alerts:
                    if alert:
                        if alert.get("severity") == "HIGH":
                            snap_path = save_snapshot(frame, session.candidate_id)
                            alert["snapshot_url"] = f"/static/snapshots/{snap_path}"
                        # Persist to database
                        self._persist_violation(session, alert)
                        await self._dispatch_alert(session, alert, websocket)

                # Always push score update
                await self._broadcast_to_proctors({
                    "type": "SCORE_UPDATE",
                    **session.to_status_dict(),
                })

            except Exception as e:
                logger.error("Frame processing error for %s: %s", session.candidate_id, e, exc_info=True)

    async def _dispatch_alert(self, session: CandidateSession, alert: dict, student_ws: WebSocket):
        if not alert:
            return
        try:
            await student_ws.send_json({"type": "ALERT", **alert})
        except Exception as e:
            logger.warning("Failed to send alert to student %s: %s", session.candidate_id, e)
        await self._broadcast_to_proctors({"type": "ALERT", **alert})

    async def _broadcast_to_proctors(self, data: dict):
        dead = set()
        for ws in self.proctor_sockets:
            try:
                await ws.send_json(data)
            except Exception:
                logger.debug("Dropping dead proctor connection", exc_info=True)
                dead.add(ws)
        if dead:
            logger.info("Removing %d dead proctor connections", len(dead))
        self.proctor_sockets -= dead

    def get_session(self, candidate_id: str) -> CandidateSession | None:
        return self.student_sessions.get(candidate_id)
