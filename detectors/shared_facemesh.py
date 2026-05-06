"""
Shared FaceMesh manager — processes once, provides landmarks to both
EyeTracker and HeadPoseEstimator, eliminating duplicate MediaPipe instances.
"""
import numpy as np
import cv2
import logging

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger("proctorAI.facemesh")


class SharedFaceMesh:
    """Singleton-style shared FaceMesh processor."""

    def __init__(self):
        self._face_mesh = None
        self._last_landmarks = None
        self._last_frame_shape = None

        if MEDIAPIPE_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,   # Needed for iris landmarks (468+)
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("Shared FaceMesh initialized")

    def process(self, frame: np.ndarray):
        """
        Process a frame and cache the landmarks.
        Returns the landmarks list or None.
        """
        if not MEDIAPIPE_AVAILABLE or self._face_mesh is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._last_landmarks = None
            self._last_frame_shape = None
            return None

        self._last_landmarks = results.multi_face_landmarks[0].landmark
        self._last_frame_shape = frame.shape[:2]  # (h, w)
        return self._last_landmarks

    @property
    def landmarks(self):
        return self._last_landmarks

    @property
    def frame_shape(self):
        return self._last_frame_shape

    def __del__(self):
        if self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass
