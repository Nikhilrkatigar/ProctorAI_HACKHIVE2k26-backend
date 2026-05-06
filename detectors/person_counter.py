import numpy as np
import cv2
import logging

logger = logging.getLogger("proctorAI.person_counter")

try:
    from ultralytics import YOLO
    _model = YOLO("yolov8n.pt")   # downloads on first run
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    _model = None

# Fallback: OpenCV HOG person detector
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# YOLOv8 COCO class IDs
PERSON_CLASS = 0
PHONE_CLASS = 67   # "cell phone" in COCO dataset


class PersonCounter:
    """Counts persons and detects phones visible in a frame."""

    def count(self, frame: np.ndarray) -> int:
        if YOLO_AVAILABLE and _model is not None:
            return self._yolo_count(frame)
        return self._hog_count(frame)

    def detect_objects(self, frame: np.ndarray) -> dict:
        """
        Returns dict with:
          - person_count: int
          - phone_detected: bool
        """
        if YOLO_AVAILABLE and _model is not None:
            return self._yolo_detect(frame)
        return {"person_count": self._hog_count(frame), "phone_detected": False}

    # ------------------------------------------------------------------
    def _yolo_detect(self, frame: np.ndarray) -> dict:
        """Full detection: persons + phones."""
        results = _model(frame, classes=[PERSON_CLASS, PHONE_CLASS], verbose=False)
        person_count = 0
        phone_detected = False
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == PERSON_CLASS:
                    person_count += 1
                elif cls == PHONE_CLASS:
                    phone_detected = True
        return {"person_count": person_count, "phone_detected": phone_detected}

    # ------------------------------------------------------------------
    def _yolo_count(self, frame: np.ndarray) -> int:
        results = _model(frame, classes=[PERSON_CLASS], verbose=False)
        count = 0
        for r in results:
            count += len(r.boxes)
        return count

    # ------------------------------------------------------------------
    def _hog_count(self, frame: np.ndarray) -> int:
        small = cv2.resize(frame, (320, 240))
        rects, _ = _hog.detectMultiScale(
            small, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        return len(rects)
