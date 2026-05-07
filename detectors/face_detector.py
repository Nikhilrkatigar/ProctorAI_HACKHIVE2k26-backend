import base64
import os
import numpy as np
import cv2

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

TOLERANCE = 0.5
MIN_REFERENCE_FACE_RATIO = 0.04
MAX_REFERENCE_FACE_RATIO = 0.70
FALLBACK_DISTANCE = 0.55

# Paths to OpenCV DNN face detector (SSD ResNet-10, much more accurate than Haar)
_BASE = os.path.join(os.path.dirname(__file__), '..', 'models')
_PROTO = os.path.join(_BASE, 'deploy.prototxt')
_MODEL = os.path.join(_BASE, 'res10_300x300_ssd_iter_140000.caffemodel')

def _load_dnn_net():
    if os.path.exists(_PROTO) and os.path.exists(_MODEL):
        net = cv2.dnn.readNetFromCaffe(_PROTO, _MODEL)
        return net
    return None

_DNN_NET = _load_dnn_net()
DNN_CONFIDENCE = 0.5


def _dnn_detect_faces(frame: np.ndarray, min_confidence: float = DNN_CONFIDENCE):
    """Returns list of (x, y, w, h) bounding boxes using SSD ResNet-10."""
    if _DNN_NET is None:
        return []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False
    )
    _DNN_NET.setInput(blob)
    detections = _DNN_NET.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < min_confidence:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw > 0 and bh > 0:
            boxes.append((x1, y1, bw, bh))
    return boxes


class FaceDetector:
    """
    Handles face detection and identity verification.
    Priority: face_recognition (dlib) → OpenCV DNN SSD → Haar cascade fallback.
    """

    def __init__(self):
        self._hog = None
        if not FACE_RECOGNITION_AVAILABLE:
            # Only use Haar as last resort if DNN also unavailable
            if _DNN_NET is None:
                self._cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
            else:
                self._cascade = None
            self._hog = cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)

    # ------------------------------------------------------------------
    def encode_reference(self, frame_b64: str):
        """
        Given a base64 JPEG frame, return a face encoding.
        Returns None if no face found or face is too small/large.
        """
        frame = self._b64_to_frame(frame_b64)
        if frame is None:
            return None

        if FACE_RECOGNITION_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            if len(locs) != 1 or not self._valid_reference_face(frame, locs[0]):
                return None
            encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
            return encs[0] if encs else None

        # DNN path
        boxes = _dnn_detect_faces(frame)
        if len(boxes) != 1:
            return None
        box = max(boxes, key=lambda b: b[2] * b[3])
        if not self._valid_reference_box(frame, box):
            return None
        return self._dnn_features(frame, box)

    # ------------------------------------------------------------------
    def verify(self, frame: np.ndarray, reference_encoding) -> str:
        """Returns: 'OK', 'MISMATCH', or 'NO_FACE'"""
        if FACE_RECOGNITION_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            if not locs:
                return "NO_FACE"
            encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
            if not encs:
                return "NO_FACE"
            distances = face_recognition.face_distance(encs, reference_encoding)
            return "OK" if float(np.min(distances)) <= TOLERANCE else "MISMATCH"

        # DNN path
        boxes = _dnn_detect_faces(frame)
        if not boxes:
            return "NO_FACE"
        box = max(boxes, key=lambda b: b[2] * b[3])
        current = self._dnn_features(frame, box)
        dist = 1.0 - float(np.dot(reference_encoding, current))
        return "OK" if dist < FALLBACK_DISTANCE else "MISMATCH"

    # ------------------------------------------------------------------
    def count_faces(self, frame: np.ndarray) -> int:
        if FACE_RECOGNITION_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return len(face_recognition.face_locations(rgb, model="hog"))

        boxes = _dnn_detect_faces(frame)
        return len(boxes)

    # ------------------------------------------------------------------
    @staticmethod
    def _b64_to_frame(b64: str):
        try:
            raw = base64.b64decode(b64.split(",")[-1])
            arr = np.frombuffer(raw, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _valid_reference_face(frame: np.ndarray, loc) -> bool:
        top, right, bottom, left = loc
        return FaceDetector._valid_reference_box(frame, (left, top, right - left, bottom - top))

    @staticmethod
    def _valid_reference_box(frame: np.ndarray, box) -> bool:
        _, _, w, h = box
        frame_h, frame_w = frame.shape[:2]
        ratio = (w * h) / max(frame_w * frame_h, 1)
        return MIN_REFERENCE_FACE_RATIO <= ratio <= MAX_REFERENCE_FACE_RATIO

    def _dnn_features(self, frame: np.ndarray, box) -> np.ndarray:
        """Extract a normalized HOG feature vector from the detected face region."""
        x, y, w, h = box
        pad_x = int(w * 0.18)
        pad_y = int(h * 0.22)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)

        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 96))
        gray = cv2.equalizeHist(gray)

        hog = self._hog.compute(gray).flatten().astype(np.float32)
        norm = np.linalg.norm(hog)
        return hog / norm if norm > 0 else hog
