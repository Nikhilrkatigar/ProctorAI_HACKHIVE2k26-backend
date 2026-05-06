import base64
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
FALLBACK_DISTANCE = 0.62


class FaceDetector:
    """
    Handles face detection and identity verification.
    Falls back to OpenCV Haar cascade if face_recognition is not installed.
    """

    def __init__(self):
        self._cascade = None
        self._hog = None
        if not FACE_RECOGNITION_AVAILABLE:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._hog = cv2.HOGDescriptor(
                (96, 96),
                (16, 16),
                (8, 8),
                (8, 8),
                9,
            )

    # ------------------------------------------------------------------
    def encode_reference(self, frame_b64: str):
        """
        Given a base64 JPEG frame, return the face encoding (128-d vector).
        Returns None if no face found.
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
        else:
            # Fallback: store the mean pixel value as a dummy encoding
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.08, 5, minSize=(80, 80))
            if len(faces) != 1:
                return None
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            if not self._valid_reference_box(frame, (x, y, w, h)):
                return None
            return self._fallback_features(frame, (x, y, w, h))

    # ------------------------------------------------------------------
    def verify(self, frame: np.ndarray, reference_encoding) -> str:
        """
        Returns: 'OK', 'MISMATCH', or 'NO_FACE'
        """
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
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.08, 5, minSize=(50, 50))
            if len(faces) == 0:
                return "NO_FACE"
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            current = self._fallback_features(frame, (x, y, w, h))
            dist = 1.0 - float(np.dot(reference_encoding, current))
            return "OK" if dist < FALLBACK_DISTANCE else "MISMATCH"

    # ------------------------------------------------------------------
    def count_faces(self, frame: np.ndarray) -> int:
        if FACE_RECOGNITION_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            return len(locs)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.08, 5, minSize=(50, 50))
            return len(faces)

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

    def _fallback_features(self, frame: np.ndarray, box) -> np.ndarray:
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
        if norm == 0:
            return hog
        return hog / norm
