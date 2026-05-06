import base64
import numpy as np
import cv2

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

TOLERANCE = 0.5


class FaceDetector:
    """
    Handles face detection and identity verification.
    Falls back to OpenCV Haar cascade if face_recognition is not installed.
    """

    def __init__(self):
        self._cascade = None
        if not FACE_RECOGNITION_AVAILABLE:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
            encs = face_recognition.face_encodings(rgb)
            return encs[0] if encs else None
        else:
            # Fallback: store the mean pixel value as a dummy encoding
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                return None
            x, y, w, h = faces[0]
            roi = cv2.resize(frame[y:y+h, x:x+w], (64, 64))
            return roi.flatten().astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    def verify(self, frame: np.ndarray, reference_encoding) -> str:
        """
        Returns: 'OK', 'MISMATCH', or 'NO_FACE'
        """
        if FACE_RECOGNITION_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            if not encs:
                return "NO_FACE"
            match = face_recognition.compare_faces(
                [reference_encoding], encs[0], tolerance=TOLERANCE
            )
            return "OK" if match[0] else "MISMATCH"
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                return "NO_FACE"
            x, y, w, h = faces[0]
            roi = cv2.resize(frame[y:y+h, x:x+w], (64, 64))
            current = roi.flatten().astype(np.float32) / 255.0
            dist = np.linalg.norm(reference_encoding - current)
            return "OK" if dist < 50 else "MISMATCH"

    # ------------------------------------------------------------------
    def count_faces(self, frame: np.ndarray) -> int:
        if FACE_RECOGNITION_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            return len(locs)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.1, 5)
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
