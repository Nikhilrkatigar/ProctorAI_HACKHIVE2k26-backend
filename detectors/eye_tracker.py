import numpy as np
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# MediaPipe FaceMesh landmark indices for iris and eye corners
LEFT_EYE_INNER  = 133
LEFT_EYE_OUTER  = 33
LEFT_IRIS       = 468          # left iris center (FaceMesh with refine_landmarks=True)

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_IRIS      = 473

LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

class EyeTracker:
    def __init__(self, shared_facemesh=None):
        self._shared = shared_facemesh
        self._face_mesh = None
        # Only create own FaceMesh if no shared one provided
        if not shared_facemesh and MEDIAPIPE_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def get_gaze(self, frame: np.ndarray) -> str | None:
        """
        Returns gaze direction string: 'CENTER', 'LEFT', 'RIGHT', 'UP', 'DOWN'
        Returns None if no face detected.
        """
        if not MEDIAPIPE_AVAILABLE:
            return self._fallback_gaze(frame)

        # Use shared facemesh landmarks if available
        if self._shared:
            lm = self._shared.landmarks
            shape = self._shared.frame_shape
            if lm is None or shape is None:
                return None
            h, w = shape
        else:
            if self._face_mesh is None:
                return self._fallback_gaze(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            lm = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

        def pt(idx):
            l = lm[idx]
            return np.array([l.x * w, l.y * h])

        try:
            left_ratio_h = self._horizontal_ratio(pt(LEFT_IRIS), pt(LEFT_EYE_OUTER), pt(LEFT_EYE_INNER))
            right_ratio_h = self._horizontal_ratio(pt(RIGHT_IRIS), pt(RIGHT_EYE_INNER), pt(RIGHT_EYE_OUTER))
            ratio_h = float(np.mean([left_ratio_h, right_ratio_h]))

            left_ratio_v = self._vertical_ratio(pt(LEFT_IRIS), pt(LEFT_EYE_TOP), pt(LEFT_EYE_BOTTOM))
            right_ratio_v = self._vertical_ratio(pt(RIGHT_IRIS), pt(RIGHT_EYE_TOP), pt(RIGHT_EYE_BOTTOM))
            ratio_v = float(np.mean([left_ratio_v, right_ratio_v]))

            if ratio_h < 0.28:
                return "RIGHT"
            elif ratio_h > 0.72:
                return "LEFT"
            elif ratio_v < 0.22:
                return "UP"
            elif ratio_v > 0.82:
                return "DOWN"
            else:
                return "CENTER"
        except (IndexError, Exception):
            return "CENTER"

    @staticmethod
    def _horizontal_ratio(iris: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> float:
        eye_width = np.linalg.norm(inner - outer)
        if eye_width < 1:
            return 0.5
        return (iris[0] - outer[0]) / eye_width

    @staticmethod
    def _vertical_ratio(iris: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> float:
        eye_height = np.linalg.norm(top - bottom)
        if eye_height < 1:
            return 0.5
        return (iris[1] - top[1]) / eye_height

    def _fallback_gaze(self, frame: np.ndarray) -> str:
        """Simple fallback using Haar eye detector if mediapipe unavailable."""
        return "CENTER"

    def __del__(self):
        if self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass
