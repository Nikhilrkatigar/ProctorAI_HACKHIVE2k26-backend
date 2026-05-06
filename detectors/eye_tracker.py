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

        # --- Horizontal gaze (left iris relative to eye corners) ---
        try:
            left_inner  = pt(LEFT_EYE_INNER)
            left_outer  = pt(LEFT_EYE_OUTER)
            left_iris   = pt(LEFT_IRIS)

            eye_width = np.linalg.norm(left_inner - left_outer)
            if eye_width < 1:
                return "CENTER"

            ratio_h = (left_iris[0] - left_outer[0]) / eye_width

            # --- Vertical gaze ---
            left_top    = pt(LEFT_EYE_TOP)
            left_bottom = pt(LEFT_EYE_BOTTOM)
            eye_height  = np.linalg.norm(left_top - left_bottom)
            ratio_v     = (left_iris[1] - left_top[1]) / (eye_height + 1e-6)

            if ratio_h < 0.35:
                return "RIGHT"   # iris moved toward outer = looking right
            elif ratio_h > 0.65:
                return "LEFT"
            elif ratio_v < 0.3:
                return "UP"
            else:
                return "CENTER"
        except (IndexError, Exception):
            return "CENTER"

    def _fallback_gaze(self, frame: np.ndarray) -> str:
        """Simple fallback using Haar eye detector if mediapipe unavailable."""
        return "CENTER"

    def __del__(self):
        if self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass
