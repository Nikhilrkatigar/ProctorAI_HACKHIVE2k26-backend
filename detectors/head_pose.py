import numpy as np
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# 3D model points for generic human face
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# MediaPipe FaceMesh landmark indices matching the 6 model points above
LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]


class HeadPoseEstimator:
    def __init__(self, shared_facemesh=None):
        self._shared = shared_facemesh
        self._face_mesh = None
        # Only create own FaceMesh if no shared one provided
        if not shared_facemesh and MEDIAPIPE_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def estimate(self, frame: np.ndarray):
        """
        Returns (yaw, pitch, roll) in degrees, or None if no face detected.
        yaw   > 0 → looking right, < 0 → looking left
        pitch > 0 → looking up,   < 0 → looking down
        """
        if not MEDIAPIPE_AVAILABLE:
            return None

        h, w = frame.shape[:2]

        # Use shared facemesh landmarks if available
        if self._shared:
            lm = self._shared.landmarks
            if lm is None:
                return None
        else:
            if self._face_mesh is None:
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            lm = results.multi_face_landmarks[0].landmark

        image_points = np.array([
            (lm[i].x * w, lm[i].y * h)
            for i in LANDMARK_INDICES
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat([rotation_matrix, np.zeros((3, 1))])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler[0])
        yaw   = float(euler[1])
        roll  = float(euler[2])

        return yaw, pitch, roll

    def __del__(self):
        if self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass
