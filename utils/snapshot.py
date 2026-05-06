import os
import time
import numpy as np
import cv2

SNAPSHOT_DIR = "static/snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def save_snapshot(frame: np.ndarray, candidate_id: str) -> str:
    """
    Saves a JPEG snapshot and returns the filename (not full path).
    Full URL: /static/snapshots/<filename>
    """
    ts = int(time.time() * 1000)
    filename = f"{candidate_id}_{ts}.jpg"
    full_path = os.path.join(SNAPSHOT_DIR, filename)

    # Resize to reduce file size
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640 / w
        frame = cv2.resize(frame, (640, int(h * scale)))

    cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return filename
