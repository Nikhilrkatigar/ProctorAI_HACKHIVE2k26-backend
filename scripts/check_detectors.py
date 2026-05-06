import importlib

print('Checking detector libraries...')

modules = [
    ('mediapipe', 'MediaPipe (FaceMesh)'),
    ('ultralytics', 'YOLOv8 (ultralytics)'),
    ('face_recognition', 'face_recognition (dlib)'),
    ('cv2', 'OpenCV (cv2)'),
]

for mod, name in modules:
    try:
        importlib.import_module(mod)
        print(f'✓ {name} available ({mod})')
    except Exception as e:
        print(f'✗ {name} missing or failed to import ({mod}) — {e}')

# Also check internal availability flags
print('\nChecking in-project detector availability flags...')
try:
    from detectors import shared_facemesh, person_counter, face_detector, eye_tracker
    print('SharedFaceMesh MEDIAPIPE_AVAILABLE =', getattr(shared_facemesh, 'MEDIAPIPE_AVAILABLE', 'N/A'))
    print('PersonCounter YOLO_AVAILABLE =', getattr(person_counter, '_model', None) is not None)
    print('FaceDetector FACE_RECOGNITION_AVAILABLE =', getattr(face_detector, 'FACE_RECOGNITION_AVAILABLE', 'N/A'))
    print('EyeTracker MEDIAPIPE_AVAILABLE =', getattr(eye_tracker, 'MEDIAPIPE_AVAILABLE', 'N/A'))
except Exception as e:
    print('Failed to import detectors module flags:', e)
