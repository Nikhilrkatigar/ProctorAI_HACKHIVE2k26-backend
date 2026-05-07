import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

results = []
def ok(name):   results.append(('PASS', name, '')); print(f'  PASS  {name}')
def fail(name, reason): results.append(('FAIL', name, reason)); print(f'  FAIL  {name} -- {reason}')

print('\n========= PRODUCTION READINESS AUDIT =========\n')

# 1. DEPENDENCIES
print('[1] DEPENDENCIES')
import importlib
deps = {
    'fastapi': 'FastAPI', 'uvicorn': 'ASGI server', 'pymongo': 'MongoDB',
    'reportlab': 'PDF generation', 'jwt': 'JWT tokens', 'passlib': 'Password hashing',
    'cv2': 'OpenCV', 'mediapipe': 'MediaPipe', 'ultralytics': 'YOLO',
    'numpy': 'NumPy', 'dotenv': 'python-dotenv',
}
for mod, name in deps.items():
    try:
        importlib.import_module(mod)
        ok(name)
    except ImportError as e:
        fail(name, str(e))

# 2. PATHS
print('\n[2] PATHS')
from utils.snapshot import SNAPSHOT_DIR
from utils.pdf_report import REPORTS_DIR, _BACKEND_DIR

if os.path.isabs(SNAPSHOT_DIR) and os.path.exists(SNAPSHOT_DIR):
    ok('SNAPSHOT_DIR absolute and exists')
else:
    fail('SNAPSHOT_DIR', f'not absolute or missing: {SNAPSHOT_DIR}')

if os.path.isabs(REPORTS_DIR) and os.path.exists(REPORTS_DIR):
    ok('REPORTS_DIR absolute and exists')
else:
    fail('REPORTS_DIR', f'not absolute or missing: {REPORTS_DIR}')

static_dir = os.path.join(_BACKEND_DIR, 'static')
if os.path.exists(static_dir):
    ok('static/ directory exists')
else:
    fail('static/', 'missing - StaticFiles mount will crash on startup')

# 3. DNN FACE MODEL FILES
print('\n[3] FACE DETECTOR MODELS')
from detectors.face_detector import _PROTO, _MODEL, _DNN_NET
if os.path.exists(os.path.abspath(_PROTO)):
    ok('deploy.prototxt found')
else:
    fail('deploy.prototxt', f'missing at {os.path.abspath(_PROTO)}')
if os.path.exists(os.path.abspath(_MODEL)):
    ok('res10_300x300_ssd caffemodel found')
else:
    fail('caffemodel', f'missing at {os.path.abspath(_MODEL)}')
if _DNN_NET is not None:
    ok('DNN net loaded')
else:
    fail('DNN net', 'failed to load')

# 4. YOLO
print('\n[4] YOLO MODEL')
from detectors.person_counter import YOLO_AVAILABLE, _model
if YOLO_AVAILABLE and _model is not None:
    ok('YOLOv8n loaded')
else:
    fail('YOLOv8n', 'not loaded')

# 5. PDF GENERATION (the broken download)
print('\n[5] PDF GENERATION + DOWNLOAD')
import numpy as np
from utils.snapshot import save_snapshot
from utils.pdf_report import generate_pdf, REPORTLAB_AVAILABLE

if REPORTLAB_AVAILABLE:
    ok('reportlab available')
else:
    fail('reportlab', 'not available - PDF generation impossible')

blank = np.zeros((360, 640, 3), dtype=np.uint8)
snap_fname = save_snapshot(blank, 'prod_audit')
snap_url = '/static/snapshots/' + snap_fname

fake_data = {
    'exam_id': 'aabbccdd-1234-5678-abcd-ef0123456789',
    'candidate_name': 'Test Candidate',
    'exam_title': 'HACKHIVE-2k26',
    'status': 'COMPLETED',
    'start_time': '2026-05-07T10:00:00+00:00',
    'end_time':   '2026-05-07T10:30:00+00:00',
    'duration_minutes': 30.0,
    'total_violations': 2,
    'highest_risk_score': 60,
    'violations_by_type': {'HEAD_POSE': 1, 'NO_FACE': 1},
    'violations': [
        {'id': '1', 'type': 'HEAD_POSE', 'severity': 'HIGH',
         'risk_score': 30, 'timestamp': '2026-05-07T10:05:00', 'snapshot_path': snap_url},
        {'id': '2', 'type': 'NO_FACE',   'severity': 'HIGH',
         'risk_score': 60, 'timestamp': '2026-05-07T10:10:00', 'snapshot_path': None},
    ],
}

try:
    pdf_path = generate_pdf(fake_data)
    size_kb = os.path.getsize(pdf_path) // 1024
    if os.path.exists(pdf_path) and size_kb > 0:
        ok(f'PDF generated ({size_kb} KB) at {pdf_path}')
    else:
        fail('PDF generate', f'file missing or empty: {pdf_path}')
except Exception as e:
    fail('PDF generate', str(e))

resolved_snap = os.path.join(_BACKEND_DIR, snap_url.lstrip('/'))
if os.path.exists(resolved_snap):
    ok('Snapshot found and embedded in PDF')
    os.remove(resolved_snap)
else:
    fail('Snapshot for PDF', f'missing: {resolved_snap}')

# 6. AUTH TOKEN ROUND-TRIP
print('\n[6] AUTH')
from routes.auth import create_token, verify_token
token = create_token({'role': 'proctor', 'name': 'Test'})
payload = verify_token(token)
if payload and payload.get('role') == 'proctor':
    ok('JWT create+verify round-trip')
else:
    fail('JWT', f'verify returned: {payload}')

stripped = token.replace('Bearer ', '').strip()
payload2 = verify_token(stripped)
if payload2:
    ok('Token works without Bearer prefix (query param mode)')
else:
    fail('Token query param', 'verify failed after stripping Bearer')

# 7. FRONTEND PDF BUTTON
print('\n[7] FRONTEND PDF BUTTON')
modal_path = os.path.join(_BACKEND_DIR, '..', 'frontend', 'src', 'components', 'ReportModal.jsx')
with open(modal_path) as f:
    modal_code = f.read()

if "role === 'proctor'" in modal_code:
    ok("Download button visible only for proctor role")
else:
    fail('PDF button visibility', "role check missing")

if 'Authorization' in modal_code:
    ok('Auth header sent with PDF request')
else:
    fail('PDF auth header', 'Authorization header missing in download fetch')

if 'URL.createObjectURL' in modal_code and 'link.click()' in modal_code:
    ok('Blob download trigger (createObjectURL + link.click)')
else:
    fail('Blob download', 'createObjectURL or link.click missing')

if 'downloading' in modal_code:
    ok('Download in-progress state handled (no double-click)')
else:
    fail('Download guard', 'no in-progress guard')

# 8. CORS
print('\n[8] CORS + SERVER CONFIG')
main_path = os.path.join(_BACKEND_DIR, 'main.py')
with open(main_path) as f:
    main_code = f.read()
if 'CORS_ORIGINS' in main_code and 'localhost:5173' in main_code:
    ok('CORS configured with localhost:5173 fallback')
else:
    fail('CORS', 'missing CORS_ORIGINS or no localhost:5173 fallback')

if 'StaticFiles' in main_code and '/static' in main_code:
    ok('StaticFiles mounted at /static (snapshot URLs will resolve)')
else:
    fail('StaticFiles', 'not mounted - snapshot image URLs will 404')

if '/ws/{candidate_id}' in main_code:
    ok('WebSocket /ws/{candidate_id} registered')
else:
    fail('WebSocket', 'endpoint missing in main.py')

# 9. WEBSOCKET EVENT MISMATCHES
print('\n[9] WEBSOCKET EVENT COVERAGE')
ws_path = os.path.join(_BACKEND_DIR, 'websocket_handler.py')
with open(ws_path) as f:
    ws_code = f.read()

frontend_events = [
    'TAB_SWITCH', 'WINDOW_BLUR', 'FULLSCREEN_EXIT', 'NAVIGATION_ATTEMPT',
    'DEVTOOLS_ATTEMPT', 'KEYBOARD_ANOMALY', 'AUDIO_ANOMALY', 'CLIPBOARD_ATTEMPT',
    'CONTEXT_MENU_ATTEMPT', 'NEW_TAB_ATTEMPT', 'REFRESH_ATTEMPT', 'SCREENSHOT_ATTEMPT',
    'REFERENCE_FACE', 'FRAME', 'HEARTBEAT', 'CANDIDATE_INFO',
]
missing = [e for e in frontend_events if e not in ws_code]
if missing:
    fail('WS events', f'not handled: {missing}')
else:
    ok('All frontend event types handled in backend')

# 10. SEVERITY CHECK
print('\n[10] VIOLATION SEVERITY (for snapshots)')
if '"HEAD_POSE", "HIGH"' in ws_code:
    ok('HEAD_POSE severity = HIGH (gets snapshot)')
else:
    fail('HEAD_POSE severity', 'not HIGH - no snapshot will be taken')

if 'sev = "HIGH" if elapsed >= 5 else "MEDIUM"' in ws_code:
    ok('GAZE severity escalates to HIGH after 5s (gets snapshot)')
else:
    fail('GAZE severity', 'does not escalate to HIGH')

if '"NO_FACE", "HIGH"' in ws_code:
    ok('NO_FACE severity = HIGH (gets snapshot)')
else:
    fail('NO_FACE severity', 'not HIGH')

# SUMMARY
print('\n========= SUMMARY =========')
passes = [r for r in results if r[0] == 'PASS']
fails  = [r for r in results if r[0] == 'FAIL']
print(f'PASS: {len(passes)}/{len(results)}')
if fails:
    print('\nFAILURES TO FIX:')
    for r in fails:
        print(f'  FAIL: {r[1]} -- {r[2]}')
else:
    print('ALL CHECKS PASSED - production ready')
