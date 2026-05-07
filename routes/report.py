import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import FileResponse
from datetime import timezone
from bson import ObjectId

from models.database import get_db, Exam, Violation
from utils.pdf_report import generate_pdf
from routes.auth import get_current_user, verify_token

router = APIRouter()


def require_proctor(user: dict):
    if user.get("role") != "proctor":
        raise HTTPException(status_code=403, detail="Reports are available to proctors only")


def _build_report_data(exam_doc: dict, violations: list) -> dict:
    start = exam_doc.get('start_time')
    end = exam_doc.get('end_time') or __import__("datetime").datetime.now(timezone.utc)
    
    # Handle string ISO timestamps
    if isinstance(start, str):
        from datetime import datetime
        start = datetime.fromisoformat(start.replace('Z', '+00:00'))
    if isinstance(end, str):
        from datetime import datetime
        end = datetime.fromisoformat(end.replace('Z', '+00:00'))

    # Make both timezone-aware
    if start.tzinfo is None:
        from datetime import timezone as tz
        start = start.replace(tzinfo=tz.utc)
    if end.tzinfo is None:
        from datetime import timezone as tz
        end = end.replace(tzinfo=tz.utc)

    duration = (end - start).total_seconds() / 60

    by_type: dict[str, int] = {}
    for v in violations:
        v_type = v.get('type', 'unknown')
        by_type[v_type] = by_type.get(v_type, 0) + 1

    max_score = max((v.get('risk_score', 0) for v in violations), default=0)

    return {
        "exam_id":           exam_doc.get('_id'),
        "candidate_name":    exam_doc.get('candidate_name'),
        "exam_title":        exam_doc.get('exam_title'),
        "status":            exam_doc.get('status'),
        "start_time":        start.isoformat(),
        "end_time":          end.isoformat(),
        "duration_minutes":  round(duration, 2),
        "total_violations":  len(violations),
        "highest_risk_score":max_score,
        "violations_by_type":by_type,
        "violations": [
            {
                "id":            str(v.get('_id', '')),
                "type":          v.get('type'),
                "severity":      v.get('severity'),
                "risk_score":    v.get('risk_score'),
                "timestamp":     v.get('timestamp', '').isoformat() if hasattr(v.get('timestamp'), 'isoformat') else v.get('timestamp', ''),
                "snapshot_path": v.get('snapshot_path'),
            }
            for v in violations
        ],
    }


@router.get("/{exam_id}/json")
def report_json(
    exam_id: str,
    db=Depends(get_db),
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
):
    # Support Authorization header or fallback to ?token=...
    auth_value = authorization or token
    if not auth_value:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token_val = auth_value.replace("Bearer ", "").strip()
    user = verify_token(token_val)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    require_proctor(user)
    exam_doc = db.exams.find_one({"_id": exam_id})
    if not exam_doc:
        raise HTTPException(status_code=404, detail="Exam not found")
    violations = list(db.violations.find({"exam_id": exam_id}))
    return _build_report_data(exam_doc, violations)


@router.get("/{exam_id}/pdf")
def report_pdf(
    exam_id: str,
    db=Depends(get_db),
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
):
    # Support Authorization header or fallback to ?token=...
    auth_value = authorization or token
    if not auth_value:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token_val = auth_value.replace("Bearer ", "").strip()
    user = verify_token(token_val)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    require_proctor(user)
    exam_doc = db.exams.find_one({"_id": exam_id})
    if not exam_doc:
        raise HTTPException(status_code=404, detail="Exam not found")
    violations = list(db.violations.find({"exam_id": exam_id}))
    data = _build_report_data(exam_doc, violations)

    pdf_path = generate_pdf(data)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"ProctorAI_Report_{exam_id[:8]}.pdf",
    )
