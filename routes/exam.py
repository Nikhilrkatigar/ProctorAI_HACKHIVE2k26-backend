import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException

from models.database import get_db, Exam, Violation
from models.schemas import ExamStartRequest, ExamEndRequest, ExamStatusResponse
from routes.auth import get_current_user

router = APIRouter()


@router.post("/start")
def start_exam(req: ExamStartRequest, db=Depends(get_db), user=Depends(get_current_user)):
    exam_id = str(uuid.uuid4())
    Exam.create(db,
        id=exam_id,
        candidate_id=req.candidate_id,
        candidate_name=req.candidate_name,
        exam_title=req.exam_title,
    )
    return {
        "exam_id": exam_id,
        "candidate_id": req.candidate_id,
        "status": "ACTIVE",
        "message": "Exam started. Capture reference face via WebSocket.",
    }


@router.post("/end")
def end_exam(req: ExamEndRequest, db=Depends(get_db), user=Depends(get_current_user)):
    exam = Exam.find_by_id(db, req.exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    
    Exam.update_status(db, req.exam_id, "COMPLETED", datetime.now(timezone.utc))
    return {"exam_id": req.exam_id, "status": "COMPLETED", "report_url": f"/report/{req.exam_id}/pdf"}


@router.get("/{exam_id}/status")
def exam_status(exam_id: str, db=Depends(get_db), user=Depends(get_current_user)):
    exam = Exam.find_by_id(db, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    
    violations_count = Violation.count_by_exam(db, exam_id)
    return {
        "exam_id": exam_id,
        "candidate_name": exam.get("candidate_name"),
        "status": exam.get("status"),
        "start_time": exam.get("start_time").isoformat(),
        "violations": violations_count,
    }


@router.post("/{exam_id}/violation")
def log_violation(
    exam_id: str,
    v_type: str,
    severity: str,
    risk_score: int = 0,
    snapshot_path: str = None,
    db=Depends(get_db),
):
    Violation.create(db,
        exam_id=exam_id,
        type=v_type,
        severity=severity,
        risk_score=risk_score,
        snapshot_path=snapshot_path,
    )
    return {"status": "logged"}


@router.get("/list")
def list_exams(db=Depends(get_db), user=Depends(get_current_user)):
    exams = Exam.list_recent(db, 50)
    return [
        {
            "exam_id": e.get("_id"),
            "candidate_name": e.get("candidate_name"),
            "status": e.get("status"),
            "start_time": e.get("start_time").isoformat(),
        }
        for e in exams
    ]
