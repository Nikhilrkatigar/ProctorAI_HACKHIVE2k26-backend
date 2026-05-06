from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str
    role: str = "student"   # 'student' | 'proctor'

class TokenResponse(BaseModel):
    access_token: str
    role: str
    name: str


# ── Candidate ─────────────────────────────────────────────────────────────────
class CandidateCreate(BaseModel):
    name: str
    email: str

class CandidateOut(BaseModel):
    id: str
    name: str
    email: str

    class Config:
        from_attributes = True


# ── Exam ──────────────────────────────────────────────────────────────────────
class ExamStartRequest(BaseModel):
    candidate_id: str
    candidate_name: str
    exam_title: str = "HACKHIVE-2k26 AI/ML Track"

class ExamEndRequest(BaseModel):
    exam_id: str

class ExamStatusResponse(BaseModel):
    exam_id: str
    candidate_id: str
    candidate_name: str
    status: str
    start_time: str
    risk_score: int
    violations: int


# ── Violations ────────────────────────────────────────────────────────────────
class ViolationOut(BaseModel):
    id: str
    type: str
    severity: str
    risk_score: int
    timestamp: datetime
    snapshot_path: Optional[str]

    class Config:
        from_attributes = True


# ── Report ────────────────────────────────────────────────────────────────────
class ReportResponse(BaseModel):
    exam_id: str
    candidate_name: str
    start_time: str
    end_time: Optional[str]
    duration_minutes: float
    total_violations: int
    highest_risk_score: int
    violations_by_type: dict
    violations: List[ViolationOut]


# ── WebSocket Payloads ────────────────────────────────────────────────────────
class AlertPayload(BaseModel):
    candidate_id: str
    type: str
    severity: str
    score: int
    timestamp: str
    snapshot_url: Optional[str] = None
