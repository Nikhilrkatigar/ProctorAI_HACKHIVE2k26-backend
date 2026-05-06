import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from models.database import get_db
from routes.auth import get_current_user, _hash

logger = logging.getLogger("proctorAI.students")
router = APIRouter()


class StudentAccount(BaseModel):
    name: str
    studentId: str
    password: str


class StudentAccountResponse(BaseModel):
    id: str
    name: str
    studentId: str
    password: str          # Shown once at creation for the proctor's credential sheet
    createdAt: str


@router.post("/accounts", response_model=StudentAccountResponse)
def create_student_account(
    account: StudentAccount,
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    """Create a new student account and store in MongoDB."""
    doc = {
        "_id": account.studentId,
        "name": account.name,
        "studentId": account.studentId,
        "password_hash": _hash(account.password),
        "password_display": account.password,   # Kept for proctor credential sheet
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    try:
        db.student_accounts.insert_one(doc)
    except Exception as e:
        # If duplicate, update it
        if "E11000" in str(e) or "duplicate" in str(e).lower():
            db.student_accounts.replace_one({"_id": account.studentId}, doc)
        else:
            raise HTTPException(status_code=400, detail=str(e))

    return StudentAccountResponse(
        id=doc["_id"],
        name=doc["name"],
        studentId=doc["studentId"],
        password=doc["password_display"],
        createdAt=doc["createdAt"],
    )


@router.get("/accounts", response_model=list[StudentAccountResponse])
def list_student_accounts(db=Depends(get_db), user=Depends(get_current_user)):
    """Retrieve all student accounts from MongoDB."""
    accounts = list(db.student_accounts.find().sort("createdAt", -1))
    return [
        StudentAccountResponse(
            id=acc["_id"],
            name=acc.get("name", ""),
            studentId=acc.get("studentId", ""),
            password=acc.get("password_display", acc.get("password", "")),
            createdAt=acc.get("createdAt", ""),
        )
        for acc in accounts
    ]


@router.delete("/accounts/{student_id}")
def delete_student_account(
    student_id: str,
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    """Delete a student account from MongoDB."""
    result = db.student_accounts.delete_one({"_id": student_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Student account not found")
    return {"status": "deleted", "student_id": student_id}
