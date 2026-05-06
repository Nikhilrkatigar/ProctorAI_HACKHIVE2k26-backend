import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from models.database import get_db
from routes.auth import get_current_user

logger = logging.getLogger("proctorAI.questions")
router = APIRouter()


class QuestionCreate(BaseModel):
    type: str = "shortanswer"   # 'mcq' | 'shortanswer'
    question: str
    answer: Optional[str] = None
    options: Optional[List[str]] = None
    correctAnswer: Optional[int] = None


class QuestionOut(BaseModel):
    id: str
    type: str
    question: str
    answer: Optional[str] = None
    options: Optional[List[str]] = None
    correctAnswer: Optional[int] = None
    createdAt: Optional[str] = None


@router.post("", response_model=QuestionOut)
def create_question(q: QuestionCreate, db=Depends(get_db), user=Depends(get_current_user)):
    """Create a question (proctor only)."""
    import uuid
    qid = str(uuid.uuid4())[:12]

    doc = {
        "_id": qid,
        "type": q.type,
        "question": q.question,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }

    if q.type == "shortanswer":
        doc["answer"] = q.answer or ""
    elif q.type == "mcq":
        doc["options"] = q.options or []
        doc["correctAnswer"] = q.correctAnswer or 0

    db.questions.insert_one(doc)
    logger.info("Created question: %s", qid)

    return QuestionOut(
        id=doc["_id"],
        type=doc["type"],
        question=doc["question"],
        answer=doc.get("answer"),
        options=doc.get("options"),
        correctAnswer=doc.get("correctAnswer"),
        createdAt=doc["createdAt"],
    )


@router.get("", response_model=List[QuestionOut])
def list_questions(db=Depends(get_db)):
    """List all questions. No auth required so students can fetch."""
    questions = list(db.questions.find().sort("createdAt", -1))
    return [
        QuestionOut(
            id=str(q["_id"]),
            type=q.get("type", "shortanswer"),
            question=q.get("question", ""),
            answer=q.get("answer"),
            options=q.get("options"),
            correctAnswer=q.get("correctAnswer"),
            createdAt=q.get("createdAt", ""),
        )
        for q in questions
    ]


@router.delete("/{question_id}")
def delete_question(question_id: str, db=Depends(get_db), user=Depends(get_current_user)):
    """Delete a question (proctor only)."""
    result = db.questions.delete_one({"_id": question_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Question not found")
    logger.info("Deleted question: %s", question_id)
    return {"status": "deleted", "id": question_id}
