import hashlib
import os
import secrets
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional

from models.database import get_db, Candidate, ProctorUser
from models.schemas import LoginRequest, TokenResponse

try:
    import jwt as pyjwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except ImportError:
    pwd_context = None

logger = logging.getLogger("proctorAI.auth")
router = APIRouter()

SECRET_KEY = os.getenv("SECRET_KEY", "hackhive2k26-super-secret-key-change-me")
TOKEN_EXPIRY_HOURS = 24

# Fallback in-memory store if PyJWT not installed
_tokens: dict[str, dict] = {}


def _legacy_hash(pw: str) -> str:
    salt = "proctorAI:"
    return hashlib.sha256((salt + pw).encode()).hexdigest()


def _hash(pw: str) -> str:
    """Hash passwords with bcrypt when available, with a legacy fallback."""
    if pwd_context:
        try:
            return pwd_context.hash(pw)
        except Exception as exc:
            logger.warning("bcrypt hashing unavailable, falling back to legacy hash: %s", exc)
    return _legacy_hash(pw)


def _verify_password(password: str, stored_hash: str | None) -> bool:
    if not stored_hash:
        return False
    if stored_hash.startswith("$2") and pwd_context:
        try:
            return pwd_context.verify(password, stored_hash)
        except Exception as exc:
            logger.warning("bcrypt verify unavailable for stored hash: %s", exc)
            return False
    return stored_hash == _legacy_hash(password)


def create_token(payload: dict) -> str:
    """Create a JWT token or fallback to random hex token."""
    if JWT_AVAILABLE:
        payload["exp"] = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS)
        payload["iat"] = datetime.now(timezone.utc)
        return pyjwt.encode(payload, SECRET_KEY, algorithm="HS256")
    else:
        token = secrets.token_hex(32)
        _tokens[token] = payload
        return token


def verify_token(token: str) -> dict | None:
    """Verify and decode a token. Returns payload or None."""
    if JWT_AVAILABLE:
        try:
            return pyjwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except Exception:
            return None
    else:
        return _tokens.get(token)


def get_current_user(authorization: Optional[str] = Header(None)):
    """FastAPI dependency for protected routes."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    # Support "Bearer <token>" format
    token = authorization.replace("Bearer ", "").strip()
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


def seed_demo_users(db):
    """Create demo proctor + student on first run."""
    if not ProctorUser.find_by_email(db, "proctor@hackhive.ai"):
        ProctorUser.create(db,
            name="Dr. Proctor",
            email="proctor@hackhive.ai",
            password_hash=_hash("proctor123"),
        )
        logger.info("Created demo proctor account")
    
    if not Candidate.find_by_email(db, "alice@hackhive.ai"):
        Candidate.create(db,
            name="Alice Demo",
            email="alice@hackhive.ai",
            password_hash=_hash("student123"),
        )
        logger.info("Created demo student account")


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db=Depends(get_db)):
    seed_demo_users(db)

    identifier = req.email.strip()

    # Proctors sign in with email + password.
    user = ProctorUser.find_by_email(db, identifier)
    if user and _verify_password(req.password, user.get("password_hash")):
        token = create_token({"role": "proctor", "name": user.get("name"), "id": str(user.get("_id"))})
        logger.info("Proctor login: %s", identifier)
        return TokenResponse(access_token=token, role="proctor", name=user.get("name"))

    # Students created from the proctor dashboard sign in with student ID + password.
    user = db.student_accounts.find_one({"studentId": identifier}) or db.student_accounts.find_one({"_id": identifier})
    if user and _verify_password(req.password, user.get("password_hash")):
        token = create_token({"role": "student", "name": user.get("name"), "id": user.get("studentId")})
        logger.info("Student login by ID: %s", identifier)
        return TokenResponse(access_token=token, role="student", name=user.get("name"))

    # Legacy/demo student accounts sign in with email + password.
    user = Candidate.find_by_email(db, identifier)
    if user and _verify_password(req.password, user.get("password_hash")):
        token = create_token({"role": "student", "name": user.get("name"), "id": str(user.get("_id"))})
        logger.info("Student login by email: %s", identifier)
        return TokenResponse(access_token=token, role="student", name=user.get("name"))

    raise HTTPException(status_code=401, detail="Invalid ID/email or password")


@router.get("/me")
def me(user=Depends(get_current_user)):
    return user
