import os
import sys
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from models.database import connect_db
from routes import exam, report, auth, students, questions
from websocket_handler import ProctorWebSocketManager

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("proctorAI")

# Initialize MongoDB
connect_db()

ws_manager = ProctorWebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("static/snapshots", exist_ok=True)
    os.makedirs("static/reports", exist_ok=True)
    logger.info("✓ ProctorAI backend started")
    yield
    logger.info("ProctorAI backend shutting down")

app = FastAPI(
    title="ProctorAI — Real-Time Exam Integrity System",
    description="AI-powered online proctoring for HACKHIVE-2k26",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — use env variable with fallback
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth.router,     prefix="/auth",     tags=["Auth"])
app.include_router(exam.router,     prefix="/exam",     tags=["Exam"])
app.include_router(report.router,   prefix="/report",   tags=["Report"])
app.include_router(students.router, prefix="/students", tags=["Students"])
app.include_router(questions.router, prefix="/questions", tags=["Questions"])

@app.websocket("/ws/{candidate_id}")
async def websocket_endpoint(websocket: WebSocket, candidate_id: str, role: str = "student"):
    await ws_manager.handle(websocket, candidate_id, role)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ProctorAI"}

@app.get("/")
async def root():
    return {
        "service": "ProctorAI",
        "docs": "/docs",
        "websocket": "ws://{host}/ws/{candidate_id}",
    }
