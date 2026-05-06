import os
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure
from datetime import datetime, timezone
from contextlib import contextmanager

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "proctorAI")

client = None
db = None

def connect_db():
    """Initialize MongoDB connection and create indexes."""
    global client, db
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        db = client[DB_NAME]
        
        # Create collections and indexes
        if "candidates" not in db.list_collection_names():
            db.create_collection("candidates")
        db.candidates.create_index([("email", ASCENDING)], unique=True)
        
        if "exams" not in db.list_collection_names():
            db.create_collection("exams")
        db.exams.create_index([("candidate_id", ASCENDING)])
        db.exams.create_index([("start_time", ASCENDING)])
        
        if "violations" not in db.list_collection_names():
            db.create_collection("violations")
        db.violations.create_index([("exam_id", ASCENDING)])
        db.violations.create_index([("timestamp", ASCENDING)])
        
        if "proctor_users" not in db.list_collection_names():
            db.create_collection("proctor_users")
        db.proctor_users.create_index([("email", ASCENDING)], unique=True)
        
        if "student_accounts" not in db.list_collection_names():
            db.create_collection("student_accounts")
        db.student_accounts.create_index([("studentId", ASCENDING)])
        db.student_accounts.create_index([("createdAt", ASCENDING)])
        
        print("✓ MongoDB connected")
        return db
    except ConnectionFailure as e:
        print(f"✗ MongoDB connection failed: {e}")
        return None

def get_db():
    """Dependency to get database instance."""
    global db
    if db is None:
        db = connect_db()
    return db


# ── Document Models (MongoDB) ────────────────────────────────────────────────
# These are not ORM models, just document structure helpers

class Candidate:
    """Candidate document model."""
    @staticmethod
    def create(db, name, email, password_hash, reference_face_encoding=None):
        return db.candidates.insert_one({
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "reference_face_encoding": reference_face_encoding,
            "created_at": datetime.now(timezone.utc),
        }).inserted_id
    
    @staticmethod
    def find_by_email(db, email):
        return db.candidates.find_one({"email": email})
    
    @staticmethod
    def find_by_id(db, id):
        from bson import ObjectId
        return db.candidates.find_one({"_id": ObjectId(id) if isinstance(id, str) else id})


class Exam:
    """Exam document model."""
    @staticmethod
    def create(db, id, candidate_id, candidate_name, exam_title="Exam"):
        return db.exams.insert_one({
            "_id": id,
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "exam_title": exam_title,
            "start_time": datetime.now(timezone.utc),
            "end_time": None,
            "status": "ACTIVE",
        })
    
    @staticmethod
    def find_by_id(db, id):
        return db.exams.find_one({"_id": id})
    
    @staticmethod
    def list_recent(db, limit=50):
        return list(db.exams.find().sort("start_time", -1).limit(limit))
    
    @staticmethod
    def update_status(db, id, status, end_time=None):
        update_data = {"status": status}
        if end_time:
            update_data["end_time"] = end_time
        return db.exams.update_one({"_id": id}, {"$set": update_data})


class Violation:
    """Violation document model."""
    @staticmethod
    def create(db, exam_id, type, severity, risk_score=0, snapshot_path=None):
        return db.violations.insert_one({
            "exam_id": exam_id,
            "type": type,
            "severity": severity,
            "risk_score": risk_score,
            "timestamp": datetime.now(timezone.utc),
            "snapshot_path": snapshot_path,
        }).inserted_id
    
    @staticmethod
    def find_by_exam(db, exam_id):
        return list(db.violations.find({"exam_id": exam_id}))
    
    @staticmethod
    def count_by_exam(db, exam_id):
        return db.violations.count_documents({"exam_id": exam_id})


class ProctorUser:
    """Proctor user document model."""
    @staticmethod
    def create(db, name, email, password_hash):
        return db.proctor_users.insert_one({
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc),
        }).inserted_id
    
    @staticmethod
    def find_by_email(db, email):
        return db.proctor_users.find_one({"email": email})
