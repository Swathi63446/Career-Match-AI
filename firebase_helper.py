import os
import uuid
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore, storage
from dotenv import load_dotenv

load_dotenv()

CREDENTIAL_FILE = "serviceAccountKey.json"

# Fallback for double extension filename if present
if not os.path.exists(CREDENTIAL_FILE) and os.path.exists("serviceAccountKey.json.json"):
    CREDENTIAL_FILE = "serviceAccountKey.json.json"

if not firebase_admin._apps:
    if os.path.exists(CREDENTIAL_FILE):
        cred = credentials.Certificate(CREDENTIAL_FILE)
        project_id = cred.project_id
        
        # Reads storage bucket from .env, default fallback to classic .appspot.com
        bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", f"{project_id}.appspot.com")
        
        firebase_admin.initialize_app(cred, {
            'storageBucket': bucket_name
        })
    else:
        raise FileNotFoundError(
            f"Firebase key file '{CREDENTIAL_FILE}' not found in project root! "
            "Please ensure serviceAccountKey.json is placed in the project root."
        )

db = firestore.client()


def upload_resume_file(file_bytes: bytes, original_filename: str) -> str:
    """
    Attempts to upload PDF/DOCX resume file bytes to Firebase Storage.
    If storage bucket is unavailable or throws an error, gracefully returns 
    a fallback URL so the multi-agent AI pipeline continues without crashing.
    """
    try:
        bucket = storage.bucket()
        file_extension = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else 'pdf'
        unique_path = f"resumes/{uuid.uuid4().hex}.{file_extension}"
        
        blob = bucket.blob(unique_path)
        blob.upload_from_string(file_bytes, content_type="application/octet-stream")
        
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return blob.generate_signed_url(expiration=3600 * 24 * 7)
            
    except Exception as e:
        print(f"[Firebase Warning] Storage upload bypassed: {str(e)}")
        # Safe fallback URL so downstream agents run smoothly
        return "#storage-bypassed"


def save_analysis_result(analysis_id: str, state_data: dict, resume_url: str) -> dict:
    """
    Saves state output into the 'analyses' Firestore document collection.
    """
    record = {
        "analysis_id": analysis_id,
        "candidate_name": state_data.get("resume_data", {}).get("personal_info", {}).get("name", "Unknown Candidate"),
        "job_title": state_data.get("job_data", {}).get("target_role", "Job Role"),
        "resume_url": resume_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "match_summary": state_data.get("final_report", {}).get("overall_fit_summary", ""),
        "fit_category": state_data.get("final_report", {}).get("fit_category", "Analyzed"),
        "matches": state_data.get("matches", []),
        "skill_gaps": state_data.get("skill_gaps", []),
        "recommendations": state_data.get("recommendations", []),
        "resume_data": state_data.get("resume_data", {}),
        "job_data": state_data.get("job_data", {})
    }
    
    db.collection('analyses').document(analysis_id).set(record)
    return record


def fetch_all_analyses() -> list:
    """
    Retrieves all past candidate match documents from Firestore ordered by date.
    """
    docs = db.collection('analyses').order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    return [doc.to_dict() for doc in docs]


def fetch_single_analysis(analysis_id: str) -> dict:
    """
    Retrieves a single match document by analysis_id.
    """
    doc = db.collection('analyses').document(analysis_id).get()
    return doc.to_dict() if doc.exists else None