import os
import uuid
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore, storage
from dotenv import load_dotenv

load_dotenv()

CREDENTIAL_FILE = "serviceAccountKey.json"

if not os.path.exists(CREDENTIAL_FILE) and os.path.exists("serviceAccountKey.json.json"):
    CREDENTIAL_FILE = "serviceAccountKey.json.json"

# Initialize Firebase if credential file exists
if not firebase_admin._apps:
    if os.path.exists(CREDENTIAL_FILE):
        cred = credentials.Certificate(CREDENTIAL_FILE)
        project_id = cred.project_id
        bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", f"{project_id}.appspot.com")
        
        firebase_admin.initialize_app(cred, {
            'storageBucket': bucket_name
        })
    else:
        print(f"[Firebase Notice] Credential file '{CREDENTIAL_FILE}' not found. Running in CI/offline mode.")

# Safely attach Firestore database client
try:
    db = firestore.client() if firebase_admin._apps else None
except Exception:
    db = None


def upload_resume_file(file_bytes: bytes, original_filename: str) -> str:
    if not firebase_admin._apps:
        print("[Firebase Warning] Storage credentials missing. File upload bypassed.")
        return "#storage-bypassed"
        
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
        return "#storage-bypassed"


def save_analysis_result(analysis_id: str, state_data: dict, resume_url: str) -> dict:
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
    
    if db:
        db.collection('analyses').document(analysis_id).set(record)
    else:
        print("[Firebase Notice] Database save skipped (running in CI/offline mode).")
        
    return record


def fetch_all_analyses() -> list:
    if not db:
        return []
    docs = db.collection('analyses').order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    return [doc.to_dict() for doc in docs]


def fetch_single_analysis(analysis_id: str) -> dict:
    if not db:
        return None
    doc = db.collection('analyses').document(analysis_id).get()
    return doc.to_dict() if doc.exists else None