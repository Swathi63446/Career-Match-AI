import os
import json
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOCAL_JSON_PATH = os.path.join(LOCAL_DATA_DIR, 'analyses.json')
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)


def load_local_store() -> dict:
    if os.path.exists(LOCAL_JSON_PATH):
        try:
            with open(LOCAL_JSON_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_local_store(data: dict):
    try:
        with open(LOCAL_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Local storage write error: {e}")


def initialize_firebase():
    if not firebase_admin._apps:
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-key.json")
        bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "career-match-ai-7c9c4.firebasestorage.app")
        project_id = os.getenv("FIREBASE_PROJECT_ID", "career-match-ai-7c9c4")

        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        options = {'storageBucket': bucket_name, 'projectId': project_id}

        if os.path.exists(cred_path):
            try:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, options)
                print(f"✅ [Firebase Initialized] Loaded credentials from '{cred_path}'")
            except Exception as e:
                print(f"ℹ️ [Firebase Notice] App init note: {e}")
        else:
            firebase_admin.initialize_app(options=options)


initialize_firebase()


def save_analysis_to_firestore(analysis_id: str, payload: dict) -> bool:
    if "created_at" not in payload:
        payload["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Save to local disk store first (Guarantees immediate report availability)
    local_data = load_local_store()
    local_data[analysis_id] = payload
    save_local_store(local_data)
    print(f"✅ [Local Store] Analysis '{analysis_id}' saved to disk.")

    # Attempt Cloud Firestore sync
    try:
        db = firestore.client()
        doc_ref = db.collection('analyses').document(analysis_id)
        doc_ref.set(payload)
        print(f"✅ [Firestore Success] Synced '{analysis_id}' to Cloud Firestore.")
        return True
    except Exception:
        print(f"ℹ️ [Storage Note] Saved locally in persistent disk store.")
        return False


def upload_resume_to_storage(file_path: str, filename: str) -> str:
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"resumes/{filename}")
        blob.upload_from_filename(file_path)
        blob.make_public()
        return blob.public_url
    except Exception:
        return f"/uploads/{filename}"


def get_analysis_from_firestore(analysis_id: str) -> dict:
    # 1. Check local persistent store first
    local_data = load_local_store()
    if analysis_id in local_data:
        return local_data[analysis_id]

    # 2. Check Firestore
    try:
        db = firestore.client()
        doc = db.collection('analyses').document(analysis_id).get()
        if doc.exists:
            return doc.to_dict()
    except Exception:
        pass

    return None


def get_all_analyses_from_firestore() -> list:
    local_data = load_local_store()
    analyses_map = {item["analysis_id"]: item for item in local_data.values() if isinstance(item, dict) and "analysis_id" in item}

    try:
        db = firestore.client()
        docs = db.collection('analyses').order_by('created_at', direction=firestore.Query.DESCENDING).stream()
        for doc in docs:
            d = doc.to_dict()
            if "analysis_id" in d:
                analyses_map[d["analysis_id"]] = d
    except Exception:
        pass

    results = list(analyses_map.values())
    results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return results