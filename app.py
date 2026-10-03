import os
import uuid
import datetime
import traceback
import hmac
import hashlib
import secrets
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

from agents.manager import run_optimized_pipeline
from firebase_helper import (
    save_analysis_to_firestore,
    upload_resume_to_storage,
    get_analysis_from_firestore,
    get_all_analyses_from_firestore
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def verify_credentials_securely(input_user: str, input_pass: str) -> bool:
    env_user = os.getenv("ADMIN_USERNAME")
    env_pass = os.getenv("ADMIN_PASSWORD")

    if not env_user or not env_pass:
        return False

    input_user_digest = hashlib.sha256(input_user.encode('utf-8')).hexdigest()
    env_user_digest = hashlib.sha256(env_user.encode('utf-8')).hexdigest()

    input_pass_digest = hashlib.sha256(input_pass.encode('utf-8')).hexdigest()
    env_pass_digest = hashlib.sha256(env_pass.encode('utf-8')).hexdigest()

    return hmac.compare_digest(input_user_digest, env_user_digest) and hmac.compare_digest(input_pass_digest, env_pass_digest)


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin'):
            if request.is_json or request.path.startswith('/analyze'):
                return jsonify({"error": "Unauthorized access. Please log in."}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    session.clear()
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        session.clear()
        return render_template('login.html')

    if request.is_json:
        data = request.get_json() or {}
        username = data.get('username', '').strip()
        password = data.get('password', '')
    else:
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

    if verify_credentials_securely(username, password):
        session.clear()
        session['admin'] = True
        session['user'] = username

        if request.is_json:
            return jsonify({"success": True, "redirect": "/dashboard"})
        return redirect(url_for('dashboard'))

    error_msg = "Invalid admin username or password."
    if request.is_json:
        return jsonify({"success": False, "error": error_msg}), 401

    return render_template('login.html', error=error_msg)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/dashboard')
@admin_required
def dashboard():
    past_analyses = get_all_analyses_from_firestore()
    return render_template('dashboard.html', past_analyses=past_analyses)


@app.route('/analyze', methods=['POST'])
@admin_required
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    file = request.files['resume']
    job_description = request.form.get('job_description', '').strip()
    model_provider = request.form.get('model_provider', 'gemini').strip()

    if file.filename == '' or not job_description:
        return jsonify({"error": "Resume file and Job Description are required"}), 400

    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}_{file.filename}"
    local_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(local_path)

    try:
        final_report = run_optimized_pipeline(local_path, job_description, model_provider=model_provider)
        resume_url = upload_resume_to_storage(local_path, filename)

        # Main Firestore & Local Storage Payload
        firestore_payload = {
            "analysis_id": unique_id,
            "candidate_name": final_report.get("candidate_name", "Candidate Profile"),
            "job_title": final_report.get("job_title", "Target Role"),
            "model_provider": model_provider,  # Raw provider string (gemini, groq, huggingface)
            "selected_model": final_report.get("selected_model", "Google Gemini 3.5 Flash"),  # Display label
            "fit_category": final_report.get("fit_category", "Evaluated"),
            "match_summary": final_report.get("match_summary", ""),
            "strengths": final_report.get("strengths", []),
            "top_gaps": final_report.get("top_gaps", []),
            "matches": final_report.get("matches", []),
            "skill_gaps": final_report.get("skill_gaps", []),
            "overall_score": final_report.get("overall_score", 50),
            "overall_recommendations": final_report.get("overall_recommendations", []),
            "resume_data": final_report.get("resume_data", {}),
            "job_data": final_report.get("job_data", {}),
            "resume_url": resume_url,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        save_analysis_to_firestore(unique_id, firestore_payload)
        return jsonify({"success": True, "analysis_id": unique_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Pipeline Execution Failure: {str(e)}"}), 500
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


@app.route('/report/<analysis_id>')
@admin_required
def report(analysis_id):
    report_data = get_analysis_from_firestore(analysis_id)
    if not report_data:
        return "Analysis report not found.", 404
    return render_template('report.html', report=report_data, data=report_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)