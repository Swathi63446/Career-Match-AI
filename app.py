import os
import uuid
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from firebase_helper import (
    upload_resume_file,
    save_analysis_result,
    fetch_all_analyses,
    fetch_single_analysis
)
from agents.resume_agent import extract_text_from_file
from agents.manager import run_manager_workflow

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_dev_secret_key_12345")


# ==========================================
# 1. DASHBOARD & HISTORY ROUTES
# ==========================================

@app.route("/")
@app.route("/dashboard")
def dashboard():
    """Renders the main dashboard page with past analysis history."""
    try:
        past_analyses = fetch_all_analyses()
    except Exception as e:
        past_analyses = []
        print(f"Firestore fetch error: {str(e)}")
        
    return render_template("dashboard.html", analyses=past_analyses)


@app.route("/analysis/<analysis_id>")
def view_analysis(analysis_id):
    """Renders the detailed match report page for a specific analysis ID."""
    analysis_data = fetch_single_analysis(analysis_id)
    if not analysis_data:
        return render_template("404.html", message="Analysis report not found."), 404
        
    return render_template("report.html", data=analysis_data)


# ==========================================
# 2. MULTI-AGENT ANALYSIS API ENDPOINT
# ==========================================

@app.route("/analyze", methods=["POST"])
def analyze_match():
    """
    Handles PDF/DOCX resume file upload, extracts raw text, 
    triggers the 5-agent execution pipeline, and persists results.
    """
    try:
        # Validate file presence
        if "resume" not in request.files:
            return jsonify({"success": False, "error": "No resume file uploaded."}), 400
            
        resume_file = request.files["resume"]
        job_description = request.form.get("job_description", "").strip()

        if resume_file.filename == "":
            return jsonify({"success": False, "error": "No selected file."}), 400

        if not job_description:
            return jsonify({"success": False, "error": "Job description text cannot be empty."}), 400

        # Read file bytes in memory
        file_bytes = resume_file.read()
        filename = resume_file.filename

        # Generate unique analysis ID
        analysis_id = f"anls_{uuid.uuid4().hex[:10]}"

        # 1. Upload resume to Firebase Storage
        print(f"[{analysis_id}] Uploading file to Firebase Storage...")
        resume_url = upload_resume_file(file_bytes, filename)

        # 2. Extract plain text from PDF/DOCX file in memory
        print(f"[{analysis_id}] Extracting plain text from file...")
        resume_raw_text = extract_text_from_file(file_bytes, filename)

        # 3. Execute Multi-Agent Orchestration Pipeline
        print(f"[{analysis_id}] Triggering Multi-Agent Pipeline...")
        shared_state = run_manager_workflow(
            resume_raw_text=resume_raw_text,
            job_raw_text=job_description,
            analysis_id=analysis_id
        )

        # 4. Save analysis results to Firestore
        print(f"[{analysis_id}] Saving results to Firestore...")
        saved_record = save_analysis_result(
            analysis_id=analysis_id,
            state_data=shared_state,
            resume_url=resume_url
        )

        return jsonify({
            "success": True,
            "analysis_id": analysis_id,
            "redirect_url": f"/analysis/{analysis_id}",
            "data": saved_record
        }), 200

    except Exception as e:
        print(f"Error during analysis endpoint execution: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)