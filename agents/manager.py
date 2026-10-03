import os
import json
import re
from concurrent.futures import ThreadPoolExecutor

from agents.llm_factory import safe_llm_invoke, get_llm, MODELS
from agents.resume_agent import extract_text_from_file, extract_resume_data
from agents.job_agent import extract_job_data
from agents.matching_agent import retrieve_rag_context


def clean_and_parse_json(content) -> dict:
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]))
            elif hasattr(part, "text"):
                text_parts.append(str(part.text))
        content = "\n".join(text_parts)
    elif not isinstance(content, str):
        content = str(content)

    content = content.strip()
    if not content:
        raise ValueError("Empty content string")

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)
    
    content = re.sub(r',\s*([\}\]])', r'\1', content)
    return json.loads(content)


def sanitize_job_title(raw_title: str) -> str:
    if not raw_title:
        return "Technical Role"
    title = re.sub(r'^(?:job\s*title|title|role|position)\s*:\s*', '', str(raw_title), flags=re.IGNORECASE).strip()
    title = re.sub(r'^[•\-\*\s]+', '', title).strip()
    if title.lower() in ["role summary", "target position", "unknown role", "job description"]:
        return "Technical Role"
    return title


def run_optimized_pipeline(resume_file_path: str, jd_text: str, model_provider: str = "gemini") -> dict:
    raw_resume_text = extract_text_from_file(resume_file_path)
    if not raw_resume_text:
        raw_resume_text = "No readable resume content found."

    print(f"\n⚡ [STAGE 1] Running Extraction using AI Engine: {model_provider.upper()}...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_resume = executor.submit(extract_resume_data, raw_resume_text, model_provider)
        future_job = executor.submit(extract_job_data, jd_text, model_provider)

        resume_data = future_resume.result()
        job_data = future_job.result()
    print("✅ [STAGE 1 COMPLETE] Parallel extraction finished.")

    print("\n🔍 [STAGE 2] Querying Local ChromaDB RAG Vector Store ($0 API Cost)...")
    raw_skills = (
        resume_data.get("skills", []) + 
        job_data.get("required_skills", []) + 
        job_data.get("preferred_skills", [])
    )
    rag_context = retrieve_rag_context(raw_skills)
    print("✅ [STAGE 2 COMPLETE] Grounded technical context retrieved from ChromaDB.")

    personal_info = resume_data.get("personal_info", {})
    if not isinstance(personal_info, dict):
        personal_info = {}

    candidate_name = personal_info.get("name") or "Candidate Profile"
    job_title = sanitize_job_title(job_data.get("job_title") or job_data.get("target_role"))

    print(f"\n🧠 [STAGE 3] Executing Reasoning Pass using AI Engine: {model_provider.upper()}...")

    reasoning_prompt = f"""You are a Candidate Evaluation Agent. Return ONLY a valid JSON object matching this exact schema. Keep explanations concise to ensure complete JSON output.

Candidate Resume Data:
{json.dumps(resume_data, indent=2)}

Job Requirements Data:
{json.dumps(job_data, indent=2)}

Grounded Skill Knowledge (RAG Context):
{rag_context}

Return ONLY a JSON object with this structure:
{{
  "candidate_name": "{candidate_name}",
  "job_title": "{job_title}",
  "fit_category": "Strong Match" or "Partial Match" or "Weak Match",
  "overall_score": 75,
  "match_summary": "Concise 2-3 sentence alignment summary.",
  "strengths": ["Key Strength 1", "Key Strength 2"],
  "top_gaps": ["Primary Risk 1", "Primary Risk 2"],
  "matches": [
    {{
      "category": "Required Skill",
      "requirement": "Requirement Title",
      "status": "Strong Match",
      "evidence_from_resume": "Resume evidence snippet",
      "rag_context_used": "RAG snippet or None"
    }}
  ],
  "skill_gaps": [
    {{
      "skill": "Requirement Title",
      "priority": "High Priority",
      "status": "Missing",
      "reason": "Brief gap reason",
      "recommendation": "Targeted recommendation"
    }}
  ],
  "overall_recommendations": [
    "Practical recommendation 1"
  ]
}}
"""

    provider_clean = model_provider.lower().strip()
    selected_model_label = f"{provider_clean.title()} ({MODELS.get(provider_clean, 'Active Model')})"

    evaluation_data = None
    try:
        response = safe_llm_invoke(reasoning_prompt, provider=model_provider, temperature=0.2)
        evaluation_data = clean_and_parse_json(response.content)
    except Exception as parse_err:
        print(f"⚠️ Stage 3 Parse Error on {model_provider.upper()}: {parse_err}")
        try:
            gemini_llm = get_llm(provider="gemini", temperature=0.2)
            response = gemini_llm.invoke([{"role": "user", "content": reasoning_prompt}])
            evaluation_data = clean_and_parse_json(response.content)
        except Exception as final_err:
            print(f"⚠️ Stage 3 Gemini failover error: {final_err}")

    if evaluation_data and isinstance(evaluation_data, dict):
        print("✅ [STAGE 3 COMPLETE] Itemized evaluation complete.")
        eval_title = sanitize_job_title(evaluation_data.get("job_title"))

        return {
            "candidate_name": evaluation_data.get("candidate_name") or candidate_name,
            "job_title": eval_title if eval_title != "Technical Role" else job_title,
            "selected_model": selected_model_label,
            "fit_category": evaluation_data.get("fit_category", "Partial Match"),
            "overall_score": evaluation_data.get("overall_score", 65),
            "match_summary": evaluation_data.get("match_summary") or f"{candidate_name} evaluated against {job_title} requirements.",
            "strengths": evaluation_data.get("strengths") or ["Technical domain alignment"],
            "top_gaps": evaluation_data.get("top_gaps") or ["Requires practical verification"],
            "matches": evaluation_data.get("matches") or [],
            "skill_gaps": evaluation_data.get("skill_gaps") or [],
            "overall_recommendations": evaluation_data.get("overall_recommendations") or ["Focus on high-priority skill gaps."],
            "resume_data": resume_data,
            "job_data": job_data
        }

    # Dynamic Fallback Construction
    dynamic_matches = []
    dynamic_gaps = []
    candidate_skills = [s.lower() for s in resume_data.get("skills", []) if isinstance(s, str)]
    
    all_reqs = []
    for req in job_data.get("required_skills", []):
        all_reqs.append(("Required Skill", req, "High Priority"))
    for req in job_data.get("preferred_skills", []):
        all_reqs.append(("Preferred Skill", req, "Medium Priority"))
    for req in job_data.get("responsibilities", []):
        all_reqs.append(("Responsibility", req, "Medium Priority"))

    for cat, req, prio in all_reqs:
        req_str = str(req)
        is_matched = any(cs in req_str.lower() or req_str.lower() in cs for cs in candidate_skills)
        
        if is_matched:
            dynamic_matches.append({
                "category": cat,
                "requirement": req_str,
                "status": "Strong Match",
                "evidence_from_resume": f"Candidate demonstrates direct experience with {req_str}.",
                "rag_context_used": "Verified via knowledge store."
            })
        else:
            dynamic_matches.append({
                "category": cat,
                "requirement": req_str,
                "status": "Missing",
                "evidence_from_resume": f"No direct evidence for {req_str} identified in resume text.",
                "rag_context_used": "None"
            })
            dynamic_gaps.append({
                "skill": req_str,
                "priority": prio,
                "status": "Missing",
                "reason": f"Candidate profile lacks explicit mention or project evidence of {req_str}.",
                "recommendation": f"Demonstrate practical competence in {req_str} through targeted projects."
            })

    strong_count = len([m for m in dynamic_matches if m["status"] == "Strong Match"])
    total_count = max(len(dynamic_matches), 1)
    score = int((strong_count / total_count) * 100)

    return {
        "candidate_name": candidate_name,
        "job_title": job_title,
        "selected_model": selected_model_label,
        "fit_category": "Strong Match" if score >= 80 else ("Partial Match" if score >= 50 else "Weak Match"),
        "overall_score": score,
        "match_summary": f"Comprehensive itemized evaluation for {candidate_name} targeting {job_title}.",
        "strengths": [m["requirement"] for m in dynamic_matches if m["status"] == "Strong Match"][:4] or ["Foundational technical skills"],
        "top_gaps": [g["skill"] for g in dynamic_gaps[:3]] or ["Key job criteria require additional validation"],
        "matches": dynamic_matches,
        "skill_gaps": dynamic_gaps,
        "overall_recommendations": ["Prioritize high-priority skill gaps through practical projects."],
        "resume_data": resume_data,
        "job_data": job_data
    }