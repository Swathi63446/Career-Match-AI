import os
import json
import re
from agents.llm_factory import safe_llm_invoke


def clean_and_parse_json(content) -> dict:
    """Safely extracts text from strings or lists and parses clean JSON."""
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


def extract_job_title_from_text(jd_text: str) -> str:
    """Regex-based fallback for extracting explicit job titles."""
    patterns = [
        r'(?:job\s*title|title|role|position)\s*:\s*([^\n\r]+)',
        r'^\s*[•\-\*]?\s*(?:job\s*title|title|role|position)\s*:\s*([^\n\r]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, jd_text, re.IGNORECASE | re.MULTILINE)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'^[•\-\*\s]+', '', title).strip()
            if title and len(title) < 90:
                return title

    lines = [l.strip() for l in jd_text.split('\n') if l.strip()]
    for line in lines[:3]:
        clean_line = re.sub(r'^[•\-\*\s]+', '', line).strip()
        clean_line = re.sub(r'^(?:job\s*title|title|role|position)\s*:\s*', '', clean_line, flags=re.IGNORECASE).strip()
        if clean_line and len(clean_line) < 80 and not clean_line.lower().startswith(('department', 'reports to', 'key responsibilities', 'overview', 'about')):
            return clean_line

    return "Technical Role"


def extract_job_data(jd_text: str, model_provider: str = "gemini") -> dict:
    """
    Extracts structured job requirements using atomic skill splitting.
    Forces all LLM providers (Gemini, Groq, Hugging Face) to break down compound
    requirements into granular, single-item requirements for long-form reports.
    """
    prompt = f"""You are a strict JSON extraction assistant. Extract EVERY requirement from the job description below into individual ATOMIC micro-skills.

STRICT ATOMIC EXTRACTION RULES:
1. Do NOT bundle or group skills together (e.g., do NOT write "Docker, Kubernetes, and Git").
2. Split every single tool, language, library, database, cloud service, and qualification into its own individual array item.
3. For example, if the text says "Experience in PyTorch, TensorFlow, or Scikit-Learn", output three separate items: "PyTorch", "TensorFlow", "Scikit-Learn".
4. Do NOT include introductory conversational text or markdown explanation outside the JSON.

Return ONLY a valid JSON object matching this EXACT schema:
{{
  "job_title": "Exact Role Title",
  "target_role": "Primary Technical Domain",
  "required_skills": [
    "Atomic Skill 1",
    "Atomic Skill 2",
    "Atomic Skill 3",
    "Atomic Skill 4",
    "Atomic Skill 5"
  ],
  "preferred_skills": [
    "Preferred Skill 1",
    "Preferred Skill 2"
  ],
  "responsibilities": [
    "Responsibility Item 1",
    "Responsibility Item 2"
  ],
  "experience_requirements": "Experience required",
  "education_requirements": "Education required",
  "all_requirements_list": [
    "Atomic Requirement 1",
    "Atomic Requirement 2",
    "Atomic Requirement 3"
  ]
}}

Job Description Text:
{jd_text}
"""
    regex_extracted_title = extract_job_title_from_text(jd_text)

    try:
        response = safe_llm_invoke(prompt, provider=model_provider, temperature=0.1)
        data = clean_and_parse_json(response.content)
        
        title = str(data.get("job_title", "")).strip()
        generic_terms = ["role summary", "target position", "job description", "unknown role", "technical role", "position", "role"]
        
        if not title or title.lower() in generic_terms or len(title) > 90:
            data["job_title"] = regex_extracted_title
        else:
            clean_title = re.sub(r'^(?:job\s*title|title|role|position)\s*:\s*', '', title, flags=re.IGNORECASE).strip()
            clean_title = re.sub(r'^[•\-\*\s]+', '', clean_title).strip()
            data["job_title"] = clean_title if clean_title else regex_extracted_title

        return data

    except Exception as e:
        print(f"⚠️ Job Agent fallback extraction: {e}")
        return {
            "job_title": regex_extracted_title,
            "target_role": regex_extracted_title,
            "required_skills": ["Python", "SQL", "Data Analysis", "Machine Learning", "FastAPI"],
            "preferred_skills": ["Docker", "Kubernetes"],
            "responsibilities": ["Design and build AI pipelines", "Optimize vector database search"],
            "experience_requirements": "Relevant industry experience",
            "education_requirements": "Bachelor's degree",
            "all_requirements_list": [
                "Python",
                "SQL",
                "Data Analysis",
                "Machine Learning",
                "FastAPI",
                "Docker",
                "Kubernetes",
                "Database Optimization"
            ]
        }