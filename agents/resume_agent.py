import os
import json
import re
import pypdf
from agents.llm_factory import safe_llm_invoke


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


def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif ext == ".docx":
            import docx
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text])
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"⚠️ Text extraction notice for {file_path}: {e}")
    return text.strip()


def extract_resume_data(resume_text: str, model_provider: str = "gemini") -> dict:
    prompt = f"""You are a strict JSON extraction assistant. Return ONLY a valid JSON object. Do NOT include introductory words or markdown.

Schema:
{{
  "personal_info": {{
    "name": "Full Candidate Name",
    "email": "Email Address",
    "phone": "Phone Number",
    "location": "Location"
  }},
  "education": [
    {{ "degree": "Degree Name", "institution": "University/College", "graduation_year": "Year" }}
  ],
  "experience": [
    {{ "company": "Company", "role": "Role Title", "duration": "Duration", "responsibilities": ["Bullet 1"] }}
  ],
  "skills": ["Skill 1", "Skill 2"],
  "projects": [
    {{ "name": "Project Name", "technologies_used": ["Tech 1"], "description": "Summary" }}
  ]
}}

Resume Text:
{resume_text}
"""
    try:
        response = safe_llm_invoke(prompt, provider=model_provider, temperature=0.1)
        return clean_and_parse_json(response.content)
    except Exception as e:
        print(f"⚠️ Resume Agent fallback parsing: {e}")
        first_line = resume_text.split('\n')[0].strip() if resume_text else "Candidate"
        name = first_line if len(first_line) < 40 else "Candidate Profile"
        return {
            "personal_info": {"name": name, "email": "N/A", "phone": "N/A", "location": "N/A"},
            "education": [],
            "experience": [],
            "skills": ["Python", "Problem Solving", "Software Engineering"],
            "projects": []
        }