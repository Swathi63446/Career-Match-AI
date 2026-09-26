import os
import io
import pdfplumber
import docx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.schemas import ResumeData

load_dotenv()

# Verify API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in your .env file!")

# Initialize LLM once at module level (prevents re-creation on every request)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,  # Zero temperature for deterministic extraction
    google_api_key=api_key
)

# Bind Pydantic schema for structured output
resume_llm = llm.with_structured_output(ResumeData)


# ==========================================
# 1. RAW TEXT EXTRACTION HELPER
# ==========================================

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Extracts plain text from PDF or DOCX file bytes in memory without writing temporary files to disk.
    """
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    extracted_text = ""

    if ext == 'pdf':
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
    elif ext == 'docx':
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text.strip():
                extracted_text += para.text + "\n"
    else:
        # Fallback for plain text files (.txt)
        extracted_text = file_bytes.decode('utf-8', errors='ignore')

    if not extracted_text.strip():
        raise ValueError("Failed to extract text from the uploaded resume file. The file may be empty or corrupted.")

    return extracted_text.strip()


# ==========================================
# 2. RESUME AGENT EXECUTION
# ==========================================

def run_resume_agent(resume_raw_text: str) -> dict:
    """
    Parses raw resume text and extracts structured candidate information matching ResumeData schema.
    """
    prompt = f"""
    You are an expert Resume Parsing Agent.
    Your task is to analyze the provided raw resume text and extract candidate information into a structured schema.

    INSTRUCTIONS:
    1. Extract full personal details (Name, Email, Phone, Location, LinkedIn/GitHub).
    2. Extract education history including degrees, institutions, and graduation years.
    3. Extract work experience details, listing company, job role, duration, and key responsibilities/achievements.
    4. Extract all technical skills, frameworks, programming languages, databases, tools, and domain skills.
    5. Extract all project details including project name, technologies used, and project description.
    6. If a field is missing in the resume, provide sensible defaults (e.g., 'Not Provided' or empty lists).

    RAW RESUME TEXT:
    {resume_raw_text}
    """

    # Call Gemini LLM with structured output schema
    extracted_data: ResumeData = resume_llm.invoke(prompt)
    
    # Return dictionary representation
    return extracted_data.model_dump()