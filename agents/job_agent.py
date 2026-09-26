import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.schemas import JobData

load_dotenv()

# Verify API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in your .env file!")

# Initialize LLM client at module level
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,  # Zero temperature for deterministic extraction
    google_api_key=api_key
)

# Bind Pydantic schema for structured output
job_llm = llm.with_structured_output(JobData)


def run_job_agent(job_raw_text: str) -> dict:
    """
    Analyzes raw job description text and extracts structured requirements matching JobData schema.
    """
    if not job_raw_text or not job_raw_text.strip():
        raise ValueError("Job description text cannot be empty.")

    prompt = f"""
    You are an expert Job Description Analysis Agent.
    Analyze the provided raw Job Description text and extract structured criteria.

    INSTRUCTIONS:
    1. Extract the target job title/role (e.g., 'Python Backend Developer', 'AI Engineer').
    2. Identify all explicitly mandatory/required skills (technical skills, languages, frameworks, databases, tools).
    3. Identify all preferred, nice-to-have, or optional skills.
    4. Extract key job responsibilities and daily duties.
    5. Extract experience requirements (e.g., '1-3 years', 'Entry Level', '5+ years').
    6. Extract education requirements (e.g., 'Bachelor's in Computer Science').
    7. Extract any other requirements like certifications, communication skills, or domain expertise.

    RAW JOB DESCRIPTION:
    {job_raw_text}
    """

    # Call Gemini LLM with structured output schema
    extracted_data: JobData = job_llm.invoke(prompt)
    
    # Return dictionary representation
    return extracted_data.model_dump()