import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.schemas import GapAnalysis

load_dotenv()

# Verify API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in your .env file!")

# Initialize LLM client at module level
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,  # Zero temperature for deterministic evaluation
    google_api_key=api_key
)

# Bind Pydantic schema for structured output
gap_llm = llm.with_structured_output(GapAnalysis)


def run_gap_agent(matches: list, job_data: dict) -> dict:
    """
    Identifies skill gaps, prioritizes them based on job description criticality, 
    and generates actionable recommendations matching GapAnalysis schema.
    """
    # Filter out Strong Matches to isolate missing or partial requirements
    non_strong_matches = [
        m for m in matches 
        if isinstance(m, dict) and m.get("status") in ["Partial Match", "Weak Evidence", "Missing"]
    ]

    prompt = f"""
    You are an expert Gap Analysis Agent.
    Analyze the identified match gaps and evaluate candidate deficiencies against the job description requirements.

    JOB REQUIREMENTS SUMMARY:
    {job_data}

    IDENTIFIED GAPS (Requirements Not Fully Met):
    {non_strong_matches}

    INSTRUCTIONS:
    1. Evaluate each non-strong match item.
    2. Classify the priority of each gap based on its importance in the Job Description:
       - 'High Priority': Core mandatory skills, essential experience thresholds, or primary job responsibilities.
       - 'Medium Priority': Secondary required skills or primary preferred skills.
       - 'Low Priority': Optional/nice-to-have preferred skills or minor requirements.
    3. Clearly explain the exact reason for the gap (e.g., 'No AWS cloud experience identified', '1 year shortage in required work experience').
    4. Provide concrete, actionable, and realistic candidate recommendations to bridge each individual gap.
    5. Formulate 2-4 overall high-level practical recommendations for candidate improvement.
    """

    gap_result: GapAnalysis = gap_llm.invoke(prompt)
    return gap_result.model_dump()