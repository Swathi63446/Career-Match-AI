import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.schemas import FinalReport
from agents.resume_agent import run_resume_agent
from agents.job_agent import run_job_agent
from agents.matching_agent import run_matching_agent
from agents.gap_agent import run_gap_agent

load_dotenv()

# Verify API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in your .env file!")

# Initialize LLM client at module level
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=api_key
)

# Bind Pydantic schema for structured final report output
final_report_llm = llm.with_structured_output(FinalReport)


def run_manager_workflow(resume_raw_text: str, job_raw_text: str, analysis_id: str) -> dict:
    """
    Orchestrates the multi-agent pipeline, maintaining shared state across execution steps.
    """
    # 1. Initialize Shared State (Per-request isolated dictionary)
    shared_state = {
        "analysis_id": analysis_id,
        "resume_raw": resume_raw_text,
        "job_raw": job_raw_text,
        "resume_data": {},
        "job_data": {},
        "matches": [],
        "skill_gaps": [],
        "recommendations": [],
        "final_report": {}
    }

    # 2. Step 1: Run Resume Agent
    print(f"[{analysis_id}] Step 1: Executing Resume Agent...")
    shared_state["resume_data"] = run_resume_agent(resume_raw_text)

    # 3. Step 2: Run Job Description Agent
    print(f"[{analysis_id}] Step 2: Executing Job Description Agent...")
    shared_state["job_data"] = run_job_agent(job_raw_text)

    # 4. Step 3: Run Matching Agent (with ChromaDB RAG Context)
    print(f"[{analysis_id}] Step 3: Executing Matching Agent (ChromaDB RAG)...")
    matching_result = run_matching_agent(
        resume_data=shared_state["resume_data"],
        job_data=shared_state["job_data"]
    )
    shared_state["matches"] = matching_result.get("matches", [])

    # 5. Step 4: Run Gap Analysis Agent
    print(f"[{analysis_id}] Step 4: Executing Gap Analysis Agent...")
    gap_result = run_gap_agent(
        matches=shared_state["matches"],
        job_data=shared_state["job_data"]
    )
    shared_state["skill_gaps"] = gap_result.get("skill_gaps", [])
    shared_state["recommendations"] = gap_result.get("overall_recommendations", [])

    # 6. Step 5: Final Executive Report Synthesis
    print(f"[{analysis_id}] Step 5: Synthesizing Final Report...")
    candidate_name = shared_state["resume_data"].get("personal_info", {}).get("name", "Candidate")
    target_role = shared_state["job_data"].get("target_role", "Target Role")

    prompt = f"""
    You are the Manager Agent synthesizing the final evaluation report for {candidate_name} applying for {target_role}.

    JOB REQUIREMENTS SUMMARY:
    {shared_state['job_data']}

    MATCHES & PROOF EVALUATION:
    {shared_state['matches']}

    SKILL GAPS & RECOMMENDATIONS:
    {shared_state['skill_gaps']}

    INSTRUCTIONS:
    1. Write a concise 3-4 sentence narrative summary explaining candidate alignment and key gaps.
    2. Determine the overall fit category: 'Strong Match', 'Partial Match', or 'Weak Match'.
    3. List 2-4 key candidate strengths based on strong matches.
    4. List 2-4 primary gaps or risks identified during analysis.
    """

    final_report_obj: FinalReport = final_report_llm.invoke(prompt)
    shared_state["final_report"] = final_report_obj.model_dump()

    print(f"[{analysis_id}] Workflow Complete!")
    return shared_state