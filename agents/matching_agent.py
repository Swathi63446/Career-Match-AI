import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from agents.schemas import MatchingAnalysis

load_dotenv()

# Verify API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in your .env file!")

PERSIST_DIRECTORY = "./vectorstore"

# Initialize LLM and Embeddings once at module level
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,  # Zero temperature for deterministic evaluation
    google_api_key=api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)

# Bind Pydantic schema for structured output
matching_llm = llm.with_structured_output(MatchingAnalysis)


# ==========================================
# RAG RETRIEVAL HELPER
# ==========================================

def retrieve_rag_context(skills_to_query: list) -> str:
    """
    Queries local ChromaDB vector store for each skill to gather domain knowledge context.
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        return "No ChromaDB knowledge base found at specified path."

    try:
        # Load persistent Chroma vectorstore in read-only query mode
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        retrieved_knowledge = []

        # Deduplicate skill query items
        unique_skills = list(set([s for s in skills_to_query if isinstance(s, str) and s.strip()]))

        for skill in unique_skills:
            docs = vectorstore.similarity_search(query=skill, k=1)
            for doc in docs:
                skill_name = doc.metadata.get('skill_name', skill)
                category = doc.metadata.get('category', 'Technical Skill')
                retrieved_knowledge.append(
                    f"• Skill Context [{skill_name} ({category})]: {doc.page_content}"
                )

        return "\n".join(retrieved_knowledge) if retrieved_knowledge else "No additional RAG context retrieved."
    except Exception as e:
        return f"ChromaDB retrieval note: {str(e)}"


# ==========================================
# MATCHING AGENT EXECUTION
# ==========================================

def run_matching_agent(resume_data: dict, job_data: dict) -> dict:
    """
    Evaluates candidate resume data against job requirements using semantic reasoning and RAG context.
    """
    # 1. Collect required and preferred skills to query ChromaDB RAG layer
    required_skills = job_data.get("required_skills", [])
    preferred_skills = job_data.get("preferred_skills", [])
    skills_to_query = required_skills + preferred_skills
    
    # 2. Retrieve technical context profiles from ChromaDB
    rag_context = retrieve_rag_context(skills_to_query)

    prompt = f"""
    You are an expert Semantic Matching Agent.
    Your goal is to evaluate how well the candidate's resume profile satisfies every requirement in the job description.

    JOB REQUIREMENTS (Extracted from Job Description):
    {job_data}

    CANDIDATE RESUME PROFILE (Extracted from Resume):
    {resume_data}

    RETRIEVED TECHNICAL KNOWLEDGE LAYER (From ChromaDB RAG Vector Store):
    {rag_context}

    EVALUATION INSTRUCTIONS:
    1. Evaluate EVERY major job requirement (Required Skills, Preferred Skills, Experience Requirements, Education Requirements).
    2. Perform SEMANTIC matching rather than relying strictly on exact keyword matching.
       Use the retrieved technical knowledge layer to recognize tool relationships and framework equivalencies (e.g., Flask is a Python REST API backend framework; PostgreSQL is a relational database).
    3. Categorize each requirement into one of four strict status ratings:
       - 'Strong Match': Requirement is fully satisfied with clear, direct proof in the candidate's experience or projects.
       - 'Partial Match': Conceptually related skill exists, or requirement is partially met (e.g., candidate has 1 year vs 2+ years requested).
       - 'Weak Evidence': Skill is listed without enough context, depth, or project demonstration.
       - 'Missing': No evidence found anywhere in the resume.
    4. For EVERY evaluated requirement, extract and attach DIRECT PROOF/EVIDENCE from the candidate's resume (company names, project details, bullet points).
    5. Indicate in 'rag_context_used' if ChromaDB context helped establish technology relationships.
    """

    matching_result: MatchingAnalysis = matching_llm.invoke(prompt)
    return matching_result.model_dump()