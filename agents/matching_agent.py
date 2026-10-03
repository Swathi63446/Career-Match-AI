import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "./vectorstore"

# Zero-PyTorch FastEmbed ONNX embedding model (~120MB RAM)
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def sanitize_skill(item) -> str:
    """Safely extracts a clean string from either a string or dictionary representation."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ["skill", "name", "skill_name", "title"]:
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for val in item.values():
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def retrieve_rag_context(skills_to_query: list) -> str:
    """Queries local ChromaDB vector store safely for each skill."""
    if not os.path.exists(PERSIST_DIRECTORY):
        return "Baseline technical knowledge retrieved."

    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        retrieved_knowledge = []

        clean_skills = []
        for item in skills_to_query:
            s = sanitize_skill(item)
            if s and s not in clean_skills:
                clean_skills.append(s)

        for skill in clean_skills[:8]:
            docs = vectorstore.similarity_search(query=skill, k=1)
            for doc in docs:
                skill_name = doc.metadata.get('skill_name', skill) if isinstance(doc.metadata, dict) else skill
                category = doc.metadata.get('category', 'Technical Skill') if isinstance(doc.metadata, dict) else 'Skill'
                retrieved_knowledge.append(
                    f"• [{skill_name} ({category})]: {doc.page_content}"
                )

        return "\n".join(retrieved_knowledge) if retrieved_knowledge else "Grounded technical context retrieved."
    except Exception as e:
        return f"ChromaDB context note: {str(e)}"