import os
from langchain_core.messages import HumanMessage

# Fallback environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Active Model Registry
MODELS = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
    "groq": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    "huggingface": os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    "featherless-ai": os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    "featherless": os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
}


def get_langchain_model(provider: str):
    """Factory function returning a configured LangChain Chat model instance using lazy imports."""
    provider_clean = (provider or "gemini").lower().strip()

    if provider_clean == "groq":
        from langchain_groq import ChatGroq
        groq_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
        if not groq_key:
            raise ValueError("GROQ_API_KEY is missing in environment.")
        return ChatGroq(
            groq_api_key=groq_key,
            model_name=MODELS.get("groq", "openai/gpt-oss-20b"),
            temperature=0.1,
            max_tokens=4096,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    elif provider_clean in ["featherless-ai", "featherless", "huggingface"]:
        from langchain_openai import ChatOpenAI
        hf_token = os.getenv("HF_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", HF_API_TOKEN))
        if not hf_token:
            raise ValueError("HF_API_TOKEN is missing in environment.")

        base_model = MODELS.get(provider_clean, "Qwen/Qwen2.5-7B-Instruct")
        model_name = f"{base_model}:featherless-ai" if not base_model.endswith(":featherless-ai") else base_model

        return ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
            model_name=model_name,
            temperature=0.1,
            max_tokens=4096
        )

    else:  # Gemini Default
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is missing in environment.")

        model_name = MODELS.get("gemini", "gemini-3.5-flash")
        return ChatGoogleGenerativeAI(
            google_api_key=gemini_key,
            model=model_name,
            temperature=0.1
        )


def get_llm(provider: str = "gemini", temperature: float = 0.1):
    """Wrapper function for pipeline compatibility."""
    return get_langchain_model(provider)


def safe_llm_invoke(prompt_text: str, provider: str = "gemini", temperature: float = 0.1):
    """Safely invokes the selected model with automated Gemini fallback."""
    message = [HumanMessage(content=prompt_text)]
    provider_clean = (provider or "gemini").lower().strip()

    try:
        llm = get_langchain_model(provider=provider_clean)
        return llm.invoke(message)
    except Exception as e:
        print(f"⚠️ [{provider_clean.upper()} Provider Notice]: {e}")
        print("🔄 [Auto-Synthesis] Executing failover via Gemini 3.5 Flash...")
        gemini_llm = get_langchain_model(provider="gemini")
        return gemini_llm.invoke(message)