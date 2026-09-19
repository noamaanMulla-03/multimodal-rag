import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load the project's local configuration and secret API key.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


@lru_cache
def get_llm() -> ChatOllama:
     # Fail clearly if the required Cloud configuration is missing.
    api_key = os.getenv("OLLAMA_API_KEY")
    model_name = os.getenv("OLLAMA_MODEL")

    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is missing")

    if not model_name:
        raise RuntimeError("OLLAMA_MODEL is missing")

    return ChatOllama(
        model=model_name,
        base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
        temperature=0,
        num_predict=512,
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {api_key}",
            },
        },
    )