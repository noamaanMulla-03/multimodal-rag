import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def text_chunker(documents):
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "30")),
        length_function=count_tokens,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks
