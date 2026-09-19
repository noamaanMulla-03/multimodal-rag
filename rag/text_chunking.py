import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Keep chunking configuration in the workspace .env file.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Use a deterministic tokenizer so chunk sizes are measured consistently.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def text_chunker(documents):
    # Nothing to split when the document loader found no usable text.
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "30")),
        length_function=count_tokens,
    )

    # Preserve document metadata such as source filename and page number.
    chunks = text_splitter.split_documents(documents)
    return chunks
