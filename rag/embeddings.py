import hashlib
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_COLLECTION_NAME = "pdf_documents"


def get_chunk_source(chunk) -> str:
    return str(chunk.metadata.get("source", "unknown"))


def generate_chunk_ids(chunks):
    chunk_ids = []

    for chunk in chunks:
        source = get_chunk_source(chunk)
        source_name = Path(source).name
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        page = chunk.metadata.get("page", "unknown")
        normalized_text = re.sub(r"\s+", " ", chunk.page_content).strip()
        text_hash = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()[:16]

        chunk_ids.append(f"{source_name}:{source_hash}:{page}:{text_hash}")

    return chunk_ids


@lru_cache
def get_vector_store():
    project_root = Path(__file__).resolve().parent.parent
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    persist_directory = project_root / os.getenv("PERSIST_DIRECTORY", "chroma_db")
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=embedding_model_name),
        persist_directory=str(persist_directory),
    )


def embedd_chunks(chunks):
    vector_db = get_vector_store()
    chunk_ids = generate_chunk_ids(chunks)

    current_ids_by_source = defaultdict(set)
    for chunk, chunk_id in zip(chunks, chunk_ids):
        current_ids_by_source[get_chunk_source(chunk)].add(chunk_id)

    existing_ids_by_source = {
        source: set(vector_db.get(where={"source": source})["ids"])
        for source in current_ids_by_source
    }

    vector_db.add_documents(documents=chunks, ids=chunk_ids)

    for source, current_ids in current_ids_by_source.items():
        stale_ids = existing_ids_by_source[source] - current_ids
        if stale_ids:
            vector_db.delete(ids=list(stale_ids))

    return vector_db
