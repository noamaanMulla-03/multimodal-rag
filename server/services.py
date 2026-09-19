import os
import shutil
from pathlib import Path
from typing import BinaryIO

from rag.document_loader import load_documents
from rag.text_chunking import text_chunker
from rag.embeddings import embedd_chunks, get_vector_store


MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class VectorStoreEmptyError(Exception):
    pass


def ingest_documents():
    # The ingestion pipeline is intentionally kept outside the HTTP layer.
    documents = load_documents()

    # Documents are normalized by the loader before they are split into chunks.
    chunks = text_chunker(documents)

    if not chunks:
        raise ValueError("No PDF content was found to ingest")

    embedd_chunks(chunks)

    return {
        "documents_ingested": len(documents),
        "chunks_created": len(chunks),
    }


def search_documents(query: str, k: int):
    # The vector store handles embedding the query and finding nearest chunks.
    store = get_vector_store()

    if not store.get(include=[])["ids"]:
        raise VectorStoreEmptyError(
            "No documents have been ingested yet."
        )

    results = store.similarity_search(
        query,
        k=k
    )

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
        }
        for doc in results
    ]


def reset_vector_db():
    store = get_vector_store()

    # Chroma always returns IDs, even when no extra fields are requested.
    existing_ids = store.get(include=[])["ids"]

    if existing_ids:
        store.delete(ids=existing_ids)

    # Force the next request to create a fresh Chroma client.
    get_vector_store.cache_clear()

    return len(existing_ids)


def save_uploaded_pdf(filename: str | None, file_obj: BinaryIO):
    project_root = Path(__file__).resolve().parent.parent
    docs_dir = project_root / os.getenv("DOCS_DIR", "docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        raise ValueError("A filename is required")

    # Remove any directory path supplied by the client.
    safe_filename = Path(filename.replace("\\", "/")).name

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are allowed")

    destination = docs_dir / safe_filename

    # Confirm that the file is actually a PDF.
    file_obj.seek(0)
    if file_obj.read(5) != b"%PDF-":
        raise ValueError("The uploaded file is not a valid PDF")
    file_obj.seek(0)

    bytes_written = 0

    with destination.open("wb") as output_file:
        while chunk := file_obj.read(1024 * 1024):
            bytes_written += len(chunk)

            if bytes_written > MAX_UPLOAD_BYTES:
                destination.unlink(missing_ok=True)
                raise ValueError("PDF cannot be larger than 25 MB")

            output_file.write(chunk)

    return {
        "filename": safe_filename,
        "size_bytes": bytes_written,
    }