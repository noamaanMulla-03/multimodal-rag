from pathlib import Path
from typing import BinaryIO

from rag.document_loader import ensure_multimedia_directories, load_documents
from rag.embeddings import embedd_chunks, get_vector_store
from rag.llm import get_llm
from rag.text_chunking import text_chunker

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TOP_K = 5


class VectorStoreEmptyError(Exception):
    # Lets the route distinguish “nothing indexed” from an unexpected query failure.
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


def search_documents(query: str):
    # The vector store handles embedding the query and finding nearest chunks.
    store = get_vector_store()

    if not store.get(include=[])["ids"]:
        raise VectorStoreEmptyError(
            "No documents have been ingested yet."
        )

    results = store.similarity_search(
        query,
        k=TOP_K,
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
    # Delete indexed chunks while leaving original files in multimedia/ untouched.
    store = get_vector_store()

    # Chroma always returns IDs, even when no extra fields are requested.
    existing_ids = store.get(include=[])["ids"]

    if existing_ids:
        store.delete(ids=existing_ids)

    # Force the next request to create a fresh Chroma client.
    get_vector_store.cache_clear()

    return len(existing_ids)


def save_uploaded_pdf(filename: str | None, file_obj: BinaryIO):
    # PDFs are stored in the documents section of the fixed multimedia hierarchy.
    documents_dir = ensure_multimedia_directories() / "documents"

    if not filename:
        raise ValueError("A filename is required")

    # Remove any directory path supplied by the client.
    safe_filename = Path(filename.replace("\\", "/")).name

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are allowed")

    destination = documents_dir / safe_filename

    # Confirm that the file is actually a PDF.
    file_obj.seek(0)
    if file_obj.read(5) != b"%PDF-":
        raise ValueError("The uploaded file is not a valid PDF")
    file_obj.seek(0)

    bytes_written = 0

    # Stream the upload in bounded chunks instead of loading the whole file in memory.
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


def answer_question(query: str):
    results = search_documents(query)

    context = "\n\n".join(
        f"Source: {item['source']}, Page: {item['page']}\n"
        f"{item['content']}"
        for item in results
    )

    # Give `context` and `query` to the LLM here.
    answer = get_llm().invoke(
        f"""
        Answer only from the provided context.
        If the answer is not in the context, say so.
        Include the source and page when possible.

        Context:
        {context}

        Question:
        {query}
        """
    )

    return {
        "answer": answer.content,
        "sources": [
            {"source": item["source"], "page": item["page"]}
            for item in results
        ],
    }
