from rag.document_loader import load_documents
from rag.text_chunking import text_chunker
from rag.embeddings import embedd_chunks, get_vector_store


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
