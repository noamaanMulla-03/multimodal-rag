from rag.document_loader import load_documents
from rag.text_chunking import text_chunker
from rag.embeddings import embedd_chunks, get_vector_store


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
    results = get_vector_store().similarity_search(
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
