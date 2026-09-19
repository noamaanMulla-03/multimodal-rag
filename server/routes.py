from fastapi import APIRouter, HTTPException, status

from server.schemas import QueryRequest
from server.services import (
    ingest_documents,
    reset_vector_db,
    search_documents,
)

# APIRouter keeps HTTP details separate from the RAG service implementation.
router = APIRouter(tags=["rag"])


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
def ingest():
    # Ingestion changes the persistent vector database, so this is a POST route.
    try:
        return ingest_documents()
    # Services raise normal Python errors; routes translate them into HTTP errors.
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query")
def query_rag(request: QueryRequest):
    # QueryRequest has already validated the JSON body before this function runs.
    results = search_documents(request.query, request.k)

    return {
        "query": request.query,
        "results": results,
    }


@router.post("/reset-db")
def reset_db():
    try:
        deleted_count = reset_vector_db()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to reset vector database",
        ) from exc

    return {
        "status": "reset",
        "chunks_deleted": deleted_count,
    }