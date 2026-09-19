from fastapi import APIRouter, File, HTTPException, UploadFile, status

from server.schemas import QueryRequest
from server.services import (
    VectorStoreEmptyError,
    ingest_documents,
    reset_vector_db,
    save_uploaded_pdf,
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
    try:
        results = search_documents(request.query, request.k)

    except VectorStoreEmptyError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Query failed",
        ) from exc

    return {
        "query": request.query,
        "results": results,
    }


@router.post("/reset-db")
def reset_db():
    # Reset is destructive: it removes every indexed chunk, but keeps source PDFs.
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


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_pdf(file: UploadFile = File(...)):
    # Upload uses multipart/form-data and only saves the PDF; /ingest indexes it.
    try:
        upload_info = save_uploaded_pdf(file.filename, file.file)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "status": "uploaded",
        **upload_info,
    }
