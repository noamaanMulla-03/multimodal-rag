from fastapi import APIRouter, HTTPException, status

from server.schemas import QueryRequest
from server.services import ingest_documents, search_documents

router = APIRouter(tags=["rag"])


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
def ingest():
    try:
        return ingest_documents()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query")
def query_rag(request: QueryRequest):
    results = search_documents(request.query, request.k)

    return {
        "query": request.query,
        "results": results,
    }