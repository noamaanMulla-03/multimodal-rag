from fastapi import FastAPI

from server.routes import router

# FastAPI and Uvicorn import this object when the API server starts.
app = FastAPI()

# Keep endpoint definitions in routes.py instead of placing them in this file.
app.include_router(router)


@app.get("/")
def read_root():
    # Small health-check endpoint for confirming that the API is running.
    return {"message": "RAG API is running"}
