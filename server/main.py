from fastapi import FastAPI

from server.routes import router

app = FastAPI()
app.include_router(router)


@app.get("/")
def read_root():
    return {"message": "RAG API is running"}