from fastapi import FastAPI

from src.routes.embeddingRouter import router as embedding_router

app = FastAPI()


@app.get("/")
async def read_root():
    return {"greatings": "Welcome to the keyword embeddings API!"}


app.include_router(embedding_router)
