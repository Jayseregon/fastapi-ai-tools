from fastapi import FastAPI

from .models.models import Keywords
from .services.keywordEmbedding import Embeddings, EmbeddingService

app = FastAPI()


@app.get("/")
async def read_root():
    return {"greatings": "Welcome to the keyword embeddings API!"}


@app.post("/embeddings", response_model=Embeddings)
async def create_embeddings(keywords: Keywords):
    embedding_service = EmbeddingService()
    return embedding_service.process_keywords(keywords.keywords)
