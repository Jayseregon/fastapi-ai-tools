from functools import lru_cache

from fastapi import APIRouter, Depends

from src.models.models import Keywords
from src.services.keywordEmbedding import Embeddings, EmbeddingService

router = APIRouter(prefix="/v1/embedding", tags=["embedding"])


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Create an instance of the EmbeddingService class."""
    return EmbeddingService()


@router.post("/keywords", response_model=Embeddings, status_code=201)
async def create_embeddings(
    keywords: Keywords,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Create embeddings from a list of keywords."""
    return embedding_service.process_keywords(keywords.keywords)
