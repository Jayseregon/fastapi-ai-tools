from functools import lru_cache

from fastapi import APIRouter, Depends

from src.models.embedding import Keywords
from src.models.user import User
from src.security.jwt_auth import validate_token
from src.security.rateLimiter.depends import RateLimiter
from src.services.embedding import Embeddings, EmbeddingService

router = APIRouter(prefix="/v1/embedding", tags=["embedding"])


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@router.post("/keywords", response_model=Embeddings, status_code=201)
async def create_embeddings(
    keywords: Keywords,
    current_user: User = Depends(validate_token),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    rate: None = Depends(RateLimiter(times=3, seconds=10)),
):
    """Create embeddings from a list of keywords."""
    return embedding_service.process_keywords(keywords.keywords)
