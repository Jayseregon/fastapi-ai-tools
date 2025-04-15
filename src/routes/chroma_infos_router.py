import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.models.chroma_infos_models import ChromaStatus, CollectionsResponse
from src.security.rateLimiter.depends import RateLimiter
from src.services.db import chroma_service

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/v1/chroma-infos", tags=["ChromaDB"])


@router.get("/ping", response_model=ChromaStatus)
async def ping_chroma(
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> ChromaStatus:
    """Ping the ChromaDB service.

    This endpoint verifies connectivity to the ChromaDB vector database service
    by checking its heartbeat. Useful for system health checks and ensuring
    that the RAG system's vector store is operational.

    Args:
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        A ChromaStatus object containing:
            - status: "ok" if ChromaDB is available, "warning" if heartbeat is zero
            - message: A descriptive status message
            - heartbeat: The numeric heartbeat value from ChromaDB

    Raises:
        HTTPException: With 503 status code if ChromaDB service is unavailable
    """
    try:
        # Get client directly by calling the service instance
        client = chroma_service()
        heartbeat = client.heartbeat()

        if heartbeat > 0:
            return ChromaStatus(
                status="ok",
                message="ChromaDB is available",
                heartbeat=heartbeat,
            )
        return ChromaStatus(
            status="warning",
            message="ChromaDB returned zero heartbeat",
            heartbeat=heartbeat,
        )
    except Exception as e:
        logger.error(f"Error pinging ChromaDB: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"ChromaDB is not available: {str(e)}"
        )


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections(
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> CollectionsResponse:
    """List all collections in ChromaDB with their document counts.

    This endpoint retrieves all existing collections in the ChromaDB vector database
    and returns their names along with the number of documents stored in each.
    For agentic RAG systems, this provides a critical overview of available knowledge bases.

    Args:
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        A CollectionsResponse object mapping collection names to their document counts

    Raises:
        HTTPException: With 500 status code if the operation fails
    """
    try:
        client = chroma_service()
        collections = client.list_collections()
        collections_dict = {
            coll: client.get_collection(coll).count() for coll in collections
        }
        # Using the new RootModel properly
        return CollectionsResponse(root=collections_dict)
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )
