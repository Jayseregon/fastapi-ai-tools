import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.models.chroma_infos_models import (
    ChromaStatus,
    CollectionSourcesResponse,
    CollectionsResponse,
    DeleteCollectionRequest,
    DeleteCollectionResponse,
    DeleteSourceRequest,
    DeleteSourceResponse,
)
from src.models.user import User
from src.security.jwt_auth import validate_token
from src.security.rateLimiter.depends import RateLimiter
from src.services.db import chroma_service
from src.services.vectorstore.chroma_store import ChromaStore

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/v1/chroma-infos", tags=["ChromaDB"])


@router.get("/ping", response_model=ChromaStatus)
async def ping_chroma(
    current_user: User = Depends(validate_token),
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
    current_user: User = Depends(validate_token),
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
        logger.exception(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )


@router.post("/delete-collection", response_model=DeleteCollectionResponse)
async def delete_collection(
    req: DeleteCollectionRequest,
    current_user: User = Depends(validate_token),
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> DeleteCollectionResponse:
    """
    Delete a collection from ChromaDB by name.

    Args:
        req: DeleteCollectionRequest containing the collection name.
        current_user: User object obtained from the token validation.
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        DeleteCollectionResponse indicating success or failure.

    Raises:
        HTTPException: With 404 if collection does not exist, 500 for other errors.
    """
    try:
        client = chroma_service()
        collections = client.list_collections()
        if req.collection_name not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' does not exist.",
            )
        client.delete_collection(req.collection_name)
        return DeleteCollectionResponse(
            status="success",
            message=f"Collection '{req.collection_name}' deleted successfully.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection '{req.collection_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection '{req.collection_name}': {str(e)}",
        )


@router.get("/collections/list-sources", response_model=CollectionSourcesResponse)
async def get_collections_with_sources(
    current_user: User = Depends(validate_token),
    rate: Optional[None] = Depends(RateLimiter(times=2, seconds=10)),
) -> CollectionSourcesResponse:
    """
    Get all collections and their unique document sources.

    This endpoint retrieves all existing collections in ChromaDB and returns a list
    of unique document sources (filenames or URLs) for each collection. This provides
    a comprehensive view of what documents are stored in each knowledge base.

    Args:
        is_web: If True, treat sources as web URLs rather than file paths
        current_user: User making the request (from token)
        rate: Rate limiter dependency

    Returns:
        A CollectionSourcesResponse mapping collections to their document sources

    Raises:
        HTTPException: With 500 status code if the operation fails
    """
    try:
        chroma_store = ChromaStore()
        collections_sources = await chroma_store.get_collections_with_sources()

        return CollectionSourcesResponse(collections=collections_sources)
    except Exception as e:
        logger.exception(f"Failed to get collections with sources: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get collections with sources: {str(e)}"
        )


@router.post("/collections/delete-source", response_model=DeleteSourceResponse)
async def delete_source(
    req: DeleteSourceRequest,
    current_user: User = Depends(validate_token),
    rate: Optional[None] = Depends(RateLimiter(times=2, seconds=10)),
) -> DeleteSourceResponse:
    """
    Delete all documents from a specific source in a collection.

    This endpoint removes all document chunks that originate from a specific source file
    or URL. This is useful for removing outdated documents before replacing them with
    updated versions, or for permanently removing a source from the knowledge base.

    The source_name can be either:
    - A filename (for document files like PDFs or text files)
    - A complete URL (for web content)

    Args:
        req: DeleteSourceRequest with collection name and source name
        current_user: User making the request (from token)
        rate: Rate limiter dependency

    Returns:
        A DeleteSourceResponse with status and count of documents deleted

    Raises:
        HTTPException: With 404 if collection doesn't exist, 500 for other errors
    """
    try:
        # Verify that the collection exists
        client = chroma_service()
        collections = client.list_collections()
        if req.collection_name not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' does not exist.",
            )

        # Delete the source documents
        chroma_store = ChromaStore()
        docs_deleted = await chroma_store.delete_source_documents(
            collection_name=req.collection_name, source_name=req.source_name
        )

        if docs_deleted == 0:
            return DeleteSourceResponse(
                status="warning",
                message=f"No documents found for source '{req.source_name}' in collection '{req.collection_name}'.",
                documents_deleted=0,
            )

        return DeleteSourceResponse(
            status="success",
            message=f"Successfully deleted source '{req.source_name}' in collection '{req.collection_name}'.",
            documents_deleted=docs_deleted,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete source documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete source documents: {str(e)}"
        )
