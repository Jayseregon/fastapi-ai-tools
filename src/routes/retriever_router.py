import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.configs.env_config import config
from src.models.retriever_models import (
    DocumentMetadata,
    QueryRequest,
    RetrievedDocument,
    RetrieverResponse,
)
from src.models.user import User
from src.security.jwt_auth import validate_token
from src.security.rateLimiter.depends import RateLimiter
from src.services.retrievers import MultiQRerankedRetriever

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/retriever", tags=["Vectorstore Retriever"])


@router.post(
    "/base_collection/invoke", response_model=RetrieverResponse, status_code=200
)
async def query_base_collection(
    request: QueryRequest,
    current_user: User = Depends(validate_token),
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> RetrieverResponse:
    """
    Query the base vector store collection for relevant documents.

    This endpoint retrieves documents from the default vector database collection
    that semantically match the provided query. It uses MultiQRerankedRetriever,
    which enhances retrieval quality through multiple query formulations and reranking.

    Designed for integration as a tool for AI agents and RAG (Retrieval Augmented Generation)
    systems, it provides relevant context for answering user queries.

    Args:
        request: The request object containing the user query.
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds).

    Returns:
        RetrieverResponse: Object containing retrieved documents, including:
            - documents: List of matching documents with their content and metadata.

    Raises:
        HTTPException: If there's an error querying the vector store.
    """
    try:
        logger.info(f"Received query: {request.query}")
        retriever = MultiQRerankedRetriever()
        results = await retriever(
            query=request.query, collection_name=config.COLLECTION_NAME
        )

        # Convert langchain Documents to your Pydantic model
        documents = [
            RetrievedDocument(
                metadata=DocumentMetadata(**doc.metadata), page_content=doc.page_content
            )
            for doc in results
        ]
        return RetrieverResponse(documents=documents)
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}")
        raise HTTPException(status_code=500, detail="Error querying vector store")


@router.post(
    "/setics_collection/invoke", response_model=RetrieverResponse, status_code=200
)
async def query_setics_collection(
    request: QueryRequest,
    current_user: User = Depends(validate_token),
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> RetrieverResponse:
    """
    Query the Setics vector store collection for relevant documents.

    This endpoint retrieves documents from the Setics-specific vector database collection
    that semantically match the provided query. It uses MultiQRerankedRetriever,
    which enhances retrieval quality through multiple query formulations and reranking.

    Designed for integration as a tool for AI agents and RAG (Retrieval Augmented Generation)
    systems, it provides relevant context for answering user queries.

    Args:
        request: The request object containing the user query.
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds).

    Returns:
        RetrieverResponse: Object containing retrieved documents, including:
            - documents: List of matching documents with their content and metadata.

    Raises:
        HTTPException: If there's an error querying the vector store.
    """
    try:
        logger.info(f"Received query: {request.query}")
        retriever = MultiQRerankedRetriever()
        results = await retriever(
            query=request.query, collection_name=config.SETICS_COLLECTION
        )

        # Convert langchain Documents to your Pydantic model
        documents = [
            RetrievedDocument(
                metadata=DocumentMetadata(**doc.metadata), page_content=doc.page_content
            )
            for doc in results
        ]
        return RetrieverResponse(documents=documents)
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}")
        raise HTTPException(status_code=500, detail="Error querying vector store")
