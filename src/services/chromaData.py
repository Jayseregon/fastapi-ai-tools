import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.db import chroma_service

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/v1/chroma", tags=["ChromaDB"])


# Models for requests
class Document(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    collection_name: str
    query_texts: List[str]
    n_results: int = 5


class CollectionCreate(BaseModel):
    name: str
    metadata: Optional[Dict[str, Any]] = None


class AddDocumentsRequest(BaseModel):
    collection_name: str
    documents: List[Document]


@router.get("/ping")
async def ping_chroma():
    """Ping the ChromaDB service."""
    try:
        # Get client directly by calling the service instance
        client = chroma_service()
        heartbeat = client.heartbeat()

        if heartbeat > 0:
            return {
                "status": "ok",
                "message": "ChromaDB is available",
                "heartbeat": heartbeat,
            }
        return {"status": "warning", "message": "ChromaDB returned zero heartbeat"}
    except Exception as e:
        logger.error(f"Error pinging ChromaDB: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"ChromaDB is not available: {str(e)}"
        )


@router.get("/collections")
async def list_collections():
    """List all collections in ChromaDB"""
    try:
        client = chroma_service()
        collections = client.list_collections()
        return {"collections": [c.name for c in collections]}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )


@router.post("/collections")
async def create_collection(collection_data: CollectionCreate):
    """Create a new collection in ChromaDB"""
    try:
        client = chroma_service()
        collection = client.create_collection(
            name=collection_data.name, metadata=collection_data.metadata
        )
        return {
            "status": "success",
            "collection_name": collection.name,
            "message": "Collection created successfully",
        }
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create collection: {str(e)}"
        )


@router.post("/add")
async def add_documents(request: AddDocumentsRequest):
    """Add documents to a collection"""
    try:
        client = chroma_service()
        collection = client.get_collection(request.collection_name)

        # Extract data from request
        ids = [doc.id for doc in request.documents]
        documents = [doc.content for doc in request.documents]
        metadatas = [doc.metadata for doc in request.documents]

        # Add documents to collection
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        return {
            "status": "success",
            "message": f"Added {len(documents)} documents to collection {request.collection_name}",
        }
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to add documents: {str(e)}"
        )


@router.post("/query")
async def query_collection(query: QueryRequest):
    """Query documents from a collection"""
    try:
        client = chroma_service()
        collection = client.get_collection(query.collection_name)

        results = collection.query(
            query_texts=query.query_texts, n_results=query.n_results
        )

        return {"results": results}
    except Exception as e:
        logger.error(f"Failed to query collection: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to query collection: {str(e)}"
        )
