from typing import Dict, List

from pydantic import BaseModel, Field, RootModel


class ChromaStatus(BaseModel):
    status: str = Field(..., description="Status of the ChromaDB service")
    message: str = Field(
        ..., description="Descriptive message about the service status"
    )
    heartbeat: int = Field(..., description="Heartbeat value from ChromaDB")


class CollectionsResponse(RootModel):
    """Response model for collections and their document counts"""

    root: Dict[str, int] = Field(
        ..., description="Dictionary mapping collection names to their document counts"
    )


class DeleteCollectionRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the collection to delete")


class DeleteCollectionResponse(BaseModel):
    status: str = Field(..., description="Result status, e.g., 'success' or 'error'")
    message: str = Field(..., description="Descriptive message about the operation")


class CollectionSourcesResponse(BaseModel):
    """Response model for collections and their document sources"""

    collections: Dict[str, List[str]] = Field(
        ...,
        description="Dictionary mapping collection names to lists of their unique document sources",
    )


class DeleteSourceRequest(BaseModel):
    collection_name: str = Field(
        ..., description="Name of the collection containing the source"
    )
    source_name: str = Field(
        ..., description="Name of the source to delete (filename or full URL)"
    )


class DeleteSourceResponse(BaseModel):
    status: str = Field(..., description="Result status, e.g., 'success' or 'error'")
    message: str = Field(..., description="Descriptive message about the operation")
    documents_deleted: int = Field(..., description="Number of documents deleted")
