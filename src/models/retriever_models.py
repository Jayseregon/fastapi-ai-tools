from typing import List

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    id: str
    relevance_score: float
    description: str = ""
    title: str
    document_type: str
    source: str


class RetrievedDocument(BaseModel):
    metadata: DocumentMetadata
    page_content: str


class RetrieverResponse(BaseModel):
    documents: List[RetrievedDocument] = Field(
        ..., description="List of retrieved documents with their metadata and content"
    )


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The search query to find matching documents in the vector store",
        min_length=1,
    )
