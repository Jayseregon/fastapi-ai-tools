from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    id: Optional[str] = None
    producer: Optional[str] = None
    creator: Optional[str] = None
    creationdate: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    total_pages: Optional[int] = None
    format: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    moddate: Optional[str] = None
    trapped: Optional[str] = None
    modDate: Optional[str] = None
    creationDate: Optional[str] = None
    timestamp: Optional[str] = None
    page: Optional[int] = None
    document_type: Optional[str] = None
    description: Optional[str] = None


class StoreMetadata(BaseModel):
    nb_collections: int
    details: Dict[str, Dict[str, int]]

    class ConfigDict:
        from_attributes = True


class AddDocumentsResponse(BaseModel):
    status: str
    filename: str
    store_metadata: StoreMetadata
    added_count: int
    skipped_count: int
    skipped_sources: List[str]
    doc_sample_meta: Optional[DocumentMetadata] = None


class UpdateDocumentsResponse(BaseModel):
    status: str
    filename: str
    store_metadata: StoreMetadata
    added_count: int
    docs_replaced: int
    sources_updated: int
    doc_sample_meta: Optional[DocumentMetadata] = None


class WebUrlRequest(BaseModel):
    web_url: str = Field(
        ...,
        description="The web url to be loaded as document object in the vector store",
        min_length=1,
    )
    with_images: Optional[bool] = Field(
        False,
        description="Whether to load images present in the web page, as independent documents",
    )
