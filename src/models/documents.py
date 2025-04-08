from typing import Dict, List, Optional

from pydantic import BaseModel


# TODO: remove after testing
class DocumentMetadata(BaseModel):
    producer: Optional[str] = None
    creator: Optional[str] = None
    creationdate: Optional[str] = None
    source: Optional[str] = None
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
    page: Optional[int] = None


class StoreMetadata(BaseModel):
    nb_collections: int
    details: Dict[str, Dict[str, int]]

    class ConfigDict:
        from_attributes = True


class AddPDFResponse(BaseModel):
    status: str
    filename: str
    store_metadata: StoreMetadata
    added_count: int
    skipped_count: int
    skipped_sources: List[str]
    doc_sample_meta: Optional[DocumentMetadata] = None


class UpdatePDFResponse(BaseModel):
    status: str
    filename: str
    store_metadata: StoreMetadata
    added_count: int
    docs_replaced: int
    sources_updated: int
    doc_sample_meta: Optional[DocumentMetadata] = None
