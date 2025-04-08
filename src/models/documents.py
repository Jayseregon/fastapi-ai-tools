from typing import Dict, List

from pydantic import BaseModel


class StoreMetadata(BaseModel):
    nb_collections: int
    details: Dict[str, Dict[str, int]]

    class Config:
        from_attributes = True


class AddPDFResponse(BaseModel):
    status: str
    filename: str
    store_metadata: StoreMetadata
    added_count: int
    skipped_count: int
    skipped_sources: List[str]


class UpdatePDFResponse(BaseModel):
    status: str
    filename: str
    store_metadata: StoreMetadata
    added_count: int
    docs_replaced: int
    sources_updated: int
