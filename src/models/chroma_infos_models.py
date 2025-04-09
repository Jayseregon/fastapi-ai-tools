from typing import Dict

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

    # Define a model_config to provide schema information
    model_config = {
        "json_schema_extra": {"examples": [{"collection1": 10, "collection2": 25}]}
    }
