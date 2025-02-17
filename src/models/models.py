from typing import List

from pydantic import BaseModel


class Keywords(BaseModel):
    keywords: list[str]


class EmbeddedKeyword(BaseModel):
    word: str
    x: float
    y: float


class Embeddings(BaseModel):
    keywords: List[EmbeddedKeyword]
