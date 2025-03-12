import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Self, Union

from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """Base class for document file loaders."""

    def __init__(self):
        """Initialize the document loader."""
        self._llm_model = None
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @abstractmethod
    async def initialize(
        self, llm_model: Optional[BaseChatModel] = None, **kwargs
    ) -> Self:
        """Initialize the document loader with specific parameters."""
        pass

    @abstractmethod
    async def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a document from a file path."""
        pass

    async def close(self) -> None:
        """Clean up resources if needed."""
        self._llm_model = None
        self._initialized = False
