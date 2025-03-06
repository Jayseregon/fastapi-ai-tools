import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

from src.services.loaders.lib import DocumentLoader, HttpClient

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Base class for web content loaders."""

    def __init__(self, http_client=None, document_loader=None):
        """Initialize the web loader service."""
        self._http_client = http_client or HttpClient()
        self._document_loader = document_loader or DocumentLoader()
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
        self, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
    ) -> None:
        """Initialize the web loader service."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources asynchronously."""
        await self._http_client.close()
        self._initialized = False
