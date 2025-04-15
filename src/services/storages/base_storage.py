import logging
from abc import ABC, abstractmethod
from typing import Self

logger = logging.getLogger(__name__)


class BaseStorage(ABC):
    """Base class for storage systems."""

    def __init__(self):
        """Initialize the storage system."""

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources if needed."""
        pass
