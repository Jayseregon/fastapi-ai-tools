import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from src.services.utils.auth_web_loader_base import AuthWebLoaderBase

logger = logging.getLogger(__name__)


class PublicWebLoaderService(AuthWebLoaderBase):
    """Service for loading content from public websites without authentication requirements."""

    def __init__(self):
        """Initialize the public web loader service."""
        super().__init__()
        self.headers.update(
            {
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    async def __call__(self, headers: Optional[Dict[str, str]] = None):
        """
        Make the service callable to maintain consistent interface with other loaders.

        Args:
            headers: Optional HTTP headers to override defaults

        Returns:
            Self reference for chaining
        """
        if not self._initialized:
            await self.initialize(headers=headers)
        return self

    async def load_documents_from_urls(self, urls: str | List[str]) -> List[Document]:
        """
        Convenience method to load documents from URLs.

        Args:
            urls: Single URL or list of URLs to load

        Returns:
            List of loaded documents
        """
        if not self._initialized:
            await self.__call__()

        return await self.load_documents(urls)


# Create a singleton instance
public_web_loader_service = PublicWebLoaderService()
