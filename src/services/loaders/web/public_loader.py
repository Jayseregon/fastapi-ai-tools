import logging
from typing import AsyncIterator, Dict, List, Optional

from langchain_core.documents import Document

from src.services.loaders.web.base_web_loader import BaseWebLoader
from src.services.loaders.web.web_image_loader import create_web_image_loader

logger = logging.getLogger(__name__)


class PublicLoader(BaseWebLoader):
    """Service for loading content from public websites."""

    def __init__(self):
        """Initialize the public web loader service."""
        super().__init__()

    async def initialize(
        self, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
    ) -> None:
        """
        Initialize the public web loader service.

        Args:
            headers: Optional HTTP headers to include in requests
            timeout: Timeout in seconds for HTTP requests
        """
        # Define public site friendly headers
        public_headers = {
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        # Combine with provided headers
        if headers:
            public_headers.update(headers)

        # Initialize HTTP client
        await self._http_client.initialize(headers=public_headers, timeout=timeout)
        self._initialized = True
        logger.debug(f"Initialized public web loader service with {timeout}s timeout")

    async def load_multi_documents(
        self, urls: List[str], continue_on_failure: bool = True
    ) -> List[Document]:
        """
        Load documents from public URLs using LangChain's WebBaseLoader.

        Args:
            urls: a list of URLs to load
            continue_on_failure: Whether to continue if loading fails for some URLs

        Returns:
            List of loaded Documents
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use WebBaseLoader for compatibility with existing code
            return await self._document_loader.load_documents_with_langchain(
                http_client=self._http_client,
                urls=urls,
                continue_on_failure=continue_on_failure,
            )
        except Exception as e:
            logger.error(f"Error loading public documents: {str(e)}")
            if not continue_on_failure:
                raise
            return []

    async def load_single_document(self, url: str) -> Document:
        """
        Load a single document from a public URL.

        Args:
            url: URL string to load content from

        Returns:
            A Document object containing the webpage content or None on error
        """
        if not self._initialized:
            await self.initialize()

        try:
            documents = await self._document_loader.load_documents_with_langchain(
                http_client=self._http_client,
                urls=url,
                continue_on_failure=False,
            )
            return documents[0] if documents else Document(page_content="", metadata={})
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            return Document(page_content="", metadata={})

    async def load_single_document_with_images(self, url: str) -> List[Document]:
        """
        Load a webpage document along with any images found at the URL.

        Args:
            url: URL string to load content and images from

        Returns:
            A list of Document objects containing the webpage content and image content,
            or None on error
        """
        if not self._initialized:
            await self.initialize()

        documents = []
        try:
            single_doc = await self.load_single_document(url=url)
            documents.append(single_doc)

            img_loader = await create_web_image_loader()
            img_docs = await img_loader.download_and_parse_images(urls=url)
            documents.extend(img_docs)

            return documents
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            return []

    async def lazy_load_multi_documents(
        self, urls: str | List[str], continue_on_failure: bool = True
    ) -> AsyncIterator[Document]:
        """
        Lazily load documents from public URLs using LangChain's WebBaseLoader.

        Args:
            urls: Single URL or list of URLs to load
            continue_on_failure: Whether to continue if loading fails for some URLs

        Yields:
            Documents as they are loaded
        """
        if not self._initialized:
            await self.initialize()

        # Use lazy loading from document loader with LangChain
        async for doc in self._document_loader.lazy_load_documents_with_langchain(
            http_client=self._http_client,
            urls=urls,
            continue_on_failure=continue_on_failure,
        ):
            yield doc

    async def close(self) -> None:
        """Clean up resources asynchronously."""
        await super().close()


# Factory function for global access
async def create_public_web_loader_service() -> PublicLoader:
    """Create and initialize a public web loader service."""
    service = PublicLoader()
    await service.initialize()
    return service
