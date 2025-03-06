import logging
from typing import AsyncIterator, Dict, List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from src.services.loaders.lib.cookie_manager import CookieManager
from src.services.loaders.lib.http_client import HttpClient
from src.services.loaders.lib.session_adapter import SessionAdapter

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles document loading from HTML content and conversion to LangChain documents."""

    def __init__(self, default_parser: str = "html.parser"):
        """
        Initialize the document loader service.

        Args:
            default_parser: Default parser to use with BeautifulSoup
        """
        self.default_parser = default_parser
        self.cookieManager = CookieManager()

    def _create_session_adapter(
        self, http_client: HttpClient, cookies: Dict[str, str]
    ) -> SessionAdapter:
        """
        Create a session adapter for WebBaseLoader compatibility.

        Args:
            http_client: HTTP client to adapt
            cookies: Cookies to include in session

        Returns:
            Session adapter compatible with WebBaseLoader
        """
        return SessionAdapter(
            client=http_client.client,
            cookies=cookies,
            headers=http_client.headers,
            timeout=http_client.timeout_duration,
        )

    async def create_langchain_loader(
        self,
        http_client: HttpClient,
        urls: str | List[str],
        continue_on_failure: bool = True,
    ) -> WebBaseLoader:
        """
        Create a LangChain WebBaseLoader configured with the HTTP client.

        Args:
            http_client: HTTP client for requests
            urls: URLs to load
            continue_on_failure: Whether to continue on errors

        Returns:
            Configured WebBaseLoader instance
        """
        if isinstance(urls, str):
            urls = [urls]

        try:
            # Use the CookieManager to extract cookies
            cookies = await self.cookieManager.extract_domain_cookies(http_client, urls)
        except Exception as e:
            logger.warning(f"Cookie extraction failed: {str(e)}, using empty cookies")
            cookies = {}

        # Create session adapter
        adapter = self._create_session_adapter(http_client, cookies)

        # Create loader
        loader = WebBaseLoader(
            web_paths=urls,
            session=adapter,
            requests_kwargs={"headers": http_client.headers},
            continue_on_failure=continue_on_failure,
        )

        logger.debug(f"Created WebBaseLoader for {len(urls)} URLs")
        return loader

    async def load_documents_with_langchain(
        self,
        http_client: HttpClient,
        urls: str | List[str],
        continue_on_failure: bool = True,
    ) -> List[Document]:
        """
        Load documents using LangChain's WebBaseLoader.

        Args:
            http_client: HTTP client for requests
            urls: URLs to load
            continue_on_failure: Whether to continue on errors

        Returns:
            List of loaded documents
        """
        loader = await self.create_langchain_loader(
            http_client=http_client, urls=urls, continue_on_failure=continue_on_failure
        )

        documents = []
        async for doc in loader.alazy_load():
            documents.append(doc)

        return documents

    async def lazy_load_documents_with_langchain(
        self,
        http_client: HttpClient,
        urls: str | List[str],
        continue_on_failure: bool = True,
    ) -> AsyncIterator[Document]:
        """
        Lazily load documents using LangChain's WebBaseLoader.

        Args:
            http_client: HTTP client for requests
            urls: URLs to load
            continue_on_failure: Whether to continue on errors

        Yields:
            Documents as they are loaded
        """
        loader = await self.create_langchain_loader(
            http_client=http_client, urls=urls, continue_on_failure=continue_on_failure
        )

        async for doc in loader.alazy_load():
            yield doc
