import logging
from typing import AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SessionAdapter:
    def __init__(self, client, cookies, headers, timeout=30.0):
        """Initialize SessionAdapter with explicit timeout."""
        self.client = client
        self.cookies = type("Cookies", (), {"get_dict": lambda self: cookies})()
        self.headers = headers
        self.verify = True  # SSL verification
        self.timeout = timeout


class AuthWebLoaderBase:
    """Base service for authenticated web content loading."""

    def __init__(self):
        """Initialize the base authenticated web loader service."""
        self.client = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        self._initialized = False
        self.last_login_status = False

    async def initialize(
        self,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = 30.0,
    ) -> None:
        """
        Initialize the authenticated web loader service.

        Args:
            headers: Optional HTTP headers to include in requests
            timeout: Timeout in seconds for HTTP requests
        """
        if self._initialized:
            return

        if self.client is None:
            self.client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(timeout),
            )

        if headers:
            self.headers.update(headers)

        self._initialized = True
        logger.debug(f"{self.__class__.__name__} initialized with {timeout}s timeout")

    def _extract_token(
        self, html_content: str, token_field: str = "authenticity_token"
    ) -> Optional[str]:
        """
        Extract authentication token from HTML content.

        Args:
            html_content: HTML content to parse
            token_field: Name of the token field to look for

        Returns:
            Token value if found, None otherwise
        """
        soup = BeautifulSoup(html_content, "html.parser")
        token_input = soup.find("input", attrs={"name": token_field})
        if token_input and token_input.get("value"):
            token_value = token_input.get("value")
            logger.debug(f"Found {token_field}: {token_value[:10]}...")
            return token_value
        return None

    async def _extract_domain_aware_cookies(self, urls: List[str]) -> Dict[str, str]:
        """Extract cookies with domain awareness."""
        # No lock needed here as this is called from methods that already hold the lock
        cookie_dict: Dict[str, str] = {}
        if not self.client or not hasattr(self.client, "cookies"):
            return cookie_dict

        # Extract the first URL's domain to get proper cookies
        target_domain = urlparse(urls[0]).netloc if urls else None

        # Get all cookies from jar
        all_cookies = (
            self.client.cookies.jar._cookies
            if hasattr(self.client.cookies, "jar")
            else {}
        )

        # Build cookie dict with domain preferences
        for domain in all_cookies:
            for path in all_cookies[domain]:
                for name, cookie_obj in all_cookies[domain][path].items():
                    # Prefer target domain cookies, overwrite others
                    if target_domain and target_domain in domain:
                        cookie_dict[name] = cookie_obj.value
                    elif name not in cookie_dict:
                        cookie_dict[name] = cookie_obj.value

        return cookie_dict

    def _create_session_adapter(self, client, cookies, headers, timeout=30.0):
        """Create a session adapter with explicit timeout."""
        return SessionAdapter(client, cookies, headers, timeout)

    async def create_langchain_loader(
        self,
        urls: str | List[str],
        continue_on_failure: bool = False,
        timeout: Optional[float] = 30.0,
    ) -> WebBaseLoader:
        """
        Create a LangChain WebBaseLoader configured with the authenticated session.

        Args:
            urls: Single URL or list of URLs to load
            continue_on_failure: Whether to continue when errors occur
            timeout: Timeout for requests in seconds

        Returns:
            Configured WebBaseLoader instance
        """
        if not self._initialized:
            await self.initialize(timeout=timeout)

        if isinstance(urls, str):
            urls = [urls]

        # Extract cookies
        try:
            cookie_dict = await self._extract_domain_aware_cookies(urls)
        except Exception as e:
            logger.warning(f"Cookie extraction failed: {str(e)}")
            cookie_dict = {}

        # Create adapter
        adapter = self._create_session_adapter(
            client=self.client,
            cookies=cookie_dict,
            headers=self.headers,
            timeout=timeout,
        )

        # Set request kwargs with timeout
        requests_kwargs = {
            "headers": self.headers,
            "timeout": timeout,
        }

        # Create and configure loader
        loader = WebBaseLoader(
            web_paths=urls,
            session=adapter,
            requests_kwargs=requests_kwargs,
            continue_on_failure=continue_on_failure,
        )

        return loader

    async def load_documents(
        self,
        urls: str | List[str],
        timeout: Optional[float] = 30.0,
    ) -> List[Document]:
        """
        Load documents from URLs using the async capabilities of WebBaseLoader.

        Args:
            urls: Single URL or list of URLs to load
            timeout: Timeout for requests in seconds

        Returns:
            List of loaded documents
        """

        loader = await self.create_langchain_loader(urls, timeout=timeout)

        documents: List[Document] = []
        async for doc in loader.alazy_load():
            documents.append(doc)
        return documents

    async def lazy_load_documents(
        self, urls: str | List[str]
    ) -> AsyncIterator[Document]:
        """
        Lazily load documents from URLs using the async capabilities of WebBaseLoader.

        Args:
            urls: Single URL or list of URLs to load

        Yields:
            Documents as they are loaded
        """
        loader = await self.create_langchain_loader(urls)

        async for doc in loader.alazy_load():
            yield doc

    async def close(self):
        """Clean up resources asynchronously."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.headers.clear()
        self._initialized = False
        self.last_login_status = False
