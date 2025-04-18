import logging
from typing import AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.documents import Document

from src.services.loaders.lib import HttpClient, WebAuthentication
from src.services.loaders.lib.url_discovery import UrlDiscovery
from src.services.loaders.web.base_web_loader import BaseWebLoader

logger = logging.getLogger(__name__)


class SeticsLoader(BaseWebLoader):
    """Service for loading content from Setics authenticated websites."""

    def __init__(self):
        """Initialize the Setics web loader service with component services."""
        super().__init__()
        self._auth_service = WebAuthentication()
        self._authenticated = False
        self.login_url = None
        self.check_url = None
        self.username = None
        self.password = None

    async def initialize(
        self, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
    ) -> None:
        """
        Initialize the Setics web loader service.

        Args:
            headers: Optional HTTP headers to include in requests
            timeout: Timeout in seconds for HTTP requests
        """
        # Define Setics-specific headers
        setics_headers = {
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1",
        }

        # Combine with provided headers
        if headers:
            setics_headers.update(headers)

        # Initialize HTTP client with Setics-specific configuration
        await self._http_client.initialize(headers=setics_headers, timeout=timeout)
        self._initialized = True
        logger.debug(f"Initialized Setics web loader service with {timeout}s timeout")

    async def authenticate(
        self,
        username: str,
        password: str,
        login_url: str,
        check_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> "SeticsLoader":
        """
        Authenticate with Setics website.

        Args:
            username: Setics username/email
            password: Setics password
            login_url: URL of the login page
            check_url: Optional URL to verify successful authentication
            headers: Optional additional headers for authentication requests

        Returns:
            Self for method chaining
        """
        if not self._initialized:
            await self.initialize(headers=headers)

        # Store credentials and URLs
        self.username = username
        self.password = password
        self.login_url = login_url
        self.check_url = check_url

        # Add form submission headers
        auth_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        # Extract domain from login URL for Origin
        parsed_url = urlparse(login_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        auth_headers["Origin"] = base_domain
        auth_headers["Referer"] = login_url

        if headers:
            auth_headers.update(headers)

        # Add headers to client
        for key, value in auth_headers.items():
            if key not in self._http_client.headers:
                self._http_client.headers[key] = value

        # Create credentials dict in format expected by Setics
        credentials = {"user[email]": username, "user[password]": password}

        # Perform complete authentication flow
        self._authenticated = await self._auth_service.complete_authentication_flow(
            http_client=self._http_client,
            login_url=login_url,
            credentials=credentials,
            check_url=check_url,
            # Specific settings for Setics
            token_field="authenticity_token",
            failure_strings=["Invalid Email or password"],
        )

        if not self._authenticated:
            raise ValueError("Failed to authenticate with Setics")

        logger.info("Successfully authenticated with Setics")
        return self

    async def load_documents(
        self,
        urls: str | List[str],
        continue_on_failure: bool = False,
    ) -> List[Document]:
        """
        Load documents from authenticated Setics URLs.

        Args:
            urls: Single URL or list of URLs to load
            continue_on_failure: Whether to continue if loading fails for some URLs

        Returns:
            List of loaded Documents

        Raises:
            ValueError: If not authenticated or if document loading fails
        """
        if not self._initialized:
            raise ValueError("Service must be initialized before loading documents")

        if not self._authenticated:
            raise ValueError("Authentication required before loading documents")

        try:
            # Use WebBaseLoader for compatibility with existing code
            return await self._document_loader.load_documents_with_langchain(
                http_client=self._http_client,
                urls=urls,
                continue_on_failure=continue_on_failure,
            )
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise ValueError(f"Failed to load documents: {str(e)}")

    async def lazy_load_documents(
        self,
        urls: str | List[str],
        continue_on_failure: bool = False,
    ) -> AsyncIterator[Document]:
        """
        Lazily load documents from authenticated Setics URLs.

        Args:
            urls: Single URL or list of URLs to load
            continue_on_failure: Whether to continue if loading fails for some URLs

        Yields:
            Documents as they are loaded

        Raises:
            ValueError: If not authenticated
        """
        if not self._initialized:
            raise ValueError("Service must be initialized before loading documents")

        if not self._authenticated:
            raise ValueError("Authentication required before loading documents")

        # Use WebBaseLoader for compatibility with existing code
        async for doc in self._document_loader.lazy_load_documents_with_langchain(
            http_client=self._http_client,
            urls=urls,
            continue_on_failure=continue_on_failure,
        ):
            yield doc

    async def discover_urls(
        self,
        base_url: str,
        max_depth: int = 2,
        same_domain_only: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Discover URLs by crawling from the base URL using an authenticated session.

        Args:
            base_url: The starting URL for discovery
            max_depth: Maximum depth to crawl (default: 2)
            same_domain_only: Whether to only crawl URLs on the same domain (default: True)
            headers: Optional additional headers for discovery requests

        Returns:
            List of discovered URLs

        Raises:
            ValueError: If not authenticated
        """
        if not self._initialized:
            raise ValueError("Service must be initialized before discovering URLs")

        if not self._authenticated:
            raise ValueError("Authentication required before discovering URLs")

        # Prepare headers
        discovery_headers = self.request_headers
        if headers:
            discovery_headers.update(headers)

        logger.debug(f"Starting URL discovery from {base_url} with depth {max_depth}")

        # Use UrlDiscovery as a context manager
        async with UrlDiscovery() as discovery:
            # Use the authenticated session from this loader with async_discover
            urls = await discovery.discover(
                base_url=base_url,
                session=self._http_client,
                headers=discovery_headers,
                max_depth=max_depth,
                same_domain_only=same_domain_only,
            )

        logger.info(f"Discovered {len(urls)} URLs from {base_url}")
        return urls

    @property
    def is_authenticated(self) -> bool:
        """Check if the service is authenticated."""
        return self._authenticated

    @property
    def authenticated_client(self) -> HttpClient:
        """Get the authenticated HTTP client service."""
        if not self._initialized or not self._authenticated:
            raise ValueError("Authentication required before accessing client")
        return self._http_client

    @property
    def request_headers(self) -> Dict[str, str]:
        """Get the current request headers."""
        if not self._initialized:
            raise ValueError("Service must be initialized before accessing headers")
        return self._http_client.headers.copy()

    async def close(self) -> None:
        """Clean up resources asynchronously."""
        await super().close()
        self._authenticated = False


# Factory function for global access
async def create_setics_web_loader_service() -> SeticsLoader:
    """Create and initialize a Setics web loader service."""
    service = SeticsLoader()
    await service.initialize()
    return service
