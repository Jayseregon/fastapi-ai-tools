import logging
from typing import Dict, List, Optional, Union

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

logger = logging.getLogger(__name__)


class AuthWebLoaderBase:
    """Base service for authenticated web content loading."""

    def __init__(self):
        """Initialize the base authenticated web loader service."""
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        self._initialized = False
        self.last_login_status = False

    def initialize(
        self,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the authenticated web loader service.

        Args:
            headers: Optional HTTP headers to include in requests
        """
        if headers:
            self.headers.update(headers)
        self._initialized = True
        logger.debug(f"{self.__class__.__name__} initialized")

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

    def create_langchain_loader(self, urls: Union[str, List[str]]) -> WebBaseLoader:
        """
        Create a LangChain WebBaseLoader configured with the authenticated session.

        Args:
            urls: Single URL or list of URLs to load

        Returns:
            Configured WebBaseLoader instance
        """
        if not self._initialized:
            self.initialize()

        if isinstance(urls, str):
            urls = [urls]

        loader = WebBaseLoader(
            web_paths=urls,
            session=self.session,
            requests_kwargs={"headers": self.headers},
        )

        logger.debug(f"Created WebBaseLoader for {len(urls)} URLs")
        return loader

    def close(self):
        """Clean up resources."""
        if self.session:
            self.session.close()
            self.headers.clear()
        self._initialized = False
        self.last_login_status = False


# import httpx
# import asyncio
# from bs4 import BeautifulSoup
# import logging
# from typing import Optional, List, Dict, Union
# from langchain_community.document_loaders import WebBaseLoader

# logger = logging.getLogger(__name__)


# class AuthWebLoaderBase:
#     """Base service for authenticated web content loading."""

#     def __init__(self):
#         """Initialize the base authenticated web loader service."""
#         self.client = None  # Will initialize httpx.AsyncClient when needed
#         self.headers = {
#             "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15",
#             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#         }
#         self._initialized = False
#         self.last_login_status = False
#         self._lock = asyncio.Lock()  # Add lock to prevent race conditions

#     async def initialize(
#         self,
#         headers: Optional[Dict[str, str]] = None,
#     ) -> None:
#         """
#         Initialize the authenticated web loader service.

#         Args:
#             headers: Optional HTTP headers to include in requests
#         """
#         async with self._lock:
#             if self._initialized:
#                 return

#             if self.client is None:
#                 self.client = httpx.AsyncClient(follow_redirects=True)

#             if headers:
#                 self.headers.update(headers)

#             self._initialized = True
#             logger.debug(f"{self.__class__.__name__} initialized")

#     def _extract_token(
#         self, html_content: str, token_field: str = "authenticity_token"
#     ) -> Optional[str]:
#         """
#         Extract authentication token from HTML content.

#         Args:
#             html_content: HTML content to parse
#             token_field: Name of the token field to look for

#         Returns:
#             Token value if found, None otherwise
#         """
#         soup = BeautifulSoup(html_content, "html.parser")
#         token_input = soup.find("input", attrs={"name": token_field})
#         if token_input and token_input.get("value"):
#             token_value = token_input.get("value")
#             logger.debug(f"Found {token_field}: {token_value[:10]}...")
#             return token_value
#         return None

#     async def fetch_page(self, url: str) -> str:
#         """
#         Fetch a page using the authenticated session.

#         Args:
#             url: URL to fetch

#         Returns:
#             HTML content of the page
#         """
#         if not self._initialized:
#             await self.initialize()

#         response = await self.client.get(url, headers=self.headers)
#         response.raise_for_status()
#         return response.text

#     def create_langchain_loader(self, urls: Union[str, List[str]]) -> WebBaseLoader:
#         """
#         Create a LangChain WebBaseLoader configured with the authenticated session.
#         Note: This is a non-async method as LangChain's WebBaseLoader doesn't support async.

#         Args:
#             urls: Single URL or list of URLs to load

#         Returns:
#             Configured WebBaseLoader instance
#         """
#         if isinstance(urls, str):
#             urls = [urls]

#         # Create a synchronous requests session from our async settings
#         sync_session = httpx.Client(
#             follow_redirects=True,
#             headers=self.headers,
#             cookies=self.client.cookies if self.client else {}
#         )

#         loader = WebBaseLoader(
#             web_paths=urls,
#             session=sync_session,
#         )

#         logger.debug(f"Created WebBaseLoader for {len(urls)} URLs")
#         return loader

#     async def create_langchain_loader_async(self, urls: Union[str, List[str]]) -> WebBaseLoader:
#         """
#         Create a LangChain WebBaseLoader configured with the authenticated session.
#         Ensures the client is initialized before creating the loader.

#         Args:
#             urls: Single URL or list of URLs to load

#         Returns:
#             Configured WebBaseLoader instance
#         """
#         if not self._initialized:
#             await self.initialize()

#         return self.create_langchain_loader(urls)

#     async def close(self):
#         """Clean up resources asynchronously."""
#         async with self._lock:
#             if self.client:
#                 await self.client.aclose()
#                 self.client = None
#             self.headers.clear()
#             self._initialized = False
#             self.last_login_status = False
