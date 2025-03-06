import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class HttpClientService:
    """Manages HTTP client lifecycle and requests with proper resource handling."""

    def __init__(
        self,
        default_headers: Optional[Dict[str, str]] = None,
        follow_redirects: bool = True,
        timeout: float = 30.0,
    ):
        """
        Initialize the HTTP client service.

        Args:
            default_headers: Default headers for all requests
            follow_redirects: Whether to follow HTTP redirects
            timeout: Default request timeout in seconds
        """
        self.client: httpx.AsyncClient = None
        self.headers = default_headers or {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        self.follow_redirects = follow_redirects
        self.timeout_duration = timeout
        self._lock = asyncio.Lock()
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(
        self, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
    ) -> None:
        """
        Initialize the HTTP client with custom connection pool.

        Args:
            headers: Optional headers to add/override defaults
            timeout: Optional timeout override
        """
        async with self._lock:
            if self._initialized:
                return

            timeout_value = timeout or self.timeout_duration

            self.client = httpx.AsyncClient(
                follow_redirects=self.follow_redirects,
                timeout=httpx.Timeout(timeout_value),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )

            if headers:
                self.headers.update(headers)

            self._initialized = True
            logger.debug(f"HTTP client initialized with {timeout_value}s timeout")

    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """
        Make GET request with error handling.

        Args:
            url: Target URL
            headers: Optional request-specific headers
            params: Optional query parameters

        Returns:
            HTTPX Response object

        Raises:
            httpx.HTTPError: If the request fails
        """
        if not self._initialized:
            await self.initialize()

        request_headers = self.headers.copy()
        if headers:
            request_headers.update(headers)

        try:
            logger.debug(f"GET request to {url}")
            response = await self.client.get(
                url, headers=request_headers, params=params
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during GET to {url}: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error during GET to {url}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during GET to {url}: {str(e)}")
            raise

    async def post(
        self,
        url: str,
        data: Any = None,
        json: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """
        Make POST request with error handling.

        Args:
            url: Target URL
            data: Optional form data
            json: Optional JSON data
            headers: Optional request-specific headers

        Returns:
            HTTPX Response object

        Raises:
            httpx.HTTPError: If the request fails
        """
        if not self._initialized:
            await self.initialize()

        request_headers = self.headers.copy()
        if headers:
            request_headers.update(headers)

        try:
            logger.debug(f"POST request to {url}")
            response = await self.client.post(
                url, data=data, json=json, headers=request_headers
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during POST to {url}: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error during POST to {url}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during POST to {url}: {str(e)}")
            raise

    async def close(self) -> None:
        """Clean up resources asynchronously."""
        async with self._lock:
            if self.client:
                await self.client.aclose()
                self.client = None
            self._initialized = False
            logger.debug("HTTP client closed and resources released")
