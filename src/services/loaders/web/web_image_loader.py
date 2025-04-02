import logging
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

from src.services.loaders.lib import WebAuthentication
from src.services.loaders.lib.web_image_processor import WebImageProcessor
from src.services.loaders.web.base_web_loader import BaseWebLoader

logger = logging.getLogger(__name__)


class WebImageLoader(BaseWebLoader):
    """Service for loading images from Setics authenticated websites."""

    def __init__(self):
        """Initialize the Setics image loader service."""
        super().__init__()
        self._auth_service = WebAuthentication()
        self._authenticated = False
        self.login_url = None
        self.check_url = None
        self.username = None
        self.password = None
        self._image_processor = WebImageProcessor()

    async def initialize(
        self, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
    ) -> None:
        """
        Initialize the Setics image loader service.

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
        logger.debug(f"Initialized Setics image loader service with {timeout}s timeout")

    async def authenticate(
        self,
        username: str,
        password: str,
        login_url: str,
        check_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> "WebImageLoader":
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

    async def extract_image_urls(
        self, urls: Union[str, List[str]], continue_on_failure: bool = False
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract image URLs from the provided web pages, filtered for Setics content.

        Args:
            urls: A single URL or list of URLs to extract images from
            continue_on_failure: Whether to continue if extraction fails for some URLs

        Returns:
            Dictionary mapping page URLs to lists of image info dictionaries
        """
        if not self._initialized:
            raise ValueError("Service must be initialized before extracting images")

        if not self._authenticated:
            raise ValueError("Authentication required before extracting images")

        if isinstance(urls, str):
            urls = [urls]

        result = {}

        for url in urls:
            try:
                # Use the Setics-specific image extraction method
                image_info = (
                    await self._image_processor.extract_setics_image_urls_from_url(
                        url=url, http_client=self._http_client
                    )
                )

                # Only add to result if images were found
                if image_info:
                    result[url] = image_info
                    logger.info(f"Found {len(image_info)} relevant images at {url}")
                else:
                    logger.info(f"No relevant images found at {url}")

            except Exception as e:
                logger.error(f"Error extracting image URLs from {url}: {str(e)}")
                if not continue_on_failure:
                    raise

        return result

    @property
    def is_authenticated(self) -> bool:
        """Check if the service is authenticated."""
        return self._authenticated

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
async def create_setics_image_loader() -> WebImageLoader:
    """Create and initialize a Setics image loader service."""
    service = WebImageLoader()
    await service.initialize()
    return service
