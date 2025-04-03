import logging
from typing import ClassVar, Dict, List, Optional, Union
from urllib.parse import urlparse

from src.services.loaders.lib import WebAuthentication
from src.services.loaders.lib.web_image_processor import WebImageProcessor
from src.services.loaders.web.base_web_loader import BaseWebLoader

logger = logging.getLogger(__name__)


class WebImageLoader(BaseWebLoader):
    """Service for loading images from both public and protected websites."""

    # Class constants for loader modes
    MODE_PUBLIC: ClassVar[str] = "public"
    MODE_PROTECTED: ClassVar[str] = "protected"

    def __init__(self, mode: str = MODE_PUBLIC):
        """
        Initialize the Web Image loader service.

        Args:
            mode: The mode of operation (public or protected)
        """
        super().__init__()
        self._auth_service = WebAuthentication()
        self._authenticated = False
        self.login_url: Optional[str] = None
        self.check_url: Optional[str] = None
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self._image_processor = WebImageProcessor()
        self._mode = mode

    @classmethod
    async def create_public_loader(
        cls, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
    ) -> "WebImageLoader":
        """
        Create a loader for public websites (no authentication).

        Args:
            headers: Optional HTTP headers to include in requests
            timeout: Timeout in seconds for HTTP requests

        Returns:
            Initialized WebImageLoader instance for public websites
        """
        loader = cls(mode=cls.MODE_PUBLIC)
        await loader.initialize(headers=headers, timeout=timeout)
        return loader

    @classmethod
    async def create_protected_loader(
        cls,
        username: str,
        password: str,
        login_url: str,
        check_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> "WebImageLoader":
        """
        Create a loader for protected websites with authentication.

        Args:
            username: Username/email for authentication
            password: Password for authentication
            login_url: URL of the login page
            check_url: Optional URL to verify successful authentication
            headers: Optional HTTP headers to include in requests
            timeout: Timeout in seconds for HTTP requests

        Returns:
            Authenticated WebImageLoader instance for protected websites
        """
        loader = cls(mode=cls.MODE_PROTECTED)
        await loader.initialize(headers=headers, timeout=timeout)
        await loader.authenticate(
            username=username,
            password=password,
            login_url=login_url,
            check_url=check_url,
            headers=headers,
        )
        return loader

    async def initialize(
        self, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
    ) -> None:
        """
        Initialize the web image loader service.

        Args:
            headers: Optional HTTP headers to include in requests
            timeout: Timeout in seconds for HTTP requests
        """
        # Define default headers for web scraping
        default_headers = {
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }

        # Combine with provided headers
        if headers:
            default_headers.update(headers)

        # Initialize HTTP client
        await self._http_client.initialize(headers=default_headers, timeout=timeout)
        self._initialized = True
        logger.debug(f"Initialized web image loader service with {timeout}s timeout")

    async def authenticate(
        self,
        username: str,
        password: str,
        login_url: str,
        check_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> "WebImageLoader":
        """
        Authenticate with a protected website.

        Args:
            username: Username/email
            password: Password
            login_url: URL of the login page
            check_url: Optional URL to verify successful authentication
            headers: Optional additional headers for authentication requests

        Returns:
            Self for method chaining
        """
        if not self._initialized:
            await self.initialize(headers=headers)

        # Set mode to protected
        self._mode = self.MODE_PROTECTED

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

        # Detect the website type and adjust authentication parameters
        auth_params = self._detect_auth_params(login_url)

        # Create credentials dict in expected format - fix the typing issue
        credentials: Dict[str, str] = {
            # Ensure username_field and password_field are strings
            str(auth_params["username_field"]): username,
            str(auth_params["password_field"]): password,
        }

        # Perform complete authentication flow
        self._authenticated = await self._auth_service.complete_authentication_flow(
            http_client=self._http_client,
            login_url=login_url,
            credentials=credentials,
            check_url=check_url,
            token_field=str(auth_params["token_field"]),
            failure_strings=(
                auth_params["failure_strings"]
                if isinstance(auth_params["failure_strings"], list)
                else [auth_params["failure_strings"]]
            ),
        )

        if not self._authenticated:
            raise ValueError(f"Failed to authenticate with {login_url}")

        logger.debug(f"Successfully authenticated with {login_url}")
        return self

    def _detect_auth_params(self, login_url: str) -> Dict[str, Union[str, List[str]]]:
        """
        Detect authentication parameters based on the login URL.

        Args:
            login_url: URL of the login page

        Returns:
            Dictionary containing authentication parameters
        """
        # Default parameters
        params: Dict[str, Union[str, List[str]]] = {
            "username_field": "username",
            "password_field": "password",
            "token_field": "csrf_token",
            "failure_strings": ["Invalid credentials"],
        }

        # Detect Setics website
        if "setics" in login_url.lower():
            params = {
                "username_field": "user[email]",
                "password_field": "user[password]",
                "token_field": "authenticity_token",
                "failure_strings": ["Invalid Email or password"],
            }

        # Add more website detectors here as needed

        return params

    async def extract_image_urls(
        self, urls: Union[str, List[str]], continue_on_failure: bool = False
    ) -> List[Dict[str, str]]:
        """
        Extract image URLs from the provided web pages.

        Args:
            urls: A single URL or list of URLs to extract images from
            continue_on_failure: Whether to continue if extraction fails for some URLs

        Returns:
            List of image info dictionaries from all URLs
        """
        if not self._initialized:
            raise ValueError("Service must be initialized before extracting images")

        if self._mode == self.MODE_PROTECTED and not self._authenticated:
            raise ValueError(
                "Authentication required before extracting images in protected mode"
            )

        if isinstance(urls, str):
            urls = [urls]

        result = []

        for url in urls:
            try:
                # Extract image URLs from the page
                image_info = (
                    await self._image_processor.extract_setics_image_urls_from_url(
                        url=url, http_client=self._http_client
                    )
                )

                # Only add to result if images were found
                if image_info:
                    result.extend(image_info)
                    logger.debug(f"Found {len(image_info)} relevant images at {url}")
                else:
                    logger.debug(f"No relevant images found at {url}")

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

    @property
    def mode(self) -> str:
        """Get the current operation mode."""
        return self._mode

    async def close(self) -> None:
        """Clean up resources asynchronously."""
        await super().close()
        self._authenticated = False


# Factory function for global access
async def create_web_image_loader(
    protected: bool = False,
    username: Optional[str] = None,
    password: Optional[str] = None,
    login_url: Optional[str] = None,
    check_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> WebImageLoader:
    """
    Create and initialize a web image loader service.

    Args:
        protected: Whether the loader is for protected sites requiring auth
        username: Username for authentication (required if protected=True)
        password: Password for authentication (required if protected=True)
        login_url: Login URL (required if protected=True)
        check_url: URL to check successful authentication
        headers: Optional HTTP headers
        timeout: Request timeout in seconds

    Returns:
        Initialized WebImageLoader instance
    """
    if protected:
        if not all([username, password, login_url]):
            raise ValueError(
                "Username, password, and login_url are required for protected mode"
            )

        # Use assert statements to help mypy understand these are not None
        assert username is not None  # For type checking
        assert password is not None  # For type checking
        assert login_url is not None  # For type checking

        return await WebImageLoader.create_protected_loader(
            username=username,
            password=password,
            login_url=login_url,
            check_url=check_url,
            headers=headers,
            timeout=timeout,
        )
    else:
        return await WebImageLoader.create_public_loader(
            headers=headers, timeout=timeout
        )
