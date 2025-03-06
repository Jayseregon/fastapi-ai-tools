import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.services.utils.httpClientService import HttpClientService

logger = logging.getLogger(__name__)


class WebAuthenticationService:
    """Handles web authentication flows and token/cookie management."""

    def __init__(self):
        """Initialize the web authentication service."""
        self._last_authentication_status = False

    def extract_token(
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

    async def extract_domain_cookies(
        self, http_client: HttpClientService, urls: List[str]
    ) -> Dict[str, str]:
        """
        Extract cookies with domain-awareness.

        Args:
            http_client: HTTP client with cookies
            urls: URLs to consider for cookie domain matching

        Returns:
            Dictionary of cookies, prioritizing those matching target domains
        """
        if not http_client.client or not hasattr(http_client.client, "cookies"):
            return {}

        # Extract the first URL's domain to get proper cookies
        target_domain = urlparse(urls[0]).netloc if urls else None
        cookie_dict: Dict[str, str] = {}

        # Get all cookies from jar
        all_cookies = (
            http_client.client.cookies.jar._cookies
            if hasattr(http_client.client.cookies, "jar")
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

    async def get_authenticity_token(
        self,
        http_client: HttpClientService,
        login_url: str,
        token_field: str = "authenticity_token",
        browser_headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Retrieve authenticity token from a login page.

        Args:
            http_client: HTTP client for requests
            login_url: URL of the login page
            token_field: Name of the token field
            browser_headers: Optional additional headers to simulate browser behavior

        Returns:
            Authentication token string

        Raises:
            ValueError: If token cannot be found
        """
        try:
            # Add browser-like headers if provided
            request_headers = None
            if browser_headers:
                request_headers = browser_headers

            # Extract domain from login URL
            parsed_url = urlparse(login_url)
            base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # First visit homepage to capture cookies
            await http_client.get(base_domain, headers=request_headers)

            # Then visit login page
            login_page = await http_client.get(login_url, headers=request_headers)

            # Extract token from login page
            token = self.extract_token(login_page.text, token_field=token_field)
            if not token:
                raise ValueError(f"Could not find {token_field} on login page")

            logger.debug(f"Retrieved authentication token for {login_url}")
            return token

        except Exception as e:
            logger.error(f"Failed to get authentication token: {str(e)}")
            raise

    async def perform_form_authentication(
        self,
        http_client: HttpClientService,
        login_url: str,
        credentials: Dict[str, str],
        token_field: str = "authenticity_token",
        token_value: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, httpx.Response]:
        """
        Handle form-based authentication flow.

        Args:
            http_client: HTTP client for requests
            login_url: URL of the login form
            credentials: Dictionary with login credentials (e.g., username/password)
            token_field: Name of the token field
            token_value: Optional token value (if already obtained)
            extra_params: Optional additional parameters for the login form

        Returns:
            Tuple of (success status, response object)

        Raises:
            Exception: If authentication fails
        """
        try:
            # Get token if not provided
            auth_token = token_value
            if not auth_token:
                auth_token = await self.get_authenticity_token(
                    http_client, login_url, token_field
                )

            # Build payload
            payload = {token_field: auth_token}

            # Add credentials with proper naming
            for key, value in credentials.items():
                payload[key] = value

            # Add any extra parameters
            if extra_params:
                payload.update(extra_params)

            # Add proper content-type header for forms
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            # Get the login domain as referer
            parsed_url = urlparse(login_url)
            headers["Origin"] = f"{parsed_url.scheme}://{parsed_url.netloc}"
            headers["Referer"] = login_url

            # Perform login
            login_response = await http_client.post(
                login_url, data=payload, headers=headers
            )

            logger.info(f"Login attempt status: {login_response.status_code}")

            # Check if login was successful based on status code
            success = 200 <= login_response.status_code < 300
            self._last_authentication_status = success

            return success, login_response

        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            self._last_authentication_status = False
            raise

    async def verify_authentication(
        self,
        http_client: HttpClientService,
        check_url: str,
        failure_strings: Optional[List[str]] = None,
    ) -> bool:
        """
        Verify if authentication was successful using a protected URL.

        Args:
            http_client: HTTP client for requests
            check_url: URL that should only be accessible when authenticated
            failure_strings: Optional list of strings that indicate authentication failure

        Returns:
            True if authentication is valid, False otherwise
        """
        try:
            check_response = await http_client.get(check_url)

            # Check for failure indicators in content if provided
            if failure_strings and check_response.text:
                for failure_string in failure_strings:
                    if failure_string in check_response.text:
                        logger.warning(
                            f"Authentication check failed: found '{failure_string}' in response"
                        )
                        self._last_authentication_status = False
                        return False

            # Check successful based on status code
            success = check_response.status_code == 200
            self._last_authentication_status = success

            if not success:
                logger.warning(
                    f"Authentication check failed: got status {check_response.status_code}"
                )

            return success

        except Exception as e:
            logger.error(f"Authentication verification failed: {str(e)}")
            self._last_authentication_status = False
            return False

    @property
    def is_authenticated(self) -> bool:
        """Get the current authentication status."""
        return self._last_authentication_status

    async def complete_authentication_flow(
        self,
        http_client: HttpClientService,
        login_url: str,
        credentials: Dict[str, str],
        check_url: Optional[str] = None,
        token_field: str = "authenticity_token",
        extra_params: Optional[Dict[str, Any]] = None,
        failure_strings: Optional[List[str]] = None,
    ) -> bool:
        """
        Complete the full authentication flow: token retrieval, login, and verification.

        Args:
            http_client: HTTP client for requests
            login_url: URL of the login form
            credentials: Dictionary with login credentials
            check_url: Optional URL to verify successful authentication
            token_field: Name of the token field
            extra_params: Optional additional parameters for the login form
            failure_strings: Optional list of strings that indicate authentication failure

        Returns:
            True if authentication is successful, False otherwise
        """
        try:
            # Get authenticity token
            token = await self.get_authenticity_token(
                http_client, login_url, token_field
            )

            # Perform login
            success, _ = await self.perform_form_authentication(
                http_client=http_client,
                login_url=login_url,
                credentials=credentials,
                token_field=token_field,
                token_value=token,
                extra_params=extra_params,
            )

            if not success:
                return False

            # Verify if check_url is provided
            if check_url:
                return await self.verify_authentication(
                    http_client=http_client,
                    check_url=check_url,
                    failure_strings=failure_strings,
                )

            return success

        except Exception as e:
            logger.error(f"Authentication flow failed: {str(e)}")
            self._last_authentication_status = False
            return False
