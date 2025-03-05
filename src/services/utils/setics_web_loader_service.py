import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document

from src.services.utils.auth_web_loader_base import AuthWebLoaderBase

logger = logging.getLogger(__name__)


class SeticsWebLoaderService(AuthWebLoaderBase):
    """Service for loading content from Setics authenticated websites."""

    def __init__(self):
        """Initialize the Setics web loader service."""
        super().__init__()
        self.login_url = None
        self.check_url = None
        self.username = None
        self.password = None
        self.authenticity_token = None

    async def initialize(
        self,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = 30.0,
    ) -> None:
        """Override base class initialize to match its signature."""
        await super().initialize(headers=headers, timeout=timeout)

    async def authenticate(
        self,
        username: str,
        password: str,
        login_url: str,
        check_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Authenticate with Setics website and return configured service."""
        setics_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://support.setics-sttar.com",
            "Referer": login_url,
        }
        if headers:
            setics_headers.update(headers)

        if not self._initialized:
            if self.client is None:
                self.client = httpx.AsyncClient(follow_redirects=True)

            if setics_headers:
                self.headers.update(setics_headers)

            self._initialized = True

        self.username = username
        self.password = password
        self.login_url = login_url
        self.check_url = check_url

        # Get authenticity token
        await self._get_authenticity_token()

        return await self._perform_authentication()

    async def _perform_authentication(self):
        """Internal method to perform authentication without acquiring the lock again."""
        if not self._initialized or not self.authenticity_token:
            raise ValueError("Service must be initialized before authentication")

        try:
            payload = {
                "authenticity_token": self.authenticity_token,
                "user[email]": self.username,
                "user[password]": self.password,
            }

            # Perform login
            login_response = await self.client.post(
                self.login_url,
                data=payload,
                headers=self.headers,
                follow_redirects=True,
            )
            login_response.raise_for_status()

            logger.info(f"Login attempt status: {login_response.status_code}")

            # Check if we can access a protected URL
            if self.check_url:
                check_response = await self.client.get(
                    self.check_url, headers=self.headers
                )
                self.last_login_status = check_response.status_code == 200
                if not self.last_login_status:
                    raise Exception(
                        f"Failed to access check URL: {check_response.status_code}"
                    )
            else:
                # If no check URL, assume success based on status code and cookies
                self.last_login_status = 200 <= login_response.status_code < 300

            if not self.last_login_status:
                raise Exception("Login failed: Unable to authenticate")

            return self

        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            self.last_login_status = False
            raise

    async def __call__(self):
        if not self.authenticity_token:
            raise ValueError("Authentication required before calling")
        return await self._perform_authentication()

    async def _get_authenticity_token(self) -> None:
        try:
            browser_headers = {
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
            }
            request_headers = self.headers.copy()
            request_headers.update(browser_headers)
            print("Headers: ", request_headers)

            # Extract domain from login URL
            parsed_url = urlparse(self.login_url)
            base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            print("Base domain: ", base_domain)

            # First visit homepage
            home_page = await self.client.get(
                base_domain, headers=self.headers, follow_redirects=True
            )
            print("Home page: ", home_page)
            home_page.raise_for_status()

            # Then visit login page
            login_page = await self.client.get(
                self.login_url, headers=self.headers, follow_redirects=True
            )
            print("Login page: ", login_page)
            login_page.raise_for_status()

            self.authenticity_token = self._extract_token(login_page.text)
            if not self.authenticity_token:
                raise ValueError("Could not find authenticity token on login page")

            logger.debug("Retrieved authenticity token for Setics login")
        except Exception as e:
            logger.error(f"Failed to get authenticity token: {str(e)}")
            raise

    @property
    def authenticated_client(self) -> httpx.AsyncClient:
        """Returns the authenticated client."""
        if not self._initialized or not self.last_login_status:
            raise ValueError("Authentication required before accessing client")
        return self.client

    @property
    def request_headers(self) -> Dict[str, str]:
        """Returns headers needed for requests."""
        if not self._initialized:
            raise ValueError("Service must be initialized before accessing headers")
        return self.headers.copy()  # Return copy to prevent modification

    async def load_documents_from_urls(self, urls: str | List[str]) -> List[Document]:
        """Load documents from Setics URLs."""
        if not self._initialized:
            raise ValueError("Service must be authenticated before loading documents")

        return await self.load_documents(urls)


# Create a singleton instance for Setics
setics_web_loader_service = SeticsWebLoaderService()
