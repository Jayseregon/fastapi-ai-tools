import logging
from typing import Dict, Optional

import requests

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

    def initialize(self, headers: Optional[Dict[str, str]] = None) -> None:
        """Override base class initialize to match its signature."""
        super().initialize(headers=headers)

    def authenticate(
        self,
        username: str,
        password: str,
        login_url: str,
        check_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Authenticate with Setics website and return configured service."""
        # Initialize with proper headers
        setics_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://support.setics-sttar.com",
            "Referer": login_url,
        }
        if headers:
            setics_headers.update(headers)

        # Initialize first
        self.initialize(headers=setics_headers)

        # Store Setics-specific attributes
        self.username = username
        self.password = password
        self.login_url = login_url
        self.check_url = check_url

        # Get authenticity token and perform login
        self._get_authenticity_token()
        return self.__call__()

    def __call__(self):
        if not self._initialized or not self.authenticity_token:
            raise ValueError("Service must be initialized before calling")

        try:
            payload = {
                "authenticity_token": self.authenticity_token,
                "user[email]": self.username,
                "user[password]": self.password,
            }

            # Perform login
            login_response = self.session.post(
                self.login_url, data=payload, headers=self.headers, allow_redirects=True
            )
            login_response.raise_for_status()

            logger.info(f"Login attempt status: {login_response.status_code}")

            # Check if we can access a protected URL
            if self.check_url:
                check_response = self.session.get(self.check_url, headers=self.headers)
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

    def _get_authenticity_token(self) -> None:
        try:
            login_page = self.session.get(self.login_url, headers=self.headers)
            login_page.raise_for_status()

            self.authenticity_token = self._extract_token(login_page.content)
            if not self.authenticity_token:
                raise ValueError("Could not find authenticity token on login page")

            logger.debug("Retrieved authenticity token for Setics login")
        except Exception as e:
            logger.error(f"Failed to get authenticity token: {str(e)}")
            raise

    @property
    def authenticated_session(self) -> requests.Session:
        """Returns the authenticated session."""
        if not self._initialized or not self.last_login_status:
            raise ValueError("Authentication required before accessing session")
        return self.session

    @property
    def request_headers(self) -> Dict[str, str]:
        """Returns headers needed for requests."""
        if not self._initialized:
            raise ValueError("Service must be initialized before accessing headers")
        return self.headers.copy()  # Return copy to prevent modification


# Create a singleton instance for Setics
setics_web_loader_service = SeticsWebLoaderService()
