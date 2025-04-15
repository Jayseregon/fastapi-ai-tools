import logging
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class CookieManager:
    """Manages cookie extraction and manipulation for web requests."""

    async def extract_domain_cookies(
        self, http_client, urls: Union[str, List[str]]
    ) -> Dict[str, str]:
        """
        Extract cookies with domain awareness for better targeting.

        Args:
            http_client: HTTP client with cookies
            urls: URL or URLs to consider for cookie domain matching

        Returns:
            Dictionary of cookies, prioritizing those matching target domains
        """
        # Convert single URL to list
        if isinstance(urls, str):
            urls = [urls]

        if not urls:
            return {}

        if not http_client.client or not hasattr(http_client.client, "cookies"):
            return {}

        # Extract the first URL's domain
        target_domain = urlparse(urls[0]).netloc if urls else None

        # Handle different cookie storage formats
        try:
            if hasattr(http_client.client.cookies, "jar"):
                # Standard cookiejar format
                return self._extract_from_cookiejar(
                    http_client.client.cookies.jar, target_domain
                )
            elif hasattr(http_client.client.cookies, "items"):
                # Dict-like format
                return dict(http_client.client.cookies.items())
            else:
                # Simple format (try direct conversion)
                return dict(http_client.client.cookies)
        except Exception as e:
            logger.warning(f"Could not extract cookies from HTTP client: {str(e)}")
            return {}

    def _extract_from_cookiejar(
        self, cookie_jar, target_domain: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Extract cookies from a cookie jar with domain preferences.

        Args:
            cookie_jar: Cookie jar to extract from
            target_domain: Target domain to prioritize

        Returns:
            Dictionary of cookies
        """
        cookie_dict: Dict[str, str] = {}

        # Get all cookies from jar
        all_cookies = cookie_jar._cookies if hasattr(cookie_jar, "_cookies") else {}

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
