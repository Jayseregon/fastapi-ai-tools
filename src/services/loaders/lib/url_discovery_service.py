import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class UrlDiscoveryService:
    """Service for discovering URLs by crawling websites."""

    def __init__(self):
        """Initialize the URL discovery service without any specific configuration."""
        self.base_url = None
        self.session = None
        self.headers = None
        self.max_depth = None
        self.same_domain_only = None
        self.visited_urls = set()
        self.discovered_urls = set()
        self._initialized = False
        self._lock = threading.RLock()

    def initialize(
        self,
        base_url: str,
        session: Optional[requests.Session] = None,
        headers: Optional[dict[str, str]] = None,
        max_depth: int = 2,
        same_domain_only: bool = True,
    ) -> None:
        """
        Initialize the URL discovery service with specific configuration.

        Args:
            base_url: The starting URL for discovery
            session: Optional requests.Session for making HTTP requests
            headers: Optional HTTP headers to include in requests
            max_depth: Maximum depth to crawl (default: 2)
            same_domain_only: Whether to only crawl URLs on the same domain (default: True)
        """
        self.base_url = base_url
        self.session = session or requests.Session()
        self.headers = headers or {}
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.visited_urls = set()
        self.discovered_urls = set()
        self._initialized = True
        logger.debug(f"URL Discovery Service initialized with base URL: {base_url}")

    def __call__(
        self,
        base_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
        headers: Optional[dict[str, str]] = None,
        max_depth: Optional[int] = None,
        same_domain_only: Optional[bool] = None,
    ) -> List[str]:
        """
        Discover URLs by crawling from the base URL.

        Can optionally override initialization parameters.

        Returns:
            List of discovered URLs
        """
        # Handle new parameters if provided
        if base_url or not self._initialized:
            self.initialize(
                base_url=base_url or self.base_url,
                session=session or self.session,
                headers=headers or self.headers,
                max_depth=max_depth if max_depth is not None else (self.max_depth or 2),
                same_domain_only=(
                    same_domain_only
                    if same_domain_only is not None
                    else (self.same_domain_only or True)
                ),
            )

        if not self._initialized:
            logger.error("URL Discovery Service not initialized")
            raise ValueError("Service must be initialized before discovery")

        return self._discover_urls()

    def _discover_urls(self) -> List[str]:
        # Lock to prevent concurrent access to shared state
        with self._lock:
            base_domain = urlparse(self.base_url).netloc
            # Use deque for better performance in pop/append operations
            to_crawl = deque([(self.base_url, 0)])
            self.discovered_urls.clear()
            self.visited_urls.clear()

            while to_crawl:
                url, depth = to_crawl.popleft()

                # Skip if already visited or beyond max depth
                if url in self.visited_urls:
                    continue

                self.visited_urls.add(url)
                logger.debug(f"Fetching: {url} (depth {depth})")

                try:
                    response = self.session.get(url, headers=self.headers, timeout=10)
                    response.raise_for_status()  # Raise exception for 4XX/5XX responses

                    # Add to discovered URLs
                    self.discovered_urls.add(url)

                    # Don't crawl deeper if at max depth
                    if depth >= self.max_depth:
                        continue

                    # Find all links
                    soup = BeautifulSoup(response.text, "html.parser")
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        full_url = urljoin(url, href)

                        # Skip non-HTTP URLs, fragments, etc.
                        if not self._is_valid_url(full_url, base_domain):
                            continue

                        # Skip URLs already visited or queued
                        if full_url in self.visited_urls or any(
                            full_url == u for u, _ in to_crawl
                        ):
                            continue

                        # Add to crawl queue
                        to_crawl.append((full_url, depth + 1))

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error fetching {url}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error processing {url}: {str(e)}")

        return list(self.discovered_urls)

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """
        Check if a URL is valid for crawling based on current settings.

        Args:
            url: The URL to check
            base_domain: The domain of the base URL

        Returns:
            Boolean indicating if the URL should be crawled
        """
        # Must be HTTP or HTTPS
        if not url.startswith(("http://", "https://")):
            return False

        # Domain filtering
        if self.same_domain_only and urlparse(url).netloc != base_domain:
            return False

        return True

    def to_json(self, filename: str | Path) -> None:
        with self._lock:
            if isinstance(filename, str):
                filename = Path(filename)

            filename.parent.mkdir(parents=True, exist_ok=True)

            if self.discovered_urls is None:
                self._discover_urls()

            try:
                with filename.open("w") as f:
                    json.dump(list(self.discovered_urls), f, indent=2)
                logger.info(f"URLs saved to: {filename}")
            except (PermissionError, IOError) as e:
                logger.error(f"Error saving URLs to {filename}: {str(e)}")
                raise

    def reset(self):
        """Reset the service state, clearing all URLs and settings."""
        self.base_url = None
        self.session = None
        self.headers = None
        self.max_depth = None
        self.same_domain_only = None
        self.visited_urls.clear()
        self.discovered_urls.clear()
        self._initialized = False


# Create a singleton instance
url_discovery_service = UrlDiscoveryService()
