import logging
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from src.services.utils.httpClientService import HttpClientService

logger = logging.getLogger(__name__)


class SessionAdapter:
    """Adapter to make HttpClientService compatible with WebBaseLoader."""

    def __init__(self, client, cookies, headers, timeout=30.0):
        """Initialize SessionAdapter with explicit timeout."""
        self.client = client
        self.cookies = type("Cookies", (), {"get_dict": lambda self: cookies})()
        self.headers = headers
        self.verify = True  # SSL verification
        self.timeout = timeout


class DocumentLoaderService:
    """Handles document loading from HTML content and conversion to LangChain documents."""

    def __init__(self, default_parser: str = "html.parser"):
        """
        Initialize the document loader service.

        Args:
            default_parser: Default parser to use with BeautifulSoup
        """
        self.default_parser = default_parser

    def _create_session_adapter(
        self, http_client: HttpClientService, cookies: Dict[str, str]
    ) -> SessionAdapter:
        """
        Create a session adapter for WebBaseLoader compatibility.

        Args:
            http_client: HTTP client to adapt
            cookies: Cookies to include in session

        Returns:
            Session adapter compatible with WebBaseLoader
        """
        return SessionAdapter(
            client=http_client.client,
            cookies=cookies,
            headers=http_client.headers,
            timeout=http_client.timeout_duration,
        )

    def _extract_text_from_html(
        self,
        html_content: str,
        css_selector: Optional[str] = None,
        parser: Optional[str] = None,
    ) -> str:
        """
        Parse HTML and extract text content.

        Args:
            html_content: Raw HTML content
            css_selector: Optional CSS selector to extract specific content
            parser: Parser to use with BeautifulSoup

        Returns:
            Extracted text content
        """
        soup = BeautifulSoup(html_content, parser or self.default_parser)

        # Extract from specific element if selector provided
        if css_selector:
            content_element = soup.select_one(css_selector)
            if content_element:
                return content_element.get_text(separator="\n", strip=True)

        # Remove script and style elements that might contain code
        for script in soup(["script", "style", "meta", "noscript"]):
            script.extract()

        # Get the page content
        return soup.get_text(separator="\n", strip=True)

    def _extract_metadata_from_html(
        self, html_content: str, url: str, parser: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from HTML content.

        Args:
            html_content: Raw HTML content
            url: Source URL
            parser: Parser to use with BeautifulSoup

        Returns:
            Metadata dictionary
        """
        soup = BeautifulSoup(html_content, parser or self.default_parser)
        metadata = {"source": url}

        # Extract title
        if soup.title:
            metadata["title"] = soup.title.string

        # Extract description
        description_meta = soup.find("meta", attrs={"name": "description"})
        if description_meta and description_meta.get("content"):
            metadata["description"] = description_meta.get("content")

        # Extract language
        language_meta = soup.find("meta", attrs={"name": "language"}) or soup.find(
            "html", attrs={"lang": True}
        )
        if language_meta:
            metadata["language"] = language_meta.get("content") or language_meta.get(
                "lang"
            )

        # Extract canonical URL
        canonical_link = soup.find("link", attrs={"rel": "canonical"})
        if canonical_link and canonical_link.get("href"):
            metadata["canonical_url"] = canonical_link.get("href")

        # Add domain
        try:
            parsed_url = urlparse(url)
            metadata["domain"] = parsed_url.netloc
        except Exception:
            pass

        return metadata

    def _create_document(self, content: str, metadata: Dict[str, Any]) -> Document:
        """
        Create a LangChain document with content and metadata.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            LangChain Document object
        """
        return Document(page_content=content, metadata=metadata)

    async def load_document_from_url(
        self,
        http_client: HttpClientService,
        url: str,
        css_selector: Optional[str] = None,
    ) -> Document:
        """
        Load a document from a URL.

        Args:
            http_client: HTTP client for making requests
            url: URL to load
            css_selector: Optional CSS selector to extract specific content

        Returns:
            Loaded Document

        Raises:
            ValueError: If URL is invalid or content cannot be loaded
        """
        try:
            logger.debug(f"Loading document from {url}")
            response = await http_client.get(url)

            # Extract text and metadata
            html_content = response.text
            content = self._extract_text_from_html(html_content, css_selector)
            metadata = self._extract_metadata_from_html(html_content, url)

            # Add HTTP status to metadata
            metadata["http_status"] = response.status_code

            return self._create_document(content, metadata)
        except Exception as e:
            logger.error(f"Error loading document from {url}: {str(e)}")
            raise ValueError(f"Failed to load document from {url}: {str(e)}")

    async def load_documents_from_urls(
        self,
        http_client: HttpClientService,
        urls: str | List[str],
        css_selector: Optional[str] = None,
        continue_on_failure: bool = True,
    ) -> List[Document]:
        """
        Load multiple documents from URLs.

        Args:
            http_client: HTTP client for making requests
            urls: Single URL or list of URLs to load
            css_selector: Optional CSS selector to extract specific content
            continue_on_failure: Whether to continue if loading fails for some URLs

        Returns:
            List of loaded Documents
        """
        if isinstance(urls, str):
            urls = [urls]

        documents = []
        for url in urls:
            try:
                doc = await self.load_document_from_url(http_client, url, css_selector)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading document from {url}: {str(e)}")
                if not continue_on_failure:
                    raise

        logger.info(f"Loaded {len(documents)} documents from {len(urls)} URLs")
        return documents

    async def lazy_load_documents(
        self,
        http_client: HttpClientService,
        urls: str | List[str],
        css_selector: Optional[str] = None,
        continue_on_failure: bool = True,
    ) -> AsyncIterator[Document]:
        """
        Lazily load documents from URLs as an async generator.

        Args:
            http_client: HTTP client for making requests
            urls: Single URL or list of URLs to load
            css_selector: Optional CSS selector to extract specific content
            continue_on_failure: Whether to continue if loading fails for some URLs

        Yields:
            Documents as they are loaded
        """
        if isinstance(urls, str):
            urls = [urls]

        for url in urls:
            try:
                doc = await self.load_document_from_url(http_client, url, css_selector)
                yield doc
            except Exception as e:
                logger.error(f"Error loading document from {url}: {str(e)}")
                if not continue_on_failure:
                    raise

    async def extract_domain_aware_cookies(
        self, http_client: HttpClientService, urls: List[str]
    ) -> Dict[str, str]:
        """
        Extract cookies with domain awareness for better targeting.

        Args:
            http_client: HTTP client with cookies
            urls: URLs to consider for cookie domain matching

        Returns:
            Dictionary of cookies, prioritizing those matching target domains
        """
        cookie_dict: Dict[str, str] = {}
        if not http_client.client or not hasattr(http_client.client, "cookies"):
            return cookie_dict

        # Extract the first URL's domain to get proper cookies
        target_domain = urlparse(urls[0]).netloc if urls else None

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

    async def create_langchain_loader(
        self,
        http_client: HttpClientService,
        urls: str | List[str],
        continue_on_failure: bool = True,
    ) -> WebBaseLoader:
        """
        Create a LangChain WebBaseLoader configured with the HTTP client.

        Args:
            http_client: HTTP client for requests
            urls: URLs to load
            continue_on_failure: Whether to continue on errors

        Returns:
            Configured WebBaseLoader instance
        """
        if isinstance(urls, str):
            urls = [urls]

        try:
            # Extract cookies
            cookies = await self.extract_domain_aware_cookies(http_client, urls)
        except Exception as e:
            logger.warning(f"Cookie extraction failed: {str(e)}, using empty cookies")
            cookies = {}

        # Create session adapter
        adapter = self._create_session_adapter(http_client, cookies)

        # Create loader
        loader = WebBaseLoader(
            web_paths=urls,
            session=adapter,
            requests_kwargs={"headers": http_client.headers},
            continue_on_failure=continue_on_failure,
            # bs_kwargs={"features": self.default_parser},
        )

        logger.debug(f"Created WebBaseLoader for {len(urls)} URLs")
        return loader

    async def load_documents_with_langchain(
        self,
        http_client: HttpClientService,
        urls: str | List[str],
        continue_on_failure: bool = True,
    ) -> List[Document]:
        """
        Load documents using LangChain's WebBaseLoader.

        Args:
            http_client: HTTP client for requests
            urls: URLs to load
            continue_on_failure: Whether to continue on errors

        Returns:
            List of loaded documents
        """
        loader = await self.create_langchain_loader(
            http_client=http_client, urls=urls, continue_on_failure=continue_on_failure
        )

        documents = []
        async for doc in loader.alazy_load():
            documents.append(doc)

        return documents

    async def lazy_load_documents_with_langchain(
        self,
        http_client: HttpClientService,
        urls: str | List[str],
        continue_on_failure: bool = True,
    ) -> AsyncIterator[Document]:
        """
        Lazily load documents using LangChain's WebBaseLoader.

        Args:
            http_client: HTTP client for requests
            urls: URLs to load
            continue_on_failure: Whether to continue on errors

        Yields:
            Documents as they are loaded
        """
        loader = await self.create_langchain_loader(
            http_client=http_client, urls=urls, continue_on_failure=continue_on_failure
        )

        async for doc in loader.alazy_load():
            yield doc

    # [Your existing methods for load_document_from_url, load_documents_from_urls, and lazy_load_documents]
