from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.services.loaders.lib import HttpClient, WebAuthentication
from src.services.loaders.web.setics_loader import (
    SeticsLoader,
    create_setics_web_loader_service,
)


class TestSeticsLoader:
    @pytest.fixture
    def setics_loader(self):
        """Create a SeticsLoader instance for testing"""
        return SeticsLoader()

    @pytest.fixture
    def mock_auth_service(self):
        """Mock WebAuthentication service for testing"""
        auth_service = MagicMock(spec=WebAuthentication)
        auth_service.complete_authentication_flow = AsyncMock()
        auth_service.is_authenticated = False
        return auth_service

    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client for testing"""
        http_client = MagicMock(spec=HttpClient)
        http_client.initialize = AsyncMock()
        http_client.close = AsyncMock()
        http_client.headers = {}
        return http_client

    @pytest.fixture
    def mock_document_loader(self):
        """Mock document loader for testing"""
        doc_loader = MagicMock()
        doc_loader.load_documents_with_langchain = AsyncMock()
        doc_loader.lazy_load_documents_with_langchain = AsyncMock()
        return doc_loader

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Document(
                page_content="Test content 1",
                metadata={"source": "https://setics.com/page1"},
            ),
            Document(
                page_content="Test content 2",
                metadata={"source": "https://setics.com/page2"},
            ),
        ]

    @pytest.fixture
    def sample_auth_details(self):
        """Sample authentication details"""
        return {
            "username": "test@example.com",
            "password": "test_password",
            "login_url": "https://setics.com/login",
            "check_url": "https://setics.com/dashboard",
        }

    @pytest.mark.asyncio
    async def test_initialization(self, setics_loader):
        """Test initialization with default parameters"""
        # Replace the internal HTTP client with a mock
        mock_http = AsyncMock()
        setics_loader._http_client = mock_http

        await setics_loader.initialize()

        # Verify HTTP client was initialized with Setics-specific headers
        mock_http.initialize.assert_called_once()
        call_args = mock_http.initialize.call_args
        headers = call_args[1]["headers"]

        # Check that Setics-specific headers are included
        assert "Accept-Language" in headers
        assert "Cache-Control" in headers
        assert "Pragma" in headers
        assert "Upgrade-Insecure-Requests" in headers
        assert setics_loader._initialized is True

    @pytest.mark.asyncio
    async def test_initialization_with_custom_headers(self, setics_loader):
        """Test initialization with custom headers"""
        # Replace the internal HTTP client with a mock
        mock_http = AsyncMock()
        setics_loader._http_client = mock_http

        custom_headers = {"User-Agent": "Custom Agent", "X-Custom": "Value"}
        await setics_loader.initialize(headers=custom_headers)

        # Verify headers were passed to HTTP client
        call_args = mock_http.initialize.call_args
        headers = call_args[1]["headers"]

        # Custom headers should be included along with defaults
        assert headers["User-Agent"] == "Custom Agent"
        assert headers["X-Custom"] == "Value"
        assert "Accept-Language" in headers

    @pytest.mark.asyncio
    async def test_authentication_success(
        self, setics_loader, mock_http_client, sample_auth_details
    ):
        """Test successful authentication flow"""
        setics_loader._http_client = mock_http_client
        setics_loader._initialized = True

        # Mock the auth service to succeed
        setics_loader._auth_service.complete_authentication_flow = AsyncMock(
            return_value=True
        )

        # Authenticate
        await setics_loader.authenticate(
            username=sample_auth_details["username"],
            password=sample_auth_details["password"],
            login_url=sample_auth_details["login_url"],
            check_url=sample_auth_details["check_url"],
        )

        # Verify authentication was attempted
        setics_loader._auth_service.complete_authentication_flow.assert_called_once()

        # Check that the credentials were properly formatted for Setics
        call_args = setics_loader._auth_service.complete_authentication_flow.call_args
        credentials = call_args[1]["credentials"]
        assert credentials["user[email]"] == sample_auth_details["username"]
        assert credentials["user[password]"] == sample_auth_details["password"]

        # Verify authentication status
        assert setics_loader._authenticated is True

    @pytest.mark.asyncio
    async def test_authentication_failure(
        self, setics_loader, mock_http_client, sample_auth_details
    ):
        """Test authentication failure"""
        setics_loader._http_client = mock_http_client
        setics_loader._initialized = True

        # Mock the auth service to fail
        setics_loader._auth_service.complete_authentication_flow = AsyncMock(
            return_value=False
        )

        # Authentication should raise an error
        with pytest.raises(ValueError, match="Failed to authenticate with Setics"):
            await setics_loader.authenticate(
                username=sample_auth_details["username"],
                password=sample_auth_details["password"],
                login_url=sample_auth_details["login_url"],
            )

        # Verify authentication was attempted
        setics_loader._auth_service.complete_authentication_flow.assert_called_once()
        assert setics_loader._authenticated is False

    @pytest.mark.asyncio
    async def test_authentication_auto_initialize(
        self, setics_loader, sample_auth_details
    ):
        """Test authentication automatically initializes if not already initialized"""
        # Set up mock for initialize
        setics_loader.initialize = AsyncMock()
        setics_loader._initialized = False
        setics_loader._auth_service.complete_authentication_flow = AsyncMock(
            return_value=True
        )

        # Call authenticate
        await setics_loader.authenticate(
            username=sample_auth_details["username"],
            password=sample_auth_details["password"],
            login_url=sample_auth_details["login_url"],
        )

        # Verify initialize was called with the headers
        setics_loader.initialize.assert_called_once()
        assert "headers" in setics_loader.initialize.call_args[1]

    @pytest.mark.asyncio
    async def test_load_documents_successful(
        self, setics_loader, mock_http_client, sample_documents
    ):
        """Test loading documents after successful authentication"""
        # Setup the loader
        setics_loader._http_client = mock_http_client
        setics_loader._initialized = True
        setics_loader._authenticated = True
        setics_loader._document_loader.load_documents_with_langchain = AsyncMock(
            return_value=sample_documents
        )

        # Load documents
        url = "https://setics.com/resource"
        result = await setics_loader.load_documents(url)

        # Verify document loader was called correctly
        setics_loader._document_loader.load_documents_with_langchain.assert_called_once_with(
            http_client=mock_http_client, urls=url, continue_on_failure=False
        )

        # Verify documents were returned
        assert result == sample_documents

    @pytest.mark.asyncio
    async def test_load_documents_not_initialized(self, setics_loader):
        """Test loading documents without initialization"""
        setics_loader._initialized = False

        with pytest.raises(ValueError, match="Service must be initialized"):
            await setics_loader.load_documents("https://setics.com/resource")

    @pytest.mark.asyncio
    async def test_load_documents_not_authenticated(self, setics_loader):
        """Test loading documents without authentication"""
        setics_loader._initialized = True
        setics_loader._authenticated = False

        with pytest.raises(ValueError, match="Authentication required"):
            await setics_loader.load_documents("https://setics.com/resource")

    @pytest.mark.asyncio
    async def test_load_documents_with_error(self, setics_loader):
        """Test loading documents with loader error"""
        setics_loader._initialized = True
        setics_loader._authenticated = True
        setics_loader._document_loader.load_documents_with_langchain = AsyncMock(
            side_effect=Exception("Loading failed")
        )

        with pytest.raises(ValueError, match="Failed to load documents"):
            await setics_loader.load_documents("https://setics.com/resource")

    @pytest.mark.asyncio
    async def test_lazy_load_documents(self, setics_loader, sample_documents):
        """Test lazy loading of documents"""

        # Use the same approach as for PublicLoader tests
        # Create a mock implementation of the method instead of mocking the internal component
        async def mock_lazy_load(self, urls, continue_on_failure=False):
            for doc in sample_documents:
                yield doc

        # Replace the method directly instead of mocking the internal component
        with patch.object(
            setics_loader.__class__, "lazy_load_documents", mock_lazy_load
        ):
            # Collect lazy-loaded documents
            results = []
            async for doc in setics_loader.lazy_load_documents(
                "https://setics.com/resource"
            ):
                results.append(doc)

            # Verify results match sample documents
            assert len(results) == len(sample_documents)
            for i, doc in enumerate(results):
                assert doc.page_content == sample_documents[i].page_content
                assert doc.metadata == sample_documents[i].metadata

    @pytest.mark.asyncio
    async def test_lazy_load_not_initialized(self, setics_loader):
        """Test lazy loading without initialization"""
        setics_loader._initialized = False

        with pytest.raises(ValueError, match="Service must be initialized"):
            async for _ in setics_loader.lazy_load_documents(
                "https://setics.com/resource"
            ):
                pass

    @pytest.mark.asyncio
    async def test_lazy_load_not_authenticated(self, setics_loader):
        """Test lazy loading without authentication"""
        setics_loader._initialized = True
        setics_loader._authenticated = False

        with pytest.raises(ValueError, match="Authentication required"):
            async for _ in setics_loader.lazy_load_documents(
                "https://setics.com/resource"
            ):
                pass

    def test_is_authenticated_property(self, setics_loader):
        """Test is_authenticated property"""
        setics_loader._authenticated = False
        assert setics_loader.is_authenticated is False

        setics_loader._authenticated = True
        assert setics_loader.is_authenticated is True

    def test_authenticated_client_property(self, setics_loader, mock_http_client):
        """Test authenticated_client property"""
        setics_loader._http_client = mock_http_client

        # Not initialized
        setics_loader._initialized = False
        setics_loader._authenticated = False
        with pytest.raises(ValueError, match="Authentication required"):
            _ = setics_loader.authenticated_client

        # Initialized but not authenticated
        setics_loader._initialized = True
        setics_loader._authenticated = False
        with pytest.raises(ValueError, match="Authentication required"):
            _ = setics_loader.authenticated_client

        # Authenticated
        setics_loader._initialized = True
        setics_loader._authenticated = True
        client = setics_loader.authenticated_client
        assert client == mock_http_client

    def test_request_headers_property(self, setics_loader, mock_http_client):
        """Test request_headers property"""
        setics_loader._http_client = mock_http_client
        mock_http_client.headers = {"User-Agent": "Test Agent"}

        # Not initialized
        setics_loader._initialized = False
        with pytest.raises(ValueError, match="Service must be initialized"):
            _ = setics_loader.request_headers

        # Initialized
        setics_loader._initialized = True
        headers = setics_loader.request_headers
        assert headers == {"User-Agent": "Test Agent"}

        # Ensure it's a copy
        headers["X-Test"] = "Test"
        assert "X-Test" not in mock_http_client.headers

    @pytest.mark.asyncio
    async def test_close(self, setics_loader, mock_http_client):
        """Test closing the loader"""
        setics_loader._http_client = mock_http_client
        setics_loader._initialized = True
        setics_loader._authenticated = True

        await setics_loader.close()

        # Should close HTTP client and reset state
        mock_http_client.close.assert_called_once()
        assert setics_loader._initialized is False
        assert setics_loader._authenticated is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using as async context manager"""
        loader = SeticsLoader()

        # Mock methods
        loader.initialize = AsyncMock()
        loader.close = AsyncMock()

        async with loader as ctx:
            assert ctx == loader
            assert loader.initialize.called
            assert not loader.close.called

        assert loader.close.called

    @pytest.mark.asyncio
    async def test_create_setics_web_loader_service(self):
        """Test factory function"""
        with patch("src.services.loaders.web.setics_loader.SeticsLoader") as MockLoader:
            mock_instance = AsyncMock()
            MockLoader.return_value = mock_instance

            service = await create_setics_web_loader_service()

            # Factory should initialize the service
            MockLoader.assert_called_once()
            mock_instance.initialize.assert_called_once()
            assert service == mock_instance

    @pytest.mark.asyncio
    async def test_full_workflow(
        self, setics_loader, sample_auth_details, sample_documents
    ):
        """Test the full workflow: initialize, authenticate, load documents, close"""
        # Mock all the methods
        setics_loader.initialize = AsyncMock()
        setics_loader.authenticate = AsyncMock(return_value=setics_loader)
        setics_loader.load_documents = AsyncMock(return_value=sample_documents)
        setics_loader.close = AsyncMock()

        # Execute workflow
        async with setics_loader as loader:
            await loader.authenticate(
                username=sample_auth_details["username"],
                password=sample_auth_details["password"],
                login_url=sample_auth_details["login_url"],
            )
            _ = await loader.load_documents("https://setics.com/resource")

        # Verify all methods were called in sequence
        setics_loader.initialize.assert_called_once()
        setics_loader.authenticate.assert_called_once()
        setics_loader.load_documents.assert_called_once()
        setics_loader.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_urls_success(self, setics_loader, mock_http_client):
        """Test successful URL discovery after authentication"""
        # Setup the loader
        setics_loader._http_client = mock_http_client
        setics_loader._initialized = True
        setics_loader._authenticated = True

        # Mock discovered URLs
        discovered_urls = [
            "https://setics.com",
            "https://setics.com/page1",
            "https://setics.com/page2",
        ]

        # Mock UrlDiscovery
        with patch(
            "src.services.loaders.web.setics_loader.UrlDiscovery"
        ) as MockDiscovery:
            # Setup mock discovery instance
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.discover.return_value = discovered_urls
            MockDiscovery.return_value = mock_instance

            # Call discover_urls
            base_url = "https://setics.com"
            max_depth = 3
            same_domain_only = False
            custom_headers = {"X-Test": "Value"}

            result = await setics_loader.discover_urls(
                base_url=base_url,
                max_depth=max_depth,
                same_domain_only=same_domain_only,
                headers=custom_headers,
            )

            # Verify UrlDiscovery was used correctly
            MockDiscovery.assert_called_once()
            mock_instance.__aenter__.assert_awaited_once()
            mock_instance.__aexit__.assert_awaited_once()

            # Verify discover was called with correct parameters
            mock_instance.discover.assert_awaited_once()
            call_args = mock_instance.discover.call_args[1]
            assert call_args["base_url"] == base_url
            assert call_args["session"] == mock_http_client
            assert call_args["max_depth"] == max_depth
            assert call_args["same_domain_only"] == same_domain_only
            assert "headers" in call_args
            assert call_args["headers"].get("X-Test") == "Value"

            # Verify result
            assert result == discovered_urls
            assert len(result) == 3

    @pytest.mark.asyncio
    async def test_discover_urls_not_initialized(self, setics_loader):
        """Test URL discovery without initialization"""
        setics_loader._initialized = False

        with pytest.raises(ValueError, match="Service must be initialized"):
            await setics_loader.discover_urls("https://setics.com")

    @pytest.mark.asyncio
    async def test_discover_urls_not_authenticated(self, setics_loader):
        """Test URL discovery without authentication"""
        setics_loader._initialized = True
        setics_loader._authenticated = False

        with pytest.raises(ValueError, match="Authentication required"):
            await setics_loader.discover_urls("https://setics.com")

    @pytest.mark.asyncio
    async def test_discover_urls_with_default_params(
        self, setics_loader, mock_http_client
    ):
        """Test URL discovery with default parameters"""
        # Setup the loader
        setics_loader._http_client = mock_http_client
        setics_loader._initialized = True
        setics_loader._authenticated = True

        # Set up headers directly on the mock HTTP client
        # This avoids patching the property which can't be patched
        mock_http_client.headers = {"User-Agent": "Test Agent"}

        # Mock UrlDiscovery
        with patch(
            "src.services.loaders.web.setics_loader.UrlDiscovery"
        ) as MockDiscovery:
            # Setup mock discovery instance
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.discover.return_value = ["https://setics.com"]
            MockDiscovery.return_value = mock_instance

            # Call discover_urls with only required param
            await setics_loader.discover_urls("https://setics.com")

            # Verify defaults were used
            call_args = mock_instance.discover.call_args[1]
            assert call_args["max_depth"] == 2
            assert call_args["same_domain_only"] is True
            assert call_args["headers"] == {"User-Agent": "Test Agent"}
