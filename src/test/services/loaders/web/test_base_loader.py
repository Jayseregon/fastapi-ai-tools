from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.loaders.lib import DocumentLoader, HttpClient
from src.services.loaders.web.base_web_loader import BaseWebLoader


# Create a concrete implementation of BaseWebLoader for testing
class _TestableLoader(BaseWebLoader):
    """Concrete implementation of BaseWebLoader for testing"""

    async def initialize(self, headers=None, timeout=30.0):
        """Implementation of abstract initialize method"""
        await self._http_client.initialize(headers=headers, timeout=timeout)
        self._initialized = True
        return self

    async def close(self):
        """Implementation of abstract close method"""
        await self._http_client.close()
        self._initialized = False


class TestBaseWebLoader:
    @pytest.fixture
    def mock_http_client(self):
        client = AsyncMock(spec=HttpClient)
        client.initialize = AsyncMock()
        client.close = AsyncMock()
        return client

    @pytest.fixture
    def mock_document_loader(self):
        return MagicMock(spec=DocumentLoader)

    @pytest.fixture
    def base_loader(self, mock_http_client, mock_document_loader):
        """Create a TestableLoader instance with mocked dependencies"""
        return _TestableLoader(
            http_client=mock_http_client, document_loader=mock_document_loader
        )

    def test_initialization(self, base_loader, mock_http_client, mock_document_loader):
        """Test the constructor properly sets up attributes"""
        assert base_loader._http_client == mock_http_client
        assert base_loader._document_loader == mock_document_loader
        assert base_loader._initialized is False

    def test_default_initialization(self):
        """Test initialization with default parameters"""
        loader = _TestableLoader()
        assert isinstance(loader._http_client, HttpClient)
        assert isinstance(loader._document_loader, DocumentLoader)
        assert loader._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_method(self, base_loader, mock_http_client):
        """Test the initialize method properly initializes the HTTP client"""
        custom_headers = {"Custom-Header": "Value"}
        custom_timeout = 60.0

        await base_loader.initialize(headers=custom_headers, timeout=custom_timeout)

        # Verify HTTP client was initialized with correct parameters
        mock_http_client.initialize.assert_called_once_with(
            headers=custom_headers, timeout=custom_timeout
        )
        assert base_loader._initialized is True

    @pytest.mark.asyncio
    async def test_close_method(self, base_loader, mock_http_client):
        """Test the close method properly closes the HTTP client"""
        # Set the initialized flag to True
        base_loader._initialized = True

        await base_loader.close()

        # Verify HTTP client was closed
        mock_http_client.close.assert_called_once()
        assert base_loader._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager(self, base_loader, mock_http_client):
        """Test the loader works as an async context manager"""
        async with base_loader as loader:
            # Verify initialize was called
            assert loader == base_loader
            mock_http_client.initialize.assert_called_once()

        # Verify close was called
        mock_http_client.close.assert_called_once()

    def test_abstract_methods(self):
        """Test that BaseWebLoader properly enforces abstract methods"""
        with pytest.raises(TypeError, match=r"abstract method"):
            # Trying to instantiate the abstract base class should fail
            BaseWebLoader()

    @pytest.mark.asyncio
    async def test_abstract_initialize_implementation(
        self, base_loader, mock_http_client
    ):
        """Test that our concrete implementation of initialize works correctly"""
        await base_loader.initialize()

        mock_http_client.initialize.assert_called_once()
        assert base_loader._initialized is True

    @pytest.mark.asyncio
    async def test_abstract_close_implementation(self, base_loader, mock_http_client):
        """Test that our concrete implementation of close works correctly"""
        base_loader._initialized = True

        await base_loader.close()

        mock_http_client.close.assert_called_once()
        assert base_loader._initialized is False
