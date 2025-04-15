from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.services.loaders.web.web_image_loader import (
    WebImageLoader,
    create_web_image_loader,
)


class TestWebImageLoader:
    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client"""
        client = MagicMock()
        client.initialize = AsyncMock()
        client.headers = {"User-Agent": "Test Agent"}
        client.client = AsyncMock()
        # Mock binary content for image download tests
        client.client.get = AsyncMock(
            return_value=MagicMock(
                status_code=200, read=MagicMock(return_value=b"fake-image-data")
            )
        )
        return client

    @pytest.fixture
    def mock_auth_service(self):
        """Create a mock WebAuthentication service"""
        auth_service = MagicMock()
        auth_service.complete_authentication_flow = AsyncMock(return_value=True)
        return auth_service

    @pytest.fixture
    def mock_image_processor(self):
        """Create a mock WebImageProcessor"""
        processor = MagicMock()
        processor.extract_setics_image_urls_from_url = AsyncMock(
            return_value=[
                {
                    "url": "https://example.com/image1.jpg",
                    "id": "page-img0-12345678",
                    "source": "https://example.com/page",
                    "title": "Test Image 1",
                    "caption": "A test image",
                    "document_type": "image",
                    "timestamp": "2023-01-01T12:00:00",
                },
                {
                    "url": "https://example.com/image2.jpg",
                    "id": "page-img1-87654321",
                    "source": "https://example.com/page",
                    "title": "Test Image 2",
                    "caption": "Another test image",
                    "document_type": "image",
                    "timestamp": "2023-01-01T12:00:00",
                },
            ]
        )
        return processor

    @pytest.fixture
    def mock_image_parser(self):
        """Create a mock LLMImageBlobParser"""
        parser = MagicMock()
        parser.parse = MagicMock(
            return_value=[
                Document(
                    page_content="This is a test image showing a chart with data visualization",
                    metadata={
                        "url": "https://example.com/image1.jpg",
                        "id": "page-img0-12345678",
                        "source": "https://example.com/page",
                    },
                )
            ]
        )
        return parser

    @pytest.fixture
    def image_loader(
        self,
        mock_http_client,
        mock_auth_service,
        mock_image_processor,
        mock_image_parser,
    ):
        """Create a base WebImageLoader with mocked dependencies"""
        with (
            patch(
                "src.services.loaders.web.web_image_loader.WebAuthentication"
            ) as mock_auth_cls,
            patch(
                "src.services.loaders.web.web_image_loader.WebImageProcessor"
            ) as mock_processor_cls,
            patch(
                "src.services.loaders.web.base_web_loader.HttpClient"
            ) as mock_client_cls,
            patch(
                "src.services.loaders.web.web_image_loader.LLMImageBlobParser"
            ) as mock_parser_cls,
            patch("src.services.loaders.web.web_image_loader.ChatOpenAI"),
        ):

            # Configure mocks
            mock_auth_cls.return_value = mock_auth_service
            mock_processor_cls.return_value = mock_image_processor
            mock_client_cls.return_value = mock_http_client
            mock_parser_cls.return_value = mock_image_parser

            # Create loader instance
            loader = WebImageLoader()
            loader._initialized = False  # Start uninitialized

            yield loader

    @pytest.mark.asyncio
    async def test_initialize(self, image_loader, mock_http_client):
        """Test initializing the loader with default and custom settings"""
        # Test with default settings
        await image_loader.initialize()

        # Check that HTTP client was initialized with default headers
        mock_http_client.initialize.assert_called_once()
        assert image_loader._initialized is True

        # Reset for custom headers test
        mock_http_client.initialize.reset_mock()
        image_loader._initialized = False

        # Test with custom headers and timeout
        custom_headers = {"X-Test": "Value"}
        custom_timeout = 60.0
        await image_loader.initialize(headers=custom_headers, timeout=custom_timeout)

        # Check custom values were passed to HTTP client
        mock_http_client.initialize.assert_called_once()
        # The call_args is a tuple (args, kwargs)
        call_kwargs = mock_http_client.initialize.call_args[1]
        assert call_kwargs["timeout"] == custom_timeout
        assert "headers" in call_kwargs
        assert "X-Test" in call_kwargs["headers"]
        assert call_kwargs["headers"]["X-Test"] == "Value"

    @pytest.mark.asyncio
    async def test_create_public_loader(self, image_loader):
        """Test creating a public loader"""
        with (
            patch(
                "src.services.loaders.web.web_image_loader.WebImageLoader"
            ) as mock_loader_cls,
            patch.object(WebImageLoader, "initialize") as mock_initialize,
        ):

            mock_loader_cls.return_value = image_loader
            mock_initialize.return_value = None

            # Create public loader
            loader = await WebImageLoader.create_public_loader(
                headers={"X-Test": "Value"}, timeout=45.0
            )

            # Verify loader was initialized properly
            assert loader.mode == WebImageLoader.MODE_PUBLIC
            mock_initialize.assert_called_once_with(
                headers={"X-Test": "Value"}, timeout=45.0
            )

    @pytest.mark.asyncio
    async def test_create_protected_loader(self, image_loader):
        """Test creating a protected loader with authentication"""
        with (
            patch(
                "src.services.loaders.web.web_image_loader.WebImageLoader"
            ) as mock_loader_cls,
            patch.object(WebImageLoader, "initialize") as mock_initialize,
            patch.object(WebImageLoader, "authenticate") as mock_authenticate,
        ):

            mock_loader_cls.return_value = image_loader
            mock_initialize.return_value = None
            mock_authenticate.return_value = image_loader

            # Create protected loader
            loader = await WebImageLoader.create_protected_loader(
                username="testuser",
                password="testpass",
                login_url="https://example.com/login",
                check_url="https://example.com/dashboard",
                headers={"X-Test": "Value"},
                timeout=45.0,
            )

            # Verify loader was initialized and authenticated properly
            assert loader.mode == WebImageLoader.MODE_PROTECTED
            mock_initialize.assert_called_once_with(
                headers={"X-Test": "Value"}, timeout=45.0
            )
            mock_authenticate.assert_called_once_with(
                username="testuser",
                password="testpass",
                login_url="https://example.com/login",
                check_url="https://example.com/dashboard",
                headers={"X-Test": "Value"},
            )

    @pytest.mark.asyncio
    async def test_authenticate(self, image_loader, mock_auth_service):
        """Test authentication with a protected website"""
        # Initialize the loader first
        await image_loader.initialize()

        # Test authentication
        await image_loader.authenticate(
            username="testuser",
            password="testpass",
            login_url="https://example.com/login",
            check_url="https://example.com/dashboard",
        )

        # Verify authentication flow was called with correct parameters
        mock_auth_service.complete_authentication_flow.assert_called_once()
        call_kwargs = mock_auth_service.complete_authentication_flow.call_args[1]

        assert call_kwargs["login_url"] == "https://example.com/login"
        assert "credentials" in call_kwargs
        assert call_kwargs["credentials"].get("username") == "testuser"
        assert call_kwargs["credentials"].get("password") == "testpass"
        assert call_kwargs["check_url"] == "https://example.com/dashboard"

        # Verify authentication status was updated
        assert image_loader.is_authenticated is True
        assert image_loader.mode == WebImageLoader.MODE_PROTECTED

    @pytest.mark.asyncio
    async def test_authenticate_failure(self, image_loader, mock_auth_service):
        """Test authentication failure handling"""
        # Set authentication to fail
        mock_auth_service.complete_authentication_flow.return_value = False

        # Initialize the loader
        await image_loader.initialize()

        # Test authentication failure
        with pytest.raises(ValueError) as excinfo:
            await image_loader.authenticate(
                username="testuser",
                password="wrongpass",
                login_url="https://example.com/login",
            )

        # Verify error message
        assert "Failed to authenticate" in str(excinfo.value)

        # Verify authentication status was not updated
        assert image_loader.is_authenticated is False

    def test_detect_auth_params_default(self, image_loader):
        """Test default authentication parameters"""
        # Test with generic URL
        params = image_loader._detect_auth_params("https://example.com/login")

        assert params["username_field"] == "username"
        assert params["password_field"] == "password"
        assert params["token_field"] == "csrf_token"
        assert "Invalid credentials" in params["failure_strings"]

    def test_detect_auth_params_setics(self, image_loader):
        """Test Setics-specific authentication parameters"""
        # Test with Setics URL
        params = image_loader._detect_auth_params("https://app.setics.com/login")

        assert params["username_field"] == "user[email]"
        assert params["password_field"] == "user[password]"
        assert params["token_field"] == "authenticity_token"
        assert "Invalid Email or password" in params["failure_strings"]

    @pytest.mark.asyncio
    async def test_extract_image_urls_uninitialized(self, image_loader):
        """Test extracting image URLs when uninitialized"""
        # Ensure loader is not initialized
        image_loader._initialized = False

        # Attempt to extract image URLs
        with pytest.raises(ValueError) as excinfo:
            await image_loader._extract_image_urls_from_pages("https://example.com")

        # Verify error message
        assert "must be initialized" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_extract_image_urls_protected_unauthenticated(self, image_loader):
        """Test extracting image URLs in protected mode without authentication"""
        # Set initialized but not authenticated
        image_loader._initialized = True
        image_loader._authenticated = False
        image_loader._mode = WebImageLoader.MODE_PROTECTED

        # Attempt to extract image URLs
        with pytest.raises(ValueError) as excinfo:
            await image_loader._extract_image_urls_from_pages("https://example.com")

        # Verify error message
        assert "Authentication required" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_extract_image_urls_single(self, image_loader, mock_image_processor):
        """Test extracting image URLs from a single page"""
        # Set initialized
        image_loader._initialized = True
        image_loader._mode = WebImageLoader.MODE_PUBLIC

        # Extract image URLs
        result = await image_loader._extract_image_urls_from_pages(
            "https://example.com/page"
        )

        # Verify processor was called
        mock_image_processor.extract_setics_image_urls_from_url.assert_called_once_with(
            url="https://example.com/page", http_client=image_loader._http_client
        )

        # Verify results
        assert len(result) == 2
        assert result[0]["url"] == "https://example.com/image1.jpg"
        assert result[1]["url"] == "https://example.com/image2.jpg"

    @pytest.mark.asyncio
    async def test_extract_image_urls_multiple(
        self, image_loader, mock_image_processor
    ):
        """Test extracting image URLs from multiple pages"""
        # Set initialized
        image_loader._initialized = True
        image_loader._mode = WebImageLoader.MODE_PUBLIC

        # Extract image URLs from multiple pages
        result = await image_loader._extract_image_urls_from_pages(
            ["https://example.com/page1", "https://example.com/page2"]
        )

        # Verify processor was called twice
        assert mock_image_processor.extract_setics_image_urls_from_url.call_count == 2

        # Verify results were combined
        assert len(result) == 4  # Two images from each page

    @pytest.mark.asyncio
    async def test_extract_image_urls_error_handling(
        self, image_loader, mock_image_processor
    ):
        """Test error handling during image URL extraction"""
        # Set initialized
        image_loader._initialized = True
        image_loader._mode = WebImageLoader.MODE_PUBLIC

        # Set one URL to fail
        mock_image_processor.extract_setics_image_urls_from_url.side_effect = [
            Exception("Connection error"),
            [{"url": "https://example.com/image2.jpg"}],
        ]

        # Extract with continue_on_failure=True
        result = await image_loader._extract_image_urls_from_pages(
            ["https://example.com/page1", "https://example.com/page2"],
            continue_on_failure=True,
        )

        # Verify we got results from the second URL only
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/image2.jpg"

        # Reset side effect
        mock_image_processor.extract_setics_image_urls_from_url.side_effect = None
        mock_image_processor.extract_setics_image_urls_from_url.reset_mock()

        # Test with continue_on_failure=False
        mock_image_processor.extract_setics_image_urls_from_url.side_effect = Exception(
            "Connection error"
        )

        with pytest.raises(Exception) as excinfo:
            await image_loader._extract_image_urls_from_pages(
                "https://example.com/page", continue_on_failure=False
            )

        assert "Connection error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_custom_prompt(self, image_loader):
        """Test the custom prompt template"""
        prompt_template = image_loader._custom_prompt()

        # Verify it's a PromptTemplate with expected content
        assert hasattr(prompt_template, "format")

        # Check prompt content
        prompt_str = prompt_template.template
        assert "Concise Summary" in prompt_str
        assert "Visual Description" in prompt_str
        assert "Text Content" in prompt_str
        assert "FORMATTING INSTRUCTIONS" in prompt_str

    @pytest.mark.asyncio
    @patch("src.services.loaders.web.web_image_loader.LLMImageBlobParser")
    @patch("src.services.loaders.web.web_image_loader.ChatOpenAI")
    async def test_download_and_parse_images(
        self,
        mock_chat,
        mock_parser_cls,
        image_loader,
        mock_image_processor,
        mock_image_parser,
    ):
        """Test downloading and parsing images"""
        # Setup mocks
        mock_parser_cls.return_value = mock_image_parser

        # Set initialized
        image_loader._initialized = True
        image_loader._mode = WebImageLoader.MODE_PUBLIC

        # Download and parse images
        results = await image_loader.download_and_parse_images(
            "https://example.com/page"
        )

        # Verify extraction was called
        mock_image_processor.extract_setics_image_urls_from_url.assert_called_once()

        # Verify HTTP client was used to download images
        assert image_loader._http_client.client.get.call_count == 2  # Two images

        # Verify parser was called
        assert mock_image_parser.parse.call_count == 2

        # Verify results
        assert len(results) == 2  # One document per image
        assert (
            results[0].page_content
            == "This is a test image showing a chart with data visualization"
        )

    @pytest.mark.asyncio
    async def test_download_and_parse_images_no_images(
        self, image_loader, mock_image_processor
    ):
        """Test downloading when no images are found"""
        # Set initialized
        image_loader._initialized = True
        image_loader._mode = WebImageLoader.MODE_PUBLIC

        # Set image processor to return empty list
        mock_image_processor.extract_setics_image_urls_from_url.return_value = []

        # Download and parse images
        results = await image_loader.download_and_parse_images(
            "https://example.com/page"
        )

        # Verify results
        assert results == []

        # Verify HTTP client was not called to download images
        assert image_loader._http_client.client.get.call_count == 0

    @pytest.mark.asyncio
    @patch("src.services.loaders.web.web_image_loader.LLMImageBlobParser")
    @patch("src.services.loaders.web.web_image_loader.ChatOpenAI")
    async def test_download_and_parse_images_error_handling(
        self,
        mock_chat,
        mock_parser_cls,
        image_loader,
        mock_image_processor,
        mock_image_parser,
    ):
        """Test error handling during image download and parsing"""
        # Setup mocks
        mock_parser_cls.return_value = mock_image_parser

        # Set initialized
        image_loader._initialized = True
        image_loader._mode = WebImageLoader.MODE_PUBLIC

        # Set HTTP client to fail on second image
        image_loader._http_client.client.get.side_effect = [
            MagicMock(status_code=200, read=MagicMock(return_value=b"fake-image-data")),
            Exception("Download failed"),
        ]

        # Download with continue_on_failure=True
        results = await image_loader.download_and_parse_images(
            "https://example.com/page", continue_on_failure=True
        )

        # Verify we got results from the first image only
        assert len(results) == 1

        # Reset side effect
        image_loader._http_client.client.get.side_effect = None

        # Test with continue_on_failure=False
        image_loader._http_client.client.get.side_effect = Exception("Download failed")

        with pytest.raises(Exception) as excinfo:
            await image_loader.download_and_parse_images(
                "https://example.com/page", continue_on_failure=False
            )

        assert "Download failed" in str(excinfo.value)

    def test_is_authenticated_property(self, image_loader):
        """Test is_authenticated property"""
        # Default state
        assert image_loader.is_authenticated is False

        # After authentication
        image_loader._authenticated = True
        assert image_loader.is_authenticated is True

    def test_request_headers_property(self, image_loader):
        """Test request_headers property"""
        # Set initialized
        image_loader._initialized = True
        image_loader._http_client.headers = {"User-Agent": "Test", "X-Custom": "Value"}

        # Get headers
        headers = image_loader.request_headers

        # Verify headers
        assert headers == {"User-Agent": "Test", "X-Custom": "Value"}

        # Verify we get a copy, not the original
        headers["New-Header"] = "New-Value"
        assert "New-Header" not in image_loader._http_client.headers

    def test_request_headers_uninitialized(self, image_loader):
        """Test request_headers property when uninitialized"""
        # Ensure loader is not initialized
        image_loader._initialized = False

        # Attempt to get headers
        with pytest.raises(ValueError) as excinfo:
            _ = image_loader.request_headers

        # Verify error message
        assert "must be initialized" in str(excinfo.value)

    def test_mode_property(self, image_loader):
        """Test mode property"""
        # Default mode
        assert image_loader.mode == WebImageLoader.MODE_PUBLIC

        # Change mode
        image_loader._mode = WebImageLoader.MODE_PROTECTED
        assert image_loader.mode == WebImageLoader.MODE_PROTECTED

    @pytest.mark.asyncio
    async def test_close(self, image_loader):
        """Test closing the loader"""
        # Set state before closing
        image_loader._initialized = True
        image_loader._authenticated = True

        # Create mock for parent close method
        with patch(
            "src.services.loaders.web.base_web_loader.BaseWebLoader.close"
        ) as mock_parent_close:
            mock_parent_close.return_value = None

            # Close the loader
            await image_loader.close()

            # Verify parent close was called
            mock_parent_close.assert_called_once()

            # Verify authentication status was reset
            assert image_loader._authenticated is False

    @pytest.mark.asyncio
    async def test_create_web_image_loader_public(self):
        """Test create_web_image_loader factory function for public mode"""
        with patch(
            "src.services.loaders.web.web_image_loader.WebImageLoader"
        ) as mock_loader_cls:
            # Create mock for public loader creation
            mock_public_loader = AsyncMock()
            mock_loader_cls.create_public_loader = mock_public_loader

            # Call factory function
            await create_web_image_loader(
                protected=False, headers={"X-Test": "Value"}, timeout=45.0
            )

            # Verify public loader creation was called
            mock_public_loader.assert_called_once_with(
                headers={"X-Test": "Value"}, timeout=45.0
            )

    @pytest.mark.asyncio
    async def test_create_web_image_loader_protected(self):
        """Test create_web_image_loader factory function for protected mode"""
        with patch(
            "src.services.loaders.web.web_image_loader.WebImageLoader"
        ) as mock_loader_cls:
            # Create mock for protected loader creation
            mock_protected_loader = AsyncMock()
            mock_loader_cls.create_protected_loader = mock_protected_loader

            # Call factory function
            await create_web_image_loader(
                protected=True,
                username="testuser",
                password="testpass",
                login_url="https://example.com/login",
                check_url="https://example.com/dashboard",
                headers={"X-Test": "Value"},
                timeout=45.0,
            )

            # Verify protected loader creation was called
            mock_protected_loader.assert_called_once_with(
                username="testuser",
                password="testpass",
                login_url="https://example.com/login",
                check_url="https://example.com/dashboard",
                headers={"X-Test": "Value"},
                timeout=45.0,
            )

    @pytest.mark.asyncio
    async def test_create_web_image_loader_protected_missing_params(self):
        """Test create_web_image_loader with missing protected mode parameters"""
        # Missing username
        with pytest.raises(ValueError) as excinfo:
            await create_web_image_loader(
                protected=True,
                password="testpass",
                login_url="https://example.com/login",
            )
        assert "Username" in str(excinfo.value)

        # Missing password
        with pytest.raises(ValueError) as excinfo:
            await create_web_image_loader(
                protected=True,
                username="testuser",
                login_url="https://example.com/login",
            )
        assert "password" in str(excinfo.value)

        # Missing login_url
        with pytest.raises(ValueError) as excinfo:
            await create_web_image_loader(
                protected=True, username="testuser", password="testpass"
            )
        assert "login_url" in str(excinfo.value)
