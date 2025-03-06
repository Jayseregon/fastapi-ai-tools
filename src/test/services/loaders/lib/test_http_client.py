from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.loaders.lib.http_client import HttpClient


class TestHttpClient:
    @pytest.fixture
    def mock_httpx_client(self):
        """Create a mock httpx.AsyncClient"""
        with patch("httpx.AsyncClient") as mock_client:
            # Create an instance mock that will be returned by the constructor
            instance = AsyncMock()
            mock_client.return_value = instance

            # Setup standard response
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            instance.get.return_value = mock_response
            instance.post.return_value = mock_response

            # Return both the class mock and instance mock for flexibility in tests
            yield {
                "class_mock": mock_client,
                "instance": instance,
                "response": mock_response,
            }

    @pytest.mark.asyncio
    async def test_initialization_default_values(self, mock_httpx_client):
        """Test initialization with default values"""
        client = HttpClient()

        # Verify initial state
        assert client.client is None
        assert "User-Agent" in client.headers
        assert "Accept" in client.headers
        assert client.follow_redirects is True
        assert client.timeout_duration == 30.0
        assert client._initialized is False

        # Initialize and verify
        await client.initialize()
        assert client._initialized is True

        # Verify the httpx client was created with correct parameters
        mock_httpx_client["class_mock"].assert_called_once()
        call_kwargs = mock_httpx_client["class_mock"].call_args.kwargs
        assert call_kwargs["follow_redirects"] is True
        assert isinstance(call_kwargs["timeout"], httpx.Timeout)
        assert isinstance(call_kwargs["limits"], httpx.Limits)

    @pytest.mark.asyncio
    async def test_initialization_custom_values(self, mock_httpx_client):
        """Test initialization with custom values"""
        custom_headers = {"X-Test": "Test Value", "User-Agent": "Custom Agent"}
        custom_timeout = 60.0

        client = HttpClient(
            default_headers=custom_headers,
            follow_redirects=False,
            timeout=custom_timeout,
        )

        # Verify initial state reflects custom values
        assert client.headers["X-Test"] == "Test Value"
        assert client.headers["User-Agent"] == "Custom Agent"
        assert client.follow_redirects is False
        assert client.timeout_duration == custom_timeout

        # Initialize and verify
        await client.initialize()

        # Verify the httpx client was created with correct parameters
        mock_httpx_client["class_mock"].assert_called_once()
        call_kwargs = mock_httpx_client["class_mock"].call_args.kwargs
        assert call_kwargs["follow_redirects"] is False
        assert isinstance(call_kwargs["timeout"], httpx.Timeout)

    @pytest.mark.asyncio
    async def test_initialize_with_custom_parameters(self, mock_httpx_client):
        """Test initialize method with custom headers and timeout"""
        client = HttpClient()
        custom_headers = {"X-Custom": "Value"}
        custom_timeout = 45.0

        await client.initialize(headers=custom_headers, timeout=custom_timeout)

        # Verify headers were updated
        assert client.headers["X-Custom"] == "Value"

        # Verify timeout was passed to httpx client
        mock_httpx_client["class_mock"].assert_called_once()
        call_kwargs = mock_httpx_client["class_mock"].call_args.kwargs
        assert call_kwargs["timeout"].read == custom_timeout

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mock_httpx_client):
        """Test initialize is idempotent (can be called multiple times safely)"""
        client = HttpClient()

        # First initialization
        await client.initialize()
        assert client._initialized is True
        first_call_count = mock_httpx_client["class_mock"].call_count

        # Second initialization shouldn't create a new client
        await client.initialize()
        assert mock_httpx_client["class_mock"].call_count == first_call_count

    @pytest.mark.asyncio
    async def test_get_request(self, mock_httpx_client):
        """Test GET request functionality"""
        client = HttpClient()
        await client.initialize()

        # Setup test parameters
        url = "https://example.com"
        headers = {"X-Test": "Value"}
        params = {"q": "test"}

        # Make request
        _ = await client.get(url, headers=headers, params=params)

        # Verify the request was made with correct parameters
        mock_instance = mock_httpx_client["instance"]
        mock_instance.get.assert_called_once()
        call_args, call_kwargs = mock_instance.get.call_args
        assert call_args[0] == url
        assert "X-Test" in call_kwargs["headers"]
        assert call_kwargs["params"] == params

        # Verify raise_for_status was called
        mock_httpx_client["response"].raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_request(self, mock_httpx_client):
        """Test POST request functionality"""
        client = HttpClient()
        await client.initialize()

        # Setup test parameters
        url = "https://example.com"
        data = {"form": "data"}
        json = {"json": "data"}
        headers = {"X-Test": "Value"}

        # Make request with form data
        await client.post(url, data=data, headers=headers)

        # Verify the request was made with correct parameters
        mock_instance = mock_httpx_client["instance"]
        mock_instance.post.assert_called_once()
        call_args, call_kwargs = mock_instance.post.call_args
        assert call_args[0] == url
        assert call_kwargs["data"] == data
        assert "X-Test" in call_kwargs["headers"]

        # Reset mock
        mock_instance.post.reset_mock()

        # Make request with JSON data
        await client.post(url, json=json)

        # Verify the JSON request
        mock_instance.post.assert_called_once()
        call_args, call_kwargs = mock_instance.post.call_args
        assert call_kwargs["json"] == json

    @pytest.mark.asyncio
    async def test_get_auto_initialize(self, mock_httpx_client):
        """Test GET automatically initializes if needed"""
        client = HttpClient()
        assert client._initialized is False

        # Make GET request without explicit initialization
        await client.get("https://example.com")

        # Verify client was initialized
        assert client._initialized is True
        mock_httpx_client["instance"].get.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_auto_initialize(self, mock_httpx_client):
        """Test POST automatically initializes if needed"""
        client = HttpClient()
        assert client._initialized is False

        # Make POST request without explicit initialization
        await client.post("https://example.com")

        # Verify client was initialized
        assert client._initialized is True
        mock_httpx_client["instance"].post.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_error_handling(self, mock_httpx_client):
        """Test handling of HTTP status errors"""
        client = HttpClient()
        await client.initialize()

        # Mock a 404 response
        http_error = httpx.HTTPStatusError(
            message="Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )
        mock_httpx_client["response"].raise_for_status.side_effect = http_error

        # Verify error is propagated
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await client.get("https://example.com")

        assert excinfo.value == http_error

    @pytest.mark.asyncio
    async def test_request_error_handling(self, mock_httpx_client):
        """Test handling of request errors (network, timeout, etc.)"""
        client = HttpClient()
        await client.initialize()

        # Mock a request error (like connection error)
        mock_instance = mock_httpx_client["instance"]
        request_error = httpx.RequestError("Connection error", request=MagicMock())
        mock_instance.get.side_effect = request_error

        # Verify error is propagated
        with pytest.raises(httpx.RequestError) as excinfo:
            await client.get("https://example.com")

        assert excinfo.value == request_error

    @pytest.mark.asyncio
    async def test_general_error_handling(self, mock_httpx_client):
        """Test handling of general exceptions"""
        client = HttpClient()
        await client.initialize()

        # Mock a general exception
        mock_instance = mock_httpx_client["instance"]
        mock_instance.get.side_effect = Exception("Unexpected error")

        # Verify error is propagated
        with pytest.raises(Exception) as excinfo:
            await client.get("https://example.com")

        assert str(excinfo.value) == "Unexpected error"

    @pytest.mark.asyncio
    async def test_close(self, mock_httpx_client):
        """Test client cleanup"""
        client = HttpClient()
        await client.initialize()
        assert client._initialized is True

        # Close the client
        await client.close()

        # Verify client was closed and state reset
        mock_httpx_client["instance"].aclose.assert_called_once()
        assert client.client is None
        assert client._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_httpx_client):
        """Test async context manager functionality"""
        # Use client as async context manager
        async with HttpClient() as client:
            # Verify client was initialized
            assert client._initialized is True
            assert client.client is not None

            # Make a request inside context
            await client.get("https://example.com")
            mock_httpx_client["instance"].get.assert_called_once()

        # Verify client was closed after exiting context
        mock_httpx_client["instance"].aclose.assert_called_once()
        assert client.client is None
        assert client._initialized is False
