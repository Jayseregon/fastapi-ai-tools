import asyncio
import json
from unittest.mock import MagicMock, mock_open, patch

import pytest
import pytest_asyncio

from src.services.loaders.lib.url_discovery import UrlDiscovery


class AsyncMock(MagicMock):
    """Improved AsyncMock that works with inspect.iscoroutinefunction"""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def __await__(self):
        return self().__await__()


@pytest_asyncio.fixture
async def url_discovery_instance():
    """Fixture that returns a fresh instance of UrlDiscovery for each test."""
    instance = UrlDiscovery()
    yield instance
    await instance.reset()


@pytest_asyncio.fixture
async def mock_async_client():
    """Fixture that returns a mocked async HTTP client."""
    mock_client = MagicMock()

    # Create an AsyncMock that works with inspection
    mock_get = AsyncMock()

    # Make the AsyncMock pass the inspection
    async def _mock_get(*args, **kwargs):
        return await mock_get(*args, **kwargs)

    # Set the async method and maintain the mock
    mock_client.get = _mock_get
    mock_client._mock_get = (
        mock_get  # Store reference to the actual mock for assertions
    )

    # Configure the mock's return value
    mock_response = MagicMock()
    mock_response.text = "<html></html>"
    mock_get.return_value = mock_response

    return mock_client


class TestUrlDiscoveryInitialization:
    def test_init_default_values(self, url_discovery_instance):
        """Test that initialization sets default values correctly."""
        assert url_discovery_instance.base_url is None
        assert url_discovery_instance.session is None
        assert url_discovery_instance.headers is None
        assert url_discovery_instance.max_depth is None
        assert url_discovery_instance.same_domain_only is None
        assert url_discovery_instance.visited_urls == set()
        assert url_discovery_instance.discovered_urls == set()
        assert url_discovery_instance._initialized is False

    def test_initialize_method(self, url_discovery_instance, mock_async_client):
        """Test the initialize method with custom parameters."""
        base_url = "https://example.com"
        headers = {"User-Agent": "TestBot"}
        max_depth = 3

        url_discovery_instance.initialize(
            base_url=base_url,
            session=mock_async_client,
            headers=headers,
            max_depth=max_depth,
            same_domain_only=False,
        )

        assert url_discovery_instance.base_url == base_url
        assert url_discovery_instance.session == mock_async_client
        assert url_discovery_instance.headers == headers
        assert url_discovery_instance.max_depth == max_depth
        assert url_discovery_instance.same_domain_only is False
        assert url_discovery_instance._initialized is True

    def test_initialize_invalid_session(self, url_discovery_instance):
        """Test initialize with invalid session (non-async get method)."""

        class InvalidSession:
            def get(self):  # Not an async method
                pass

        with pytest.raises(ValueError, match="Session must have an async get method"):
            url_discovery_instance.initialize(
                base_url="https://example.com", session=InvalidSession()
            )

    @pytest.mark.asyncio
    async def test_reset_method(self, url_discovery_instance, mock_async_client):
        """Test that reset clears all state properly."""
        # Apply timeout to all async tests
        async with asyncio.timeout(2.0):
            # First initialize with values
            url_discovery_instance.initialize(
                "https://example.com", session=mock_async_client
            )
            assert url_discovery_instance._initialized is True

            # Then reset
            await url_discovery_instance.reset()

            # Check that values are reset to defaults
            assert url_discovery_instance.base_url is None
            assert url_discovery_instance.session is None
            assert url_discovery_instance.headers is None
            assert url_discovery_instance.max_depth is None
            assert url_discovery_instance.same_domain_only is None
            assert url_discovery_instance.visited_urls == set()
            assert url_discovery_instance.discovered_urls == set()
            assert url_discovery_instance._initialized is False


class TestUrlDiscoveryProcess:
    @pytest.mark.asyncio
    async def test_discover_urls_basic(self, url_discovery_instance, mock_async_client):
        """Test basic URL discovery functionality."""
        # Apply timeout to all async tests
        async with asyncio.timeout(2.0):
            # Setup mock response
            mock_response = MagicMock()
            mock_response.text = "<html><body><a href='https://example.com/page1'>Page 1</a></body></html>"
            mock_async_client._mock_get.return_value = mock_response

            # Setup BeautifulSoup mock to return empty links to prevent crawling
            mock_soup = MagicMock()
            mock_soup.find_all.return_value = []  # No links to follow

            with patch(
                "src.services.loaders.lib.url_discovery.BeautifulSoup",
                return_value=mock_soup,
            ):
                # Run the discovery
                results = await url_discovery_instance.discover(
                    base_url="https://example.com", session=mock_async_client
                )

                # Assertions
                assert "https://example.com" in results
                assert len(url_discovery_instance.visited_urls) >= 1

                # Verify the HTTP client was called correctly with any_call instead of last call
                mock_async_client._mock_get.assert_any_call(
                    "https://example.com", headers={}
                )

    @pytest.mark.asyncio
    async def test_discover_urls_depth_simple(
        self, url_discovery_instance, mock_async_client
    ):
        """Test URL discovery with multiple depth levels."""
        # Apply timeout to all async tests
        async with asyncio.timeout(2.0):
            # Setup URLs and responses
            base_url = "https://example.com"
            page1_url = "https://example.com/page1"

            # Setup a counter to ensure we exit after correct number of calls
            call_count = 0

            # Use a much simpler approach - manually control responses and track calls
            async def mock_get(url, **kwargs):
                nonlocal call_count
                call_count += 1

                response = MagicMock()

                # First call returns base page with link to page1
                if url == base_url and call_count == 1:
                    response.text = (
                        f'<html><body><a href="{page1_url}">Link</a></body></html>'
                    )
                # Second call should be to page1, return empty page
                elif url == page1_url and call_count == 2:
                    response.text = "<html><body>Empty page</body></html>"
                else:
                    # Unexpected URL - help with debugging
                    response.text = f"<html><body>Unexpected URL: {url}</body></html>"

                return response

            # Setup the client to use our controlled function
            mock_async_client.get = mock_get

            # Create a simple mock soup implementation
            with patch(
                "src.services.loaders.lib.url_discovery.BeautifulSoup"
            ) as mock_bs:
                # Base URL soup finds page1 link
                base_soup = MagicMock()
                page1_link = MagicMock()
                page1_link.__getitem__.return_value = page1_url
                base_soup.find_all.return_value = [page1_link]

                # Page1 soup finds no links
                page1_soup = MagicMock()
                page1_soup.find_all.return_value = []

                # Return different soups based on call sequence
                mock_bs.side_effect = [base_soup, page1_soup]

                # Run the test
                results = await url_discovery_instance.discover(
                    base_url=base_url, session=mock_async_client, max_depth=1
                )

                # Verify results
                assert base_url in results, "Base URL should be discovered"
                assert page1_url in results, "Page1 URL should be discovered"
                assert len(results) == 2, "Should discover exactly 2 URLs"
                assert call_count == 2, "Should make exactly 2 HTTP requests"

    @pytest.mark.asyncio
    async def test_discover_urls_error_handling(
        self, url_discovery_instance, mock_async_client
    ):
        """Test handling of errors during URL discovery."""
        # Apply timeout to all async tests
        async with asyncio.timeout(2.0):
            # Set up the mock to raise an exception when called
            mock_async_client._mock_get.side_effect = Exception("Test error")

            # Set the discovered_urls to prevent auto-discovery
            url_discovery_instance.discovered_urls = set()

            # Run discovery with timeout - should not raise exceptions to caller
            results = await url_discovery_instance.discover(
                base_url="https://example.com", session=mock_async_client
            )

            # URL should be visited but not discovered due to the error
            assert "https://example.com" in url_discovery_instance.visited_urls
            assert "https://example.com" not in url_discovery_instance.discovered_urls
            assert len(results) == 0

    def test_is_valid_url(self, url_discovery_instance, mock_async_client):
        """Test URL validation logic."""
        base_domain = "example.com"

        # Initialize with same_domain_only=True
        url_discovery_instance.initialize(
            "https://example.com", session=mock_async_client, same_domain_only=True
        )

        # Valid URLs (same domain)
        assert (
            url_discovery_instance._is_valid_url(
                "https://example.com/page", base_domain
            )
            is True
        )

        # Invalid URLs (different domain)
        assert (
            url_discovery_instance._is_valid_url("https://otherdomain.com", base_domain)
            is False
        )

        # Non-HTTP URLs
        assert (
            url_discovery_instance._is_valid_url("ftp://example.com", base_domain)
            is False
        )
        assert (
            url_discovery_instance._is_valid_url("mailto:user@example.com", base_domain)
            is False
        )

        # With same_domain_only=False
        url_discovery_instance.initialize(
            "https://example.com", session=mock_async_client, same_domain_only=False
        )
        assert (
            url_discovery_instance._is_valid_url("https://otherdomain.com", base_domain)
            is True
        )


class TestUrlDiscoveryFileOperations:
    @pytest.mark.asyncio
    async def test_to_json(self, url_discovery_instance, mock_async_client, tmp_path):
        """Test saving discovered URLs to a JSON file."""
        # Apply timeout to all async tests
        async with asyncio.timeout(2.0):
            # Setup discovered URLs
            url_discovery_instance.initialize(
                "https://example.com", session=mock_async_client
            )
            # Pre-populate the discovered URLs to prevent _discover_urls from being called
            url_discovery_instance.discovered_urls = {
                "https://example.com",
                "https://example.com/page1",
            }

            # Create a temp file path
            file_path = tmp_path / "discovered_urls.json"

            # Explicitly patch _discover_urls to avoid any possibility of it being called
            with patch.object(
                url_discovery_instance, "_discover_urls"
            ) as mock_discover:
                # Mock file operations
                with patch("pathlib.Path.open", mock_open()) as mock_file:
                    await url_discovery_instance.to_json(file_path)

                    # Check if file was opened
                    mock_file.assert_called_once_with("w")

                    # Check if JSON was written
                    file_handle = mock_file()
                    file_handle.write.assert_called()

                    # Extract and validate the written content
                    written_content = ""
                    for call_args in file_handle.write.call_args_list:
                        written_content += call_args[0][0]

                    # Convert to Python object and check
                    try:
                        urls = json.loads(written_content)
                        assert set(urls) == {
                            "https://example.com",
                            "https://example.com/page1",
                        }
                    except json.JSONDecodeError:
                        pytest.fail("Written content is not valid JSON")

                # Verify _discover_urls was not called
                mock_discover.assert_not_called()

    @pytest.mark.asyncio
    async def test_to_json_create_parent_dirs(
        self, url_discovery_instance, mock_async_client
    ):
        """Test that to_json creates parent directories if they don't exist."""
        # Apply timeout to all async tests
        async with asyncio.timeout(2.0):
            url_discovery_instance.initialize(
                "https://example.com", session=mock_async_client
            )

            # Pre-set discovered_urls to prevent auto-discovery
            url_discovery_instance.discovered_urls = {"https://example.com"}

            with patch("pathlib.Path.mkdir") as mock_mkdir:
                with patch("pathlib.Path.open", mock_open()):
                    await url_discovery_instance.to_json("some/nested/path/urls.json")
                    mock_mkdir.assert_called_with(parents=True, exist_ok=True)


class TestUrlDiscoveryAsyncContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_async_client):
        """Test the async context manager protocol."""
        # Apply timeout to all async tests
        async with asyncio.timeout(2.0):
            async with UrlDiscovery() as discovery:
                discovery.initialize("https://example.com", session=mock_async_client)
                assert discovery._initialized is True

            # Should be reset after exiting context
            assert discovery._initialized is False
            assert discovery.base_url is None

    @pytest.mark.asyncio
    async def test_async_context_manager_with_discovery(self, mock_async_client):
        """Test async context manager with discovery operation."""
        # Apply timeout to all async tests
        async with asyncio.timeout(2.0):
            # Setup mock response
            mock_response = MagicMock()
            mock_response.text = "<html><body><a href='https://example.com/page1'>Page 1</a></body></html>"
            mock_async_client._mock_get.return_value = mock_response

            # Create mock soup that returns an empty list of links to simplify test
            mock_soup = MagicMock()
            mock_soup.find_all.return_value = []

            with patch(
                "src.services.loaders.lib.url_discovery.BeautifulSoup",
                return_value=mock_soup,
            ):
                async with UrlDiscovery() as discovery:
                    discovery.initialize(
                        "https://example.com", session=mock_async_client
                    )

                    # Add a timeout to the discovery operation
                    results = await asyncio.wait_for(discovery.discover(), timeout=1.0)

                    assert "https://example.com" in results
                    assert discovery._initialized is True

                # After context exit
                assert discovery._initialized is False
