import datetime
from http.cookiejar import Cookie, CookieJar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.loaders.lib.cookie_manager import CookieManager


class TestCookieManager:

    @pytest.fixture
    def cookie_manager(self):
        return CookieManager()

    @pytest.fixture
    def mock_cookiejar(self):
        """Create a mock cookie jar with domain-specific cookies"""
        jar = MagicMock()

        # Create cookie objects
        cookie1 = MagicMock()
        cookie1.value = "test-value-1"
        cookie2 = MagicMock()
        cookie2.value = "test-value-2"
        cookie3 = MagicMock()
        cookie3.value = "other-domain-value"

        # Create the _cookies structure as used in CookieJar
        jar._cookies = {
            "example.com": {"/": {"cookie1": cookie1, "cookie2": cookie2}},
            "other-domain.com": {"/": {"cookie3": cookie3}},
        }
        return jar

    @pytest.fixture
    def real_cookiejar(self):
        """Create a real CookieJar with actual Cookie objects"""
        jar = CookieJar()

        # Add cookies to different domains
        now = datetime.datetime.now()
        future = now + datetime.timedelta(days=1)
        future_stamp = future.timestamp()

        # Example domain cookies
        jar.set_cookie(
            Cookie(
                0,
                "session",
                "abc123",
                None,
                False,
                "example.com",
                True,
                False,
                "/",
                True,
                False,
                future_stamp,
                False,
                None,
                None,
                {},
            )
        )
        jar.set_cookie(
            Cookie(
                0,
                "user",
                "testuser",
                None,
                False,
                "example.com",
                True,
                False,
                "/",
                True,
                False,
                future_stamp,
                False,
                None,
                None,
                {},
            )
        )

        # Other domain cookie
        jar.set_cookie(
            Cookie(
                0,
                "prefs",
                "dark-mode",
                None,
                False,
                "other-site.org",
                True,
                False,
                "/",
                True,
                False,
                future_stamp,
                False,
                None,
                None,
                {},
            )
        )

        return jar

    @pytest.mark.asyncio
    async def test_extract_domain_cookies_with_jar(
        self, cookie_manager, mock_cookiejar
    ):
        # Setup mock client with jar-style cookies
        mock_client = AsyncMock()
        mock_client.client = MagicMock()
        mock_client.client.cookies = MagicMock()
        mock_client.client.cookies.jar = mock_cookiejar

        # Create a mock for the extraction method
        expected_cookies = {
            "cookie1": "test-value-1",
            "cookie2": "test-value-2",
            "cookie3": "other-domain-value",
        }
        mock_extract = MagicMock(return_value=expected_cookies)

        # Replace the original method with our mock
        with patch.object(cookie_manager, "_extract_from_cookiejar", mock_extract):
            result = await cookie_manager.extract_domain_cookies(
                mock_client, "https://example.com"
            )

        assert result == expected_cookies
        mock_extract.assert_called_once_with(mock_cookiejar, "example.com")

    @pytest.mark.asyncio
    async def test_extract_domain_cookies_with_items(self, cookie_manager):
        # Setup mock client with dict-like cookies
        mock_client = AsyncMock()
        mock_client.client = MagicMock()

        # Create a dictionary-based cookies object that actually has items() method
        # rather than a MagicMock with items attribute
        cookie_dict = {"session": "abc123", "user": "testuser"}

        # Use a real dict instead of a MagicMock to ensure hasattr works properly
        class DictWithItems(dict):
            pass

        cookies_obj = DictWithItems(cookie_dict)

        # Assign the cookies object with a real items method
        mock_client.client.cookies = cookies_obj

        # Verify our test object actually has the items attribute for real
        assert hasattr(mock_client.client.cookies, "items")

        result = await cookie_manager.extract_domain_cookies(
            mock_client, "https://example.com"
        )
        assert result == {"session": "abc123", "user": "testuser"}

    @pytest.mark.asyncio
    async def test_extract_domain_cookies_direct_dict(self, cookie_manager):
        # Setup mock client with direct dictionary cookies
        mock_client = AsyncMock()
        mock_client.client = MagicMock()
        mock_client.client.cookies = {"session": "abc123", "user": "testuser"}

        result = await cookie_manager.extract_domain_cookies(
            mock_client, "https://example.com"
        )
        assert result == {"session": "abc123", "user": "testuser"}

    @pytest.mark.asyncio
    async def test_extract_domain_cookies_multiple_urls(self, cookie_manager):
        # Test with multiple URLs - should use first URL's domain
        mock_client = AsyncMock()
        mock_client.client = MagicMock()
        mock_client.client.cookies = MagicMock()
        mock_client.client.cookies.jar = MagicMock()

        urls = ["https://example.com/page1", "https://example.org/page2"]
        expected_cookies = {"cookie1": "value1"}

        # Create a proper mock for the extraction method
        mock_extract = MagicMock(return_value=expected_cookies)

        with patch.object(cookie_manager, "_extract_from_cookiejar", mock_extract):
            result = await cookie_manager.extract_domain_cookies(mock_client, urls)

        assert result == expected_cookies
        # Should extract domain from first URL
        mock_extract.assert_called_once_with(
            mock_client.client.cookies.jar, "example.com"
        )

    @pytest.mark.asyncio
    async def test_extract_domain_cookies_empty_urls(self, cookie_manager):
        mock_client = AsyncMock()
        result = await cookie_manager.extract_domain_cookies(mock_client, [])
        assert result == {}

    @pytest.mark.asyncio
    async def test_extract_domain_cookies_no_client_cookies(self, cookie_manager):
        # Test when client has no cookies attribute
        mock_client = AsyncMock()
        mock_client.client = MagicMock()
        # No cookies attribute

        result = await cookie_manager.extract_domain_cookies(
            mock_client, "https://example.com"
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_extract_domain_cookies_no_client(self, cookie_manager):
        # Test when client is None
        mock_client = AsyncMock()
        mock_client.client = None

        result = await cookie_manager.extract_domain_cookies(
            mock_client, "https://example.com"
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_extract_domain_cookies_exception_handling(self, cookie_manager):
        # Test exception handling
        mock_client = AsyncMock()
        mock_client.client = MagicMock()
        mock_client.client.cookies = MagicMock()
        mock_client.client.cookies.items.side_effect = Exception("Cookie error")
        mock_client.client.cookies.__iter__ = MagicMock(
            side_effect=Exception("Iteration error")
        )

        result = await cookie_manager.extract_domain_cookies(
            mock_client, "https://example.com"
        )
        assert result == {}

    def test_extract_from_cookiejar_with_target_domain(
        self, cookie_manager, mock_cookiejar
    ):
        # Test extraction with target domain matching
        result = cookie_manager._extract_from_cookiejar(mock_cookiejar, "example.com")

        assert len(result) == 3
        assert result["cookie1"] == "test-value-1"
        assert result["cookie2"] == "test-value-2"
        assert result["cookie3"] == "other-domain-value"

    def test_extract_from_cookiejar_domain_priority(self, cookie_manager):
        # Create a jar with same-named cookies on different domains
        jar = MagicMock()

        # Create cookies with same name but different values
        target_cookie = MagicMock()
        target_cookie.value = "target-domain-value"
        other_cookie = MagicMock()
        other_cookie.value = "other-domain-value"

        jar._cookies = {
            "example.com": {"/": {"shared": target_cookie}},
            "other-site.com": {"/": {"shared": other_cookie}},
        }

        # Target domain should take priority
        result = cookie_manager._extract_from_cookiejar(jar, "example.com")
        assert result["shared"] == "target-domain-value"

    def test_extract_from_cookiejar_no_cookies(self, cookie_manager):
        # Test with empty cookie jar
        jar = MagicMock()
        jar._cookies = {}

        result = cookie_manager._extract_from_cookiejar(jar)
        assert result == {}

    def test_extract_from_cookiejar_no_attribute(self, cookie_manager):
        # Test with a jar that doesn't have _cookies attribute
        jar = MagicMock()
        # No _cookies attribute

        result = cookie_manager._extract_from_cookiejar(jar)
        assert result == {}

    def test_extract_from_real_cookiejar(self, cookie_manager, real_cookiejar):
        # Test with a real CookieJar instance
        result = cookie_manager._extract_from_cookiejar(real_cookiejar, "example.com")

        # Should have extracted all cookies, with example.com ones having priority
        assert "session" in result
        assert "user" in result
        assert "prefs" in result
        assert result["session"] == "abc123"
        assert result["user"] == "testuser"
        assert result["prefs"] == "dark-mode"
