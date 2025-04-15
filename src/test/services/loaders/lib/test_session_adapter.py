from unittest.mock import MagicMock

import pytest

from src.services.loaders.lib.session_adapter import SessionAdapter


class TestSessionAdapter:
    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def test_cookies(self):
        return {"session_id": "abc123", "user": "test_user"}

    @pytest.fixture
    def test_headers(self):
        return {"User-Agent": "Test Agent", "Accept": "text/html"}

    def test_initialization_with_default_timeout(
        self, mock_client, test_cookies, test_headers
    ):
        """Test initialization with default timeout value"""
        adapter = SessionAdapter(
            client=mock_client, cookies=test_cookies, headers=test_headers
        )

        # Verify attributes are set correctly
        assert adapter.client == mock_client
        assert adapter.headers == test_headers
        assert adapter.verify is True  # Default SSL verification
        assert adapter.timeout == 30.0  # Default timeout

        # Verify cookies wrapper
        assert hasattr(adapter.cookies, "get_dict")
        assert adapter.cookies.get_dict() == test_cookies

    def test_initialization_with_custom_timeout(
        self, mock_client, test_cookies, test_headers
    ):
        """Test initialization with custom timeout value"""
        custom_timeout = 60.0
        adapter = SessionAdapter(
            client=mock_client,
            cookies=test_cookies,
            headers=test_headers,
            timeout=custom_timeout,
        )

        assert adapter.timeout == custom_timeout

    def test_cookies_wrapper_functionality(self, mock_client, test_headers):
        """Test that the cookies wrapper properly returns the dictionary"""
        # Test with various cookie values
        test_cases = [
            {"session": "123"},
            {"key1": "value1", "key2": "value2"},
            {},  # Empty cookies
            {"complex": {"nested": "value"}},  # Complex structure
        ]

        for cookies in test_cases:
            adapter = SessionAdapter(
                client=mock_client, cookies=cookies, headers=test_headers
            )

            # Verify the cookies.get_dict() method returns the original dict
            assert adapter.cookies.get_dict() == cookies

    def test_compatibility_with_expected_interface(
        self, mock_client, test_cookies, test_headers
    ):
        """Test that the adapter provides the interface expected by WebBaseLoader"""
        adapter = SessionAdapter(
            client=mock_client, cookies=test_cookies, headers=test_headers
        )

        # These attributes/methods are expected by WebBaseLoader
        assert hasattr(adapter, "client")
        assert hasattr(adapter, "cookies")
        assert hasattr(adapter, "headers")
        assert hasattr(adapter, "verify")
        assert hasattr(adapter, "timeout")
        assert hasattr(adapter.cookies, "get_dict")
