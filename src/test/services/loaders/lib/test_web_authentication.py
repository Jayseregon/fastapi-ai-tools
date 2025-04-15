from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.loaders.lib.http_client import HttpClient
from src.services.loaders.lib.web_authentication import WebAuthentication


class TestWebAuthentication:
    @pytest.fixture
    def web_auth(self):
        return WebAuthentication()

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HttpClient for testing"""
        client = MagicMock(spec=HttpClient)
        client.get = AsyncMock()
        client.post = AsyncMock()
        return client

    @pytest.fixture
    def test_login_html_with_token(self):
        """HTML content with token for testing"""
        return """
        <html>
            <form method="post" action="/login">
                <input type="hidden" name="authenticity_token" value="test_token_123" />
                <input type="text" name="username" />
                <input type="password" name="password" />
                <button type="submit">Login</button>
            </form>
        </html>
        """

    @pytest.fixture
    def test_login_html_without_token(self):
        """HTML content without token for testing"""
        return """
        <html>
            <form method="post" action="/login">
                <input type="text" name="username" />
                <input type="password" name="password" />
                <button type="submit">Login</button>
            </form>
        </html>
        """

    @pytest.fixture
    def test_login_html_with_custom_token(self):
        """HTML content with custom token field for testing"""
        return """
        <html>
            <form method="post" action="/login">
                <input type="hidden" name="custom_token" value="test_custom_token_456" />
                <input type="text" name="username" />
                <input type="password" name="password" />
                <button type="submit">Login</button>
            </form>
        </html>
        """

    # Tests for extract_token method

    def test_extract_token_success(self, web_auth, test_login_html_with_token):
        """Test successful token extraction from HTML"""
        token = web_auth.extract_token(test_login_html_with_token)
        assert token == "test_token_123"

    def test_extract_token_not_found(self, web_auth, test_login_html_without_token):
        """Test extract_token when token is not in HTML"""
        token = web_auth.extract_token(test_login_html_without_token)
        assert token is None

    def test_extract_token_custom_field(
        self, web_auth, test_login_html_with_custom_token
    ):
        """Test extract_token with custom token field name"""
        token = web_auth.extract_token(
            test_login_html_with_custom_token, token_field="custom_token"
        )
        assert token == "test_custom_token_456"

    # Tests for get_authenticity_token method

    @pytest.mark.asyncio
    async def test_get_authenticity_token_success(
        self, web_auth, mock_http_client, test_login_html_with_token
    ):
        """Test successful token retrieval from login page"""
        # Configure mocks
        mock_response = MagicMock()
        mock_response.text = test_login_html_with_token
        mock_http_client.get.return_value = mock_response

        token = await web_auth.get_authenticity_token(
            mock_http_client, "https://example.com/login"
        )

        assert token == "test_token_123"
        # Verify the client made requests to both the base domain and login page
        assert mock_http_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_authenticity_token_not_found(
        self, web_auth, mock_http_client, test_login_html_without_token
    ):
        """Test get_authenticity_token when token is not present"""
        mock_response = MagicMock()
        mock_response.text = test_login_html_without_token
        mock_http_client.get.return_value = mock_response

        with pytest.raises(
            ValueError, match="Could not find authenticity_token on login page"
        ):
            await web_auth.get_authenticity_token(
                mock_http_client, "https://example.com/login"
            )

    @pytest.mark.asyncio
    async def test_get_authenticity_token_custom_field(
        self, web_auth, mock_http_client, test_login_html_with_custom_token
    ):
        """Test get_authenticity_token with custom token field"""
        mock_response = MagicMock()
        mock_response.text = test_login_html_with_custom_token
        mock_http_client.get.return_value = mock_response

        token = await web_auth.get_authenticity_token(
            mock_http_client, "https://example.com/login", token_field="custom_token"
        )

        assert token == "test_custom_token_456"

    @pytest.mark.asyncio
    async def test_get_authenticity_token_with_browser_headers(
        self, web_auth, mock_http_client, test_login_html_with_token
    ):
        """Test get_authenticity_token with browser headers"""
        mock_response = MagicMock()
        mock_response.text = test_login_html_with_token
        mock_http_client.get.return_value = mock_response

        browser_headers = {"Accept-Language": "en-US,en;q=0.9"}

        token = await web_auth.get_authenticity_token(
            mock_http_client,
            "https://example.com/login",
            browser_headers=browser_headers,
        )

        assert token == "test_token_123"
        # Verify headers were passed to the client
        _, kwargs = mock_http_client.get.call_args_list[0]
        assert kwargs["headers"] == browser_headers

    @pytest.mark.asyncio
    async def test_get_authenticity_token_http_error(self, web_auth, mock_http_client):
        """Test get_authenticity_token with HTTP error"""
        # Configure mock to raise an exception
        mock_http_client.get.side_effect = httpx.HTTPError("HTTP Error")

        with pytest.raises(Exception):
            await web_auth.get_authenticity_token(
                mock_http_client, "https://example.com/login"
            )

    # Tests for perform_form_authentication method

    @pytest.mark.asyncio
    async def test_perform_form_authentication_success(
        self, web_auth, mock_http_client
    ):
        """Test successful form authentication"""
        # Mock responses
        login_response = MagicMock()
        login_response.status_code = 200
        mock_http_client.post.return_value = login_response

        credentials = {"username": "test_user", "password": "test_pass"}

        # Mock the token retrieval to avoid testing that part here
        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ):
            success, response = await web_auth.perform_form_authentication(
                mock_http_client, "https://example.com/login", credentials
            )

        assert success is True
        assert response == login_response
        assert web_auth._last_authentication_status is True

        # Verify the correct payload was sent
        _, kwargs = mock_http_client.post.call_args
        assert kwargs["data"]["authenticity_token"] == "test_token"
        assert kwargs["data"]["username"] == "test_user"
        assert kwargs["data"]["password"] == "test_pass"

    @pytest.mark.asyncio
    async def test_perform_form_authentication_with_provided_token(
        self, web_auth, mock_http_client
    ):
        """Test form authentication with pre-provided token"""
        # Mock response
        login_response = MagicMock()
        login_response.status_code = 200
        mock_http_client.post.return_value = login_response

        credentials = {"username": "test_user", "password": "test_pass"}

        # Call with provided token - this should skip token retrieval
        success, response = await web_auth.perform_form_authentication(
            mock_http_client,
            "https://example.com/login",
            credentials,
            token_value="provided_token",
        )

        assert success is True

        # Verify token retrieval was not called and provided token was used
        _, kwargs = mock_http_client.post.call_args
        assert kwargs["data"]["authenticity_token"] == "provided_token"

    @pytest.mark.asyncio
    async def test_perform_form_authentication_with_extra_params(
        self, web_auth, mock_http_client
    ):
        """Test form authentication with extra parameters"""
        # Mock response
        login_response = MagicMock()
        login_response.status_code = 200
        mock_http_client.post.return_value = login_response

        credentials = {"username": "test_user", "password": "test_pass"}
        extra_params = {"remember": "true", "redirect": "/dashboard"}

        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ):
            success, _ = await web_auth.perform_form_authentication(
                mock_http_client,
                "https://example.com/login",
                credentials,
                extra_params=extra_params,
            )

        assert success is True

        # Verify extra params were included
        _, kwargs = mock_http_client.post.call_args
        assert kwargs["data"]["remember"] == "true"
        assert kwargs["data"]["redirect"] == "/dashboard"

    @pytest.mark.asyncio
    async def test_perform_form_authentication_failure(
        self, web_auth, mock_http_client
    ):
        """Test form authentication failure"""
        # Mock failed response
        login_response = MagicMock()
        login_response.status_code = 401
        mock_http_client.post.return_value = login_response

        credentials = {"username": "wrong_user", "password": "wrong_pass"}

        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ):
            success, response = await web_auth.perform_form_authentication(
                mock_http_client, "https://example.com/login", credentials
            )

        assert success is False
        assert web_auth._last_authentication_status is False

    @pytest.mark.asyncio
    async def test_perform_form_authentication_error(self, web_auth, mock_http_client):
        """Test form authentication with error"""
        # Configure mock to raise an exception
        mock_http_client.post.side_effect = Exception("Network Error")

        credentials = {"username": "test_user", "password": "test_pass"}

        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ):
            with pytest.raises(Exception):
                await web_auth.perform_form_authentication(
                    mock_http_client, "https://example.com/login", credentials
                )

        assert web_auth._last_authentication_status is False

    # Tests for verify_authentication method

    @pytest.mark.asyncio
    async def test_verify_authentication_success(self, web_auth, mock_http_client):
        """Test successful authentication verification"""
        check_response = MagicMock()
        check_response.status_code = 200
        check_response.text = "<html><body>Welcome User!</body></html>"
        mock_http_client.get.return_value = check_response

        result = await web_auth.verify_authentication(
            mock_http_client, "https://example.com/dashboard"
        )

        assert result is True
        assert web_auth._last_authentication_status is True
        mock_http_client.get.assert_called_once_with("https://example.com/dashboard")

    @pytest.mark.asyncio
    async def test_verify_authentication_with_failure_strings(
        self, web_auth, mock_http_client
    ):
        """Test authentication verification with failure strings"""
        check_response = MagicMock()
        check_response.status_code = 200
        check_response.text = "<html><body>Login required</body></html>"
        mock_http_client.get.return_value = check_response

        result = await web_auth.verify_authentication(
            mock_http_client,
            "https://example.com/dashboard",
            failure_strings=["Login required", "Please sign in"],
        )

        assert result is False
        assert web_auth._last_authentication_status is False

    @pytest.mark.asyncio
    async def test_verify_authentication_non_200_status(
        self, web_auth, mock_http_client
    ):
        """Test authentication verification with non-200 status"""
        check_response = MagicMock()
        check_response.status_code = 302  # Redirect
        mock_http_client.get.return_value = check_response

        result = await web_auth.verify_authentication(
            mock_http_client, "https://example.com/dashboard"
        )

        assert result is False
        assert web_auth._last_authentication_status is False

    @pytest.mark.asyncio
    async def test_verify_authentication_error(self, web_auth, mock_http_client):
        """Test authentication verification with error"""
        # Configure mock to raise an exception
        mock_http_client.get.side_effect = Exception("Network Error")

        result = await web_auth.verify_authentication(
            mock_http_client, "https://example.com/dashboard"
        )

        assert result is False
        assert web_auth._last_authentication_status is False

    # Tests for is_authenticated property

    def test_is_authenticated_property(self, web_auth):
        """Test is_authenticated property reflects internal state"""
        # Initially should be False
        assert web_auth.is_authenticated is False

        # Set internal state and verify property
        web_auth._last_authentication_status = True
        assert web_auth.is_authenticated is True

        web_auth._last_authentication_status = False
        assert web_auth.is_authenticated is False

    # Tests for complete_authentication_flow method

    @pytest.mark.asyncio
    async def test_complete_authentication_flow_success(
        self, web_auth, mock_http_client
    ):
        """Test successful complete authentication flow"""
        credentials = {"username": "test_user", "password": "test_pass"}

        # Mock the component methods to isolate this test
        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ) as mock_get_token:
            with patch.object(
                web_auth,
                "perform_form_authentication",
                AsyncMock(return_value=(True, MagicMock())),
            ) as mock_login:

                result = await web_auth.complete_authentication_flow(
                    mock_http_client, "https://example.com/login", credentials
                )

                assert result is True
                mock_get_token.assert_called_once()
                mock_login.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_authentication_flow_with_verification(
        self, web_auth, mock_http_client
    ):
        """Test authentication flow with verification step"""
        credentials = {"username": "test_user", "password": "test_pass"}
        check_url = "https://example.com/dashboard"

        # Mock all component methods
        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ):
            with patch.object(
                web_auth,
                "perform_form_authentication",
                AsyncMock(return_value=(True, MagicMock())),
            ):
                with patch.object(
                    web_auth, "verify_authentication", AsyncMock(return_value=True)
                ) as mock_verify:

                    result = await web_auth.complete_authentication_flow(
                        mock_http_client,
                        "https://example.com/login",
                        credentials,
                        check_url=check_url,
                    )

                    assert result is True
                    mock_verify.assert_called_once_with(
                        http_client=mock_http_client,
                        check_url=check_url,
                        failure_strings=None,
                    )

    @pytest.mark.asyncio
    async def test_complete_authentication_flow_token_retrieval_fails(
        self, web_auth, mock_http_client
    ):
        """Test authentication flow when token retrieval fails"""
        credentials = {"username": "test_user", "password": "test_pass"}

        # Mock token retrieval to fail
        with patch.object(
            web_auth,
            "get_authenticity_token",
            AsyncMock(side_effect=ValueError("Token not found")),
        ):

            result = await web_auth.complete_authentication_flow(
                mock_http_client, "https://example.com/login", credentials
            )

            assert result is False
            assert web_auth._last_authentication_status is False

    @pytest.mark.asyncio
    async def test_complete_authentication_flow_login_fails(
        self, web_auth, mock_http_client
    ):
        """Test authentication flow when login fails"""
        credentials = {"username": "test_user", "password": "test_pass"}

        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ):
            with patch.object(
                web_auth,
                "perform_form_authentication",
                AsyncMock(return_value=(False, MagicMock())),
            ):

                result = await web_auth.complete_authentication_flow(
                    mock_http_client, "https://example.com/login", credentials
                )

                assert result is False

    @pytest.mark.asyncio
    async def test_complete_authentication_flow_verification_fails(
        self, web_auth, mock_http_client
    ):
        """Test authentication flow when verification fails"""
        credentials = {"username": "test_user", "password": "test_pass"}
        check_url = "https://example.com/dashboard"

        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ):
            with patch.object(
                web_auth,
                "perform_form_authentication",
                AsyncMock(return_value=(True, MagicMock())),
            ):
                with patch.object(
                    web_auth, "verify_authentication", AsyncMock(return_value=False)
                ):

                    result = await web_auth.complete_authentication_flow(
                        mock_http_client,
                        "https://example.com/login",
                        credentials,
                        check_url=check_url,
                    )

                    assert result is False

    @pytest.mark.asyncio
    async def test_complete_authentication_flow_with_extra_params(
        self, web_auth, mock_http_client
    ):
        """Test authentication flow with extra parameters"""
        credentials = {"username": "test_user", "password": "test_pass"}
        extra_params = {"remember": True}

        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ):
            with patch.object(
                web_auth,
                "perform_form_authentication",
                AsyncMock(return_value=(True, MagicMock())),
            ) as mock_login:

                await web_auth.complete_authentication_flow(
                    mock_http_client,
                    "https://example.com/login",
                    credentials,
                    extra_params=extra_params,
                )

                # Verify extra_params were passed to perform_form_authentication
                _, kwargs = mock_login.call_args
                assert kwargs["extra_params"] == extra_params

    @pytest.mark.asyncio
    async def test_complete_authentication_flow_with_failure_strings(
        self, web_auth, mock_http_client
    ):
        """Test authentication flow with failure strings for verification"""
        credentials = {"username": "test_user", "password": "test_pass"}
        check_url = "https://example.com/dashboard"
        failure_strings = ["Login required"]

        with patch.object(
            web_auth, "get_authenticity_token", AsyncMock(return_value="test_token")
        ):
            with patch.object(
                web_auth,
                "perform_form_authentication",
                AsyncMock(return_value=(True, MagicMock())),
            ):
                with patch.object(
                    web_auth, "verify_authentication", AsyncMock(return_value=True)
                ) as mock_verify:

                    await web_auth.complete_authentication_flow(
                        mock_http_client,
                        "https://example.com/login",
                        credentials,
                        check_url=check_url,
                        failure_strings=failure_strings,
                    )

                    # Verify failure_strings were passed to verify_authentication
                    _, kwargs = mock_verify.call_args
                    assert kwargs["failure_strings"] == failure_strings
