import os
import time
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock

import jwt
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

os.environ["ENV_STATE"] = "test"

from src.configs.env_config import config  # noqa: E402
from src.main import app  # noqa: E402


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def client() -> Generator:
    yield TestClient(app)


@pytest_asyncio.fixture
async def async_client(client) -> AsyncGenerator[AsyncClient, None]:
    """Create async client for testing"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=client.base_url) as ac:
        yield ac


@pytest.fixture()
def auth_token() -> str:
    """Create a valid JWT token for testing."""
    payload = {
        "id": "test-user-id",
        "email": "test@example.com",
        "name": "Test User",
        "iss": "testissuer",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return jwt.encode(payload, config.SECRET_KEY, algorithm="HS256")


@pytest.fixture()
def auth_headers(auth_token: str) -> dict:
    """Create authorization headers for testing."""
    return {"Authorization": f"Bearer {auth_token}"}


# Add a dummy identifier for rate limiter testing
async def dummy_identifier(request):
    return "test_identifier"


@pytest_asyncio.fixture(autouse=True)
async def mock_rate_limiter():
    """Initialize rate limiter with a fake Redis backend for all tests."""
    fake_redis = AsyncMock()
    fake_redis.script_load.return_value = "dummy_sha"
    fake_redis.evalsha.return_value = 0
    from src.security.rateLimiter import FastAPILimiter

    await FastAPILimiter.init(fake_redis, identifier=dummy_identifier)
    yield
    if FastAPILimiter.redis:
        await FastAPILimiter.close()
