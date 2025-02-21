import os
import time
from typing import AsyncGenerator, Generator

import jwt
import pytest
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


@pytest.fixture()
async def async_client(client) -> AsyncGenerator:
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
