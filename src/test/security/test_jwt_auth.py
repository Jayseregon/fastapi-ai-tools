import time

import pytest
from fastapi import HTTPException, status
from jose import jwt

from src.configs.env_config import config
from src.models.user import User
from src.security.jwt_auth import validate_token


# Helper to create a token with custom payload overrides.
def create_token(overrides: dict = {}):
    payload = {
        "id": "test-user-id",
        "email": "test@example.com",
        "name": "Test User",
        "iss": (
            config.get_allowed_issuers[0]
            if isinstance(config.get_allowed_issuers, list)
            else config.get_allowed_issuers
        ),
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    payload.update(overrides)
    return jwt.encode(payload, config.SECRET_KEY, algorithm="HS256")


@pytest.mark.asyncio
async def test_validate_token_success():
    token = create_token()
    user = await validate_token(token)
    assert isinstance(user, User)
    assert user.id == "test-user-id"
    assert user.email == "test@example.com"


@pytest.mark.asyncio
async def test_validate_token_expired():
    expired_token = create_token({"exp": int(time.time()) - 10})
    with pytest.raises(HTTPException) as exc_info:
        await validate_token(expired_token)
    # Expect the generic JWTError conversion message.
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "could not validate" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_validate_token_missing_claims():
    # Remove required claim "email" by setting it to None.
    token = create_token({"email": None})
    with pytest.raises(HTTPException) as exc_info:
        await validate_token(token)
    # Expect internal error due to missing claim value.
    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "internal server error" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_validate_token_invalid_signature():
    token = create_token()
    # Alter the secret key to force invalid signature.
    wrong_key = "wrong_secret"
    # Re-encode with wrong key.
    bad_token = jwt.encode(
        jwt.decode(token, config.SECRET_KEY, algorithms=["HS256"]),
        wrong_key,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc_info:
        await validate_token(bad_token)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "could not validate" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_validate_token_invalid_issuer(monkeypatch):
    # Create a token with an unapproved issuer.
    token = create_token({"iss": "unapproved_issuer"})
    # Override the get_allowed_issuers property on the config's class.
    monkeypatch.setattr(
        config.__class__,
        "get_allowed_issuers",
        property(lambda self: ["approved_issuer"]),
    )
    with pytest.raises(HTTPException) as exc_info:
        await validate_token(token)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    # Update expected error message substring.
    assert "could not validate" in exc_info.value.detail.lower()
