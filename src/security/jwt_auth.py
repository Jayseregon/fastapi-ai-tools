import logging
import time

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from src.configs.env_config import config
from src.models.user import User

logger = logging.getLogger(__name__)

# Defines the OAuth2 Password Bearer scheme for token authentication.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def validate_token_expiration(expiration_time: int, current_time: int) -> None:
    """
    Validates that the token has not expired.

    Args:
        expiration_time (int): The expiration timestamp from the JWT.
        current_time (int): The current timestamp.

    Raises:
        HTTPException: If the token has expired.
    """
    logger.debug(f"Check for expiration time: {expiration_time} vs {current_time}")

    if expiration_time < current_time:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )


def validate_required_claims(payload: dict) -> None:
    """
    Validates that the JWT payload contains all required claims.

    Args:
        payload (dict): The decoded JWT payload.

    Raises:
        HTTPException: If any required claim is missing.
    """
    logger.debug("Check for required claims")

    required_claims = ["id", "email", "name", "iss"]
    if not all(claim in payload for claim in required_claims):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims",
            headers={"WWW-Authenticate": "Bearer"},
        )


def validate_issuer(issuer: str, allowed_issuers: list) -> None:
    """
    Validates that the JWT issuer is in the list of allowed issuers.

    Args:
        issuer (str): The issuer from the JWT payload.
        allowed_issuers (list): A list of allowed issuer URLs.

    Raises:
        HTTPException: If the issuer is not allowed.
    """
    logger.debug(f"Check for issuer: {issuer} in {allowed_issuers}")

    if issuer not in allowed_issuers:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token issuer not allowed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def validate_token(token: str = Depends(oauth2_scheme)) -> User:
    """
    Validates the JWT token and returns a User object.

    Args:
        token (str, optional): The JWT token from the request header. Defaults to Depends(oauth2_scheme).

    Returns:
        User: A User object containing the validated information from the JWT.

    Raises:
        HTTPException: If the token is invalid or any validation fails.
    """
    try:
        current_time = int(time.time())
        payload = jwt.decode(
            token,
            config.SECRET_KEY,
            algorithms=["HS256"],
            issuer=config.get_allowed_issuers,
            options={
                "require": ["exp", "iat", "iss"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iss": True,
            },
        )

        validate_token_expiration(payload.get("exp"), current_time)
        validate_required_claims(payload)
        validate_issuer(payload.get("iss"), config.get_allowed_issuers)

        return User(
            id=payload.get("id"),
            email=payload.get("email"),
            name=payload.get("name"),
            issuer=payload.get("iss"),
            issued_at=payload.get("iat"),
            expires_at=payload.get("exp"),
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        )
