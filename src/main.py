import logging
from contextlib import asynccontextmanager

import redis.asyncio as redis
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import OAuth2PasswordBearer

from src.configs.env_config import config
from src.configs.log_config import configure_logging
from src.models.user import User
from src.routes.embedding import router as embedding_router
from src.security.jwt_auth import validate_token
from src.security.rateLimiter import FastAPILimiter
from src.security.rateLimiter.depends import RateLimiter

# Initialize logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging
    configure_logging()
    # Initialize Redis client
    redis_client = redis.from_url(config.REDIS_URL)
    if not redis_client:
        raise Exception("Please configure Redis client for rate limiting")
    # Initialize FastAPILimiter
    await FastAPILimiter.init(redis_client)
    yield
    await FastAPILimiter.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.get_allowed_hosts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if config.ENV_STATE == "prod":
    app.add_middleware(HTTPSRedirectMiddleware)

app.add_middleware(CorrelationIdMiddleware)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/")
async def read_root(rate: None = Depends(RateLimiter(times=3, seconds=10))):
    logger.debug({"Origins": config.get_allowed_hosts})
    return {"greatings": "Welcome to the keyword embeddings API!"}


@app.get("/users/me")
async def read_users_me(
    current_user: User = Depends(validate_token),
    rate: None = Depends(RateLimiter(times=3, seconds=10)),
):
    return current_user


app.include_router(embedding_router)


@app.exception_handler(HTTPException)
async def http_exception_handle_logging(request, exc):
    logger.error(f"HTTPException: {exc.status_code} {exc.detail}")
    return await http_exception_handler(request, exc)
