import logging
from contextlib import asynccontextmanager

import redis.asyncio as redis
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer

from src.configs.env_config import config
from src.configs.log_config import configure_logging
from src.models.user import User
from src.routes.embedding import router as embedding_router
from src.security.jwt_auth import validate_token
from src.security.rateLimiter import FastAPILimiter
from src.security.rateLimiter.depends import RateLimiter
from src.services.chromaData import router as chroma_router
from src.services.db import chroma_service, neo4j_service
from src.services.graphData import router as graph_router

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
    await redis_client.close()
    neo4j_service.close()
    chroma_service.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.get_allowed_hosts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(CorrelationIdMiddleware)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/", response_class=HTMLResponse)
async def read_root(rate: None = Depends(RateLimiter(times=3, seconds=10))):
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Djangomatic AI Toolbox AI Home Page</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f0f0f0; text-align: center; padding: 50px; }
            .container { background-color: white; padding: 20px; border-radius: 10px; display: inline-block; }
            .notice { border: 1px solid #ccc; padding: 20px; max-width: 500px; margin: 20px auto; text-align: justify; }
            h1 { color: #333; }
            p { color: #666; }
            a { text-decoration: none; color: #007acc; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to Djangomatic AI Toolbox</h1>
            <p>A robust FastAPI service for AI integration and Retrieval Augmented Generation workflows.</p>
            <div class="notice">
                <p><strong>Notice</strong></p>
                <p>
                    API usage is restricted to organization members only. Access is granted exclusively via the
                    Djangomatic Pro platform using SSO authentication for chatbot and AI-agent services.
                    Please authenticate via <a href="https://djangomatic-pro.azurewebsites.net">Djangomatic Pro</a>.
                </p>
            </div>
            <a href="/docs">API Documentation</a>
        </div>
    </body>
    </html>
    """


@app.get("/users/me")
async def read_users_me(
    current_user: User = Depends(validate_token),
    rate: None = Depends(RateLimiter(times=3, seconds=10)),
):
    return current_user


app.include_router(embedding_router)
app.include_router(graph_router)
app.include_router(chroma_router)


@app.exception_handler(HTTPException)
async def http_exception_handle_logging(request, exc):
    logger.error(f"HTTPException: {exc.status_code} {exc.detail}")
    return await http_exception_handler(request, exc)
