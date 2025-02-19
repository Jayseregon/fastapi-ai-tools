import logging
from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.exception_handlers import http_exception_handler

from src.logging_conf import configure_logging
from src.routes.embeddingRouter import router as embedding_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CorrelationIdMiddleware)


@app.get("/")
async def read_root():
    return {"greatings": "Welcome to the keyword embeddings API!"}


app.include_router(embedding_router)


@app.exception_handler(HTTPException)
async def http_exception_handle_logging(request, exc):
    logger.error(f"HTTPException: {exc.status_code} {exc.detail}")
    return await http_exception_handler(request, exc)
