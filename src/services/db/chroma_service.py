import logging

import chromadb
from chromadb.config import Settings

from src.configs.env_config import config

logger = logging.getLogger(__name__)


class ChromaService:
    """Service for interacting with ChromaDB."""

    def __init__(self):
        """Initialize the ChromaDB client."""
        self.client = None

    def __call__(self):
        """Get or create a ChromaDB client."""
        if self.client:
            return self.client

        try:
            logger.debug(
                f"Connecting to ChromaDB at {config.CHROMADB_HOST}:{config.CHROMADB_PORT}"
            )

            self.client = chromadb.HttpClient(
                host=config.CHROMADB_HOST,
                port=config.CHROMADB_PORT,
                ssl=False,
                settings=Settings(
                    anonymized_telemetry=False,
                    chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                    chroma_client_auth_credentials=config.CHROMA_CLIENT_AUTH_CREDENTIALS,
                ),
            )
            logger.info("ChromaDB client initialized")
            return self.client
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            self.client = None
            raise

    def close(self):
        """Clean up any resources if needed."""
        if self.client:
            self.client = None


# Create a singleton instance
chroma_service = ChromaService()
