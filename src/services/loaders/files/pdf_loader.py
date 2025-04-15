import logging
from pathlib import Path
from typing import Optional, Self

import fitz
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers.images import (
    LLMImageBlobParser,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.configs.env_config import config
from src.services.loaders.files.base_document_loader import BaseDocumentLoader
from src.services.utils import documents_to_json, json_to_documents

logger = logging.getLogger(__name__)


class PdfLoader(BaseDocumentLoader):
    """Service for loading and processing PDF documents."""

    def __init__(self, llm_model: Optional[BaseChatModel] = None):
        super().__init__()
        self._default_model = llm_model or ChatOpenAI(
            model="gpt-4o-mini", api_key=config.OPENAI_API_KEY
        )

    async def initialize(
        self, llm_model: Optional[BaseChatModel] = None, **kwargs
    ) -> Self:
        self._llm_model = llm_model or self._default_model
        self._initialized = True
        logger.debug(
            f"Initialized PDF loader with model: {self._llm_model.__class__.__name__}"
        )
        return self

    async def load_document(
        self,
        file_path: str | Path,
    ) -> list[Document]:
        if not self._initialized:
            logger.debug("PDF loader not initialized, initializing now")
            await self.initialize()

        if isinstance(file_path, str):
            file_path = Path(file_path)

        logger.debug("Attempting to load PDF document")

        if not await self._is_valid_pdf(file_path):
            logger.debug(f"PDF validation failed for: {file_path}")
            raise ValueError(f"Invalid or inaccessible PDF file: {file_path}")

        logger.debug("Creating PyMuPDFLoader instance")
        loader = PyMuPDFLoader(
            file_path=file_path.as_posix(),
            images_inner_format="markdown-img",
            extract_tables="markdown",
            extract_images=True,
            images_parser=LLMImageBlobParser(model=self._llm_model),
        )

        logger.debug("Starting async PDF loading")
        documents = await loader.aload()

        # Add document type to metadata
        for doc in documents:
            doc.metadata["document_type"] = "pdf"

        logger.debug(f"Successfully loaded PDF with {len(documents)} document chunks")
        return documents

    async def documents_to_json(
        self, documents: list[Document], filename: str | Path
    ) -> None:
        logger.debug(f"Serializing {len(documents)} documents to JSON file: {filename}")
        documents_to_json(documents, filename)
        logger.debug(f"Successfully saved documents to JSON: {filename}")

    async def json_to_documents(self, filename: str | Path) -> list[Document]:
        logger.debug(f"Loading documents from JSON file: {filename}")
        documents = json_to_documents(filename)
        logger.debug(
            f"Successfully loaded {len(documents)} documents from JSON: {filename}"
        )
        return documents

    async def _is_valid_pdf(self, file_path: str | Path) -> bool:
        """
        Verify if a file is a valid PDF document.

        Args:
            file_path: Path to the file to check

        Returns:
            Boolean indicating if the file is a valid PDF
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        logger.debug("Validating PDF file")

        # Check if file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        # Check file extension (basic check)
        if file_path.suffix.lower() != ".pdf":
            logger.warning("File does not have PDF extension")
            logger.debug("Continuing validation despite incorrect extension")
            # Continue checking - some PDFs might have wrong extension

        try:
            # Check PDF signature (faster check)
            logger.debug("Checking PDF signature")
            with open(file_path, "rb") as f:
                header = f.read(1024)
                if not header.startswith(b"%PDF-"):
                    logger.error(f"File does not have PDF signature: {file_path}")
                    return False

            logger.debug(
                "PDF signature valid, performing deeper validation with PyMuPDF"
            )
            # Deeper validation using PyMuPDF
            doc = fitz.open(file_path)
            # Basic structure validation
            logger.debug(f"PDF opened, page count: {doc.page_count}")
            if doc.page_count == 0:
                logger.error(f"PDF has no pages: {file_path}")
                doc.close()
                return False

            # Check if document is encrypted/password-protected
            if doc.is_encrypted:
                logger.error(f"PDF is encrypted and requires a password: {file_path}")
                doc.close()
                return False

            # Validate document structure
            try:
                # Try accessing metadata to verify structure
                logger.debug("Checking PDF metadata")
                _ = doc.metadata
                # Try accessing first page to verify readability
                logger.debug("Checking first page accessibility")
                _ = doc[0]
                doc.close()
                logger.debug("PDF validation successful")
                return True
            except Exception as e:
                logger.error(f"PDF structure validation failed: {str(e)}")
                doc.close()
                return False

        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False


# Factory function for global access
async def create_pdf_loader() -> PdfLoader:
    """Create and initialize a PDF loader instance."""
    loader = PdfLoader()
    await loader.initialize()
    return loader
