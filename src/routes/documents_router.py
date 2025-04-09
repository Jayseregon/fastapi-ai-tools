import os
import shutil
import tempfile
from typing import List, Optional, Tuple, cast

from fastapi import APIRouter, Depends, UploadFile
from langchain.schema import Document

from src.models.documents_models import (
    AddPDFResponse,
    DocumentMetadata,
    StoreMetadata,
    UpdatePDFResponse,
)
from src.security.rateLimiter.depends import RateLimiter
from src.services.cleaners import PdfDocumentCleaner
from src.services.loaders.files import PdfLoader
from src.services.processors import DocumentsPreprocessing
from src.services.vectorstore import ChromaStore

router = APIRouter(prefix="/v1/documents", tags=["documents"])

COLLECTION_NAME = "pdf_documents"


async def _process_pdf_file(
    pdf_file: UploadFile, temp_dir: str
) -> Tuple[List[Document], List[str], str, DocumentMetadata]:
    """Process the uploaded PDF file.

    This function handles the complete PDF processing workflow:
    1. Saves the uploaded file to a temporary directory
    2. Loads the document content using PdfLoader
    3. Cleans the document content with PdfDocumentCleaner
    4. Processes the document into chunks with DocumentsPreprocessing

    Args:
        pdf_file: The uploaded PDF file object
        temp_dir: Directory path where the file will be temporarily stored

    Returns:
        A tuple containing:
        - chunks: List of processed Document objects ready for vectorization
        - ids: List of unique identifiers for each document chunk
        - original_filename: The original name of the uploaded file
        - doc_metadata_abstract: Metadata extracted from the first document
    """

    # Use the original filename with explicit type cast to satisfy mypy
    original_filename = cast(str, pdf_file.filename or "unnamed_document.pdf")
    temp_path = os.path.join(temp_dir, original_filename)

    # Save the uploaded file with its original name
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)

    # Load document from pdf input file
    async with PdfLoader() as loader:
        raw_docs = await loader.load_document(temp_path)

    # Clean the loaded documents
    cleaner = PdfDocumentCleaner()
    cleaned_docs = await cleaner.clean_documents(documents=raw_docs)

    # TODO : remove after testing
    doc_metadata_abstract: DocumentMetadata = cleaned_docs[0].metadata

    # Process the cleaned documents to create chunks and ids
    processor = DocumentsPreprocessing()
    chunks, ids = await processor(documents=cleaned_docs)

    return chunks, ids, original_filename, doc_metadata_abstract


@router.post("/pdf/add", response_model=AddPDFResponse, status_code=200)
async def add_pdf_document(
    pdf_file: UploadFile,
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> AddPDFResponse:
    """Add a new PDF document to the vector store.

    This endpoint processes a PDF file and adds its contents to the vector database for
    future retrieval and querying. The document is chunked, cleaned, and indexed.

    Args:
        pdf_file: The PDF file to be uploaded and processed
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        AddPDFResponse: Object containing details about the operation, including:
            - status: Operation result status
            - filename: Original filename of the processed document
            - store_metadata: Information about the vector store
            - added_count: Number of chunks successfully added to the store
            - skipped_count: Number of chunks skipped (e.g., duplicates)
            - skipped_sources: List of sources that were skipped
            - doc_sample_meta: Sample metadata from the document

    Raises:
        HTTPException: If the file cannot be processed or added to the vector store
    """
    temp_dir = tempfile.mkdtemp()

    try:
        # Process the uploaded PDF file
        chunks, ids, original_filename, doc_metadata_abstract = await _process_pdf_file(
            pdf_file=pdf_file,
            temp_dir=temp_dir,
        )

        # Add the documents to the vector store
        store = ChromaStore()
        added_count, skipped_count, skipped_sources = await store.add_documents(
            documents=chunks,
            ids=ids,
            collection_name=COLLECTION_NAME,
        )

        return AddPDFResponse(
            status="success",
            filename=original_filename,
            store_metadata=StoreMetadata.model_validate(store.store_metadata),
            added_count=added_count,
            skipped_count=skipped_count,
            skipped_sources=skipped_sources,
            doc_sample_meta=doc_metadata_abstract,
        )

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post("/pdf/update", response_model=UpdatePDFResponse, status_code=200)
async def update_pdf_document(
    pdf_file: UploadFile,
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> UpdatePDFResponse:
    """Update an existing PDF document in the vector store.

    This endpoint processes a PDF file and updates its contents in the vector database.
    If the document (identified by its content) already exists, it will be replaced.
    The document is chunked, cleaned, and re-indexed.

    Args:
        pdf_file: The PDF file to be uploaded and processed for updating
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        UpdatePDFResponse: Object containing details about the update operation, including:
            - status: Operation result status
            - filename: Original filename of the processed document
            - store_metadata: Information about the vector store
            - added_count: Number of chunks successfully added to the store
            - docs_replaced: Number of document chunks that were replaced
            - sources_updated: List of document sources that were updated
            - doc_sample_meta: Sample metadata from the document

    Raises:
        HTTPException: If the file cannot be processed or updated in the vector store
    """
    temp_dir = tempfile.mkdtemp()

    try:
        # Process the uploaded PDF file
        chunks, ids, original_filename, doc_metadata_abstract = await _process_pdf_file(
            pdf_file=pdf_file,
            temp_dir=temp_dir,
        )

        # Add the documents to the vector store
        store = ChromaStore()
        added_count, docs_replaced, sources_updated = await store.replace_documents(
            documents=chunks,
            ids=ids,
            collection_name=COLLECTION_NAME,
        )

        return UpdatePDFResponse(
            status="success",
            filename=original_filename,
            store_metadata=StoreMetadata.model_validate(store.store_metadata),
            added_count=added_count,
            docs_replaced=docs_replaced,
            sources_updated=sources_updated,
            doc_sample_meta=doc_metadata_abstract,
        )
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
