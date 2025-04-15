import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, cast

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile
from langchain.schema import Document

from src.configs.env_config import config
from src.models.documents_models import (
    AddDocumentsResponse,
    DocumentMetadata,
    StoreMetadata,
    UpdateDocumentsResponse,
    WebUrlRequest,
)
from src.security.rateLimiter.depends import RateLimiter
from src.services.cleaners import PdfDocumentCleaner, WebDocumentCleaner
from src.services.loaders.files import PdfLoader
from src.services.loaders.web import PublicLoader
from src.services.processors import DocumentsPreprocessing
from src.services.storages import BlobStorage
from src.services.vectorstore import ChromaStore

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/documents", tags=["Documents Pipeline"])


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

    doc_metadata_abstract: DocumentMetadata = cleaned_docs[0].metadata

    # Process the cleaned documents to create chunks and ids
    processor = DocumentsPreprocessing()
    chunks, ids = await processor(documents=cleaned_docs)

    return chunks, ids, original_filename, doc_metadata_abstract


async def _blob_storage_process_pdf_file(
    blob_name: str,
    temp_dir: str | Path,
) -> Tuple[List[Document], List[str], DocumentMetadata]:
    """
    Process a PDF file stored in Azure Blob Storage.

    Downloads the blob, loads and cleans the document, and processes it into chunks.

    Args:
        blob_name (str): The name of the blob in Azure Blob Storage.
        temp_dir (str | Path): The directory to save the downloaded blob.

    Returns:
        Tuple[List[Document], List[str], str, DocumentMetadata]:
            - chunks: List of processed Document objects.
            - ids: List of unique identifiers for each document chunk.
            - doc_metadata_abstract: Metadata extracted from the first document.
    """
    async with BlobStorage() as storage:
        temp_pdf_path = await storage.download_blob(
            blob_name=blob_name,
            temp_dir=temp_dir,
        )

    # Load document from pdf input file
    async with PdfLoader() as loader:
        raw_docs = await loader.load_document(temp_pdf_path)

    # Clean the loaded documents
    cleaner = PdfDocumentCleaner()
    cleaned_docs = await cleaner.clean_documents(documents=raw_docs)

    doc_metadata_abstract: DocumentMetadata = cleaned_docs[0].metadata

    # Process the cleaned documents to create chunks and ids
    processor = DocumentsPreprocessing()
    chunks, ids = await processor(documents=cleaned_docs)

    return chunks, ids, doc_metadata_abstract


async def _process_web_url(
    request: WebUrlRequest,
) -> Tuple[List[Document], List[str], str, DocumentMetadata]:
    """Process the web URL request.

    This function handles the complete web page processing workflow:
    1. Loads the web page content using PublicLoader
    2. Optionally loads images from the web page if requested
    3. Cleans the document content with WebDocumentCleaner
    4. Processes the document into chunks with DocumentsPreprocessing

    Args:
        request: WebUrlRequest containing the URL to process and image loading preferences

    Returns:
        A tuple containing:
        - chunks: List of processed Document objects ready for vectorization
        - ids: List of unique identifiers for each document chunk
        - original_url: The original URL that was processed
        - doc_metadata_abstract: Metadata extracted from the first document
    """
    raw_docs = []
    web_loader = PublicLoader()

    # Load document(s)
    if not request.with_images:
        # Load only the web page content as document
        doc = await web_loader.load_single_document(url=request.web_url)
        raw_docs.append(doc)
    else:
        # Load the web page content and images as separate documents
        docs = await web_loader.load_single_document_with_images(url=request.web_url)
        raw_docs.extend(docs)

        # Clean the loaded documents
    cleaner = WebDocumentCleaner()
    cleaned_docs = await cleaner.clean_documents(documents=raw_docs)

    doc_metadata_abstract: DocumentMetadata = cleaned_docs[0].metadata

    # Process the cleaned documents to create chunks and ids
    processor = DocumentsPreprocessing()
    chunks, ids = await processor(documents=cleaned_docs)

    return chunks, ids, request.web_url, doc_metadata_abstract


@router.post("/pdf/add", response_model=AddDocumentsResponse, status_code=200)
async def add_pdf_document(
    pdf_file: UploadFile,
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> AddDocumentsResponse:
    """Add a new PDF document to the vector store.

    This endpoint processes a PDF file and adds its contents to the vector database for
    future retrieval and querying. The document is chunked, cleaned, and indexed.

    Args:
        pdf_file: The PDF file to be uploaded and processed
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        AddDocumentsResponse: Object containing details about the operation, including:
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
            collection_name=config.COLLECTION_NAME,
        )

        return AddDocumentsResponse(
            status="success",
            filename=original_filename,
            store_metadata=StoreMetadata.model_validate(store.store_metadata),
            added_count=added_count,
            skipped_count=skipped_count,
            skipped_sources=skipped_sources,
            doc_sample_meta=doc_metadata_abstract,
        )

    except Exception as e:
        logger.error(f"Error loading pdf file: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error loading pdf file: {str(e)}")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post(
    "/pdf/add-from-azure", response_model=AddDocumentsResponse, status_code=200
)
async def add_pdf_document_from_azure(
    blob_name: str = Body(
        ..., embed=True, description="Blob name in Azure Blob Storage"
    ),
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> AddDocumentsResponse:
    """
    Add a new PDF document to the vector store from Azure Blob Storage.

    This endpoint downloads a PDF from Azure Blob Storage, processes it, and adds its contents
    to the vector database for future retrieval and querying.

    Args:
        blob_name (str): The name of the blob in Azure Blob Storage.
        rate (Optional[None]): Rate limiter dependency.

    Returns:
        AddDocumentsResponse: Details about the operation, including status, filename, store metadata,
        added/skipped counts, skipped sources, and sample document metadata.

    Raises:
        HTTPException: If the file cannot be processed or added to the vector store.
    """
    temp_dir = tempfile.mkdtemp()

    try:
        # "https://djangomaticstorage.blob.core.windows.net/next-client-storage/chatbot/4.1 XNS List of Preferred Plant Materials.pdf"

        # 4.1 XNS List of Preferred Plant Materials.pdf

        # Process the PDF file from Azure Blob Storage
        chunks, ids, doc_metadata_abstract = await _blob_storage_process_pdf_file(
            blob_name=blob_name,
            temp_dir=temp_dir,
        )

        # Add the documents to the vector store
        store = ChromaStore()
        added_count, skipped_count, skipped_sources = await store.add_documents(
            documents=chunks,
            ids=ids,
            collection_name=config.COLLECTION_NAME,
        )

        return AddDocumentsResponse(
            status="success",
            filename=blob_name,
            store_metadata=StoreMetadata.model_validate(store.store_metadata),
            added_count=added_count,
            skipped_count=skipped_count,
            skipped_sources=skipped_sources,
            doc_sample_meta=doc_metadata_abstract,
        )

    except Exception as e:
        logger.error(f"Error loading pdf from Azure: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Error loading pdf from Azure: {str(e)}"
        )

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post("/pdf/update", response_model=UpdateDocumentsResponse, status_code=200)
async def update_pdf_document(
    pdf_file: UploadFile,
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> UpdateDocumentsResponse:
    """Update an existing PDF document in the vector store.

    This endpoint processes a PDF file and updates its contents in the vector database.
    If the document (identified by its content) already exists, it will be replaced.
    The document is chunked, cleaned, and re-indexed.

    Args:
        pdf_file: The PDF file to be uploaded and processed for updating
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        UpdateDocumentsResponse: Object containing details about the update operation, including:
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
            collection_name=config.COLLECTION_NAME,
        )

        return UpdateDocumentsResponse(
            status="success",
            filename=original_filename,
            store_metadata=StoreMetadata.model_validate(store.store_metadata),
            added_count=added_count,
            docs_replaced=docs_replaced,
            sources_updated=sources_updated,
            doc_sample_meta=doc_metadata_abstract,
        )

    except Exception as e:
        logger.error(f"Error loading pdf file: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error loading pdf file: {str(e)}")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post("/web/add", response_model=AddDocumentsResponse, status_code=200)
async def add_web_document(
    request: WebUrlRequest,
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> AddDocumentsResponse:
    """Add a new web page to the vector store.

    This endpoint processes a web URL and adds its contents to the vector database for
    future retrieval and querying by RAG systems and AI agents. The web content is
    cleaned, chunked, and indexed. Images can optionally be included as separate
    documents to support multimodal AI processing.

    Args:
        request: WebUrlRequest containing the URL to process and whether to include images
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        AddDocumentsResponse: Object containing details about the operation, including:
            - status: Operation result status
            - filename: The URL of the processed web page
            - store_metadata: Information about the vector store
            - added_count: Number of chunks successfully added to the store
            - skipped_count: Number of chunks skipped (e.g., duplicates)
            - skipped_sources: List of sources that were skipped
            - doc_sample_meta: Sample metadata from the document

    Raises:
        HTTPException: If the web page cannot be processed or added to the vector store
    """
    try:
        chunks, ids, original_url, doc_metadata_abstract = await _process_web_url(
            request
        )

        # Add the documents to the vector store
        store = ChromaStore()
        added_count, skipped_count, skipped_sources = await store.add_documents(
            documents=chunks,
            ids=ids,
            collection_name=config.COLLECTION_NAME,
            is_web=True,
        )

        return AddDocumentsResponse(
            status="success",
            filename=original_url,
            store_metadata=StoreMetadata.model_validate(store.store_metadata),
            added_count=added_count,
            skipped_count=skipped_count,
            skipped_sources=skipped_sources,
            doc_sample_meta=doc_metadata_abstract,
        )

    except Exception as e:
        logger.error(f"Error loading web url: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error loading web url: {str(e)}")


@router.post("/web/update", response_model=UpdateDocumentsResponse, status_code=200)
async def update_web_document(
    request: WebUrlRequest,
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> UpdateDocumentsResponse:
    """Update an existing web page in the vector store.

    This endpoint processes a web URL and updates its contents in the vector database.
    If the content already exists, it will be replaced, ensuring AI agents have access
    to the most current version. This is particularly useful for keeping RAG systems
    up-to-date with changing web content.

    Args:
        request: WebUrlRequest containing the URL to process and whether to include images
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        UpdateDocumentsResponse: Object containing details about the update operation, including:
            - status: Operation result status
            - filename: The URL of the processed web page
            - store_metadata: Information about the vector store
            - added_count: Number of chunks successfully added to the store
            - docs_replaced: Number of document chunks that were replaced
            - sources_updated: List of document sources that were updated
            - doc_sample_meta: Sample metadata from the document

    Raises:
        HTTPException: If the web page cannot be processed or updated in the vector store
    """
    try:
        chunks, ids, original_url, doc_metadata_abstract = await _process_web_url(
            request
        )

        # Add the documents to the vector store
        store = ChromaStore()
        added_count, docs_replaced, sources_updated = await store.replace_documents(
            documents=chunks,
            ids=ids,
            collection_name=config.COLLECTION_NAME,
            is_web=True,
        )

        return UpdateDocumentsResponse(
            status="success",
            filename=original_url,
            store_metadata=StoreMetadata.model_validate(store.store_metadata),
            added_count=added_count,
            docs_replaced=docs_replaced,
            sources_updated=sources_updated,
            doc_sample_meta=doc_metadata_abstract,
        )

    except Exception as e:
        logger.error(f"Error loading web url: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error loading web url: {str(e)}")


@router.post("/setics/add", response_model=AddDocumentsResponse, status_code=200)
async def add_setics_document(
    json_file: UploadFile,
    is_image: Optional[bool] = False,
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> AddDocumentsResponse:
    try:
        logger.debug(f"json_file: {json_file}")
        logger.debug(f"type: {type(json_file)}")
        json_content = await json_file.read()
        json_data = json.loads(json_content.decode("utf-8"))

        logger.debug(f"json_data: {json_data}")
        logger.debug(f"type: {type(json_data)}")

        clean_docs = [
            Document(page_content=page["page_content"], metadata=page["metadata"])
            for page in json_data
        ]

        doc_metadata_abstract: DocumentMetadata = clean_docs[0].metadata

        if not is_image:
            processor = DocumentsPreprocessing()
            chunks, ids = await processor(documents=clean_docs)
        else:
            chunks = clean_docs
            ids = [doc.metadata["id"] for doc in clean_docs]

        store = ChromaStore()
        added_count, skipped_count, skipped_sources = await store.add_documents(
            documents=chunks,
            ids=ids,
            collection_name=config.COLLECTION_NAME,
            skip_existing=False,
            is_web=True,
        )

        original_filename = cast(str, json_file.filename or "unnamed_document.json")

        return AddDocumentsResponse(
            status="success",
            filename=original_filename,
            store_metadata=StoreMetadata.model_validate(store.store_metadata),
            added_count=added_count,
            skipped_count=skipped_count,
            skipped_sources=skipped_sources,
            doc_sample_meta=doc_metadata_abstract,
        )

    except Exception as e:
        logger.error(f"Error loading json data: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Error loading json data: {str(e)}"
        )
