import os
import shutil
import tempfile
from typing import cast

from fastapi import APIRouter, Depends, UploadFile

from src.models.documents import AddPDFResponse, StoreMetadata, UpdatePDFResponse
from src.security.rateLimiter.depends import RateLimiter
from src.services.cleaners import PdfDocumentCleaner
from src.services.loaders.files.pdf_loader import PdfLoader
from src.services.processors import DocumentsPreprocessing
from src.services.vectorstore import ChromaStore

router = APIRouter(prefix="/v1/documents", tags=["documents"])

COLLECTION_NAME = "demo_embed"


@router.post("/pdf/add", response_model=AddPDFResponse, status_code=200)
async def add_pdf_document(
    pdf_file: UploadFile,
    rate: None = Depends(RateLimiter(times=3, seconds=10)),
):
    """Demo route to load a PDF and return metadata from the first document."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
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

        # Process the cleaned documents to create chunks and ids
        processor = DocumentsPreprocessing()
        chunks, ids = await processor(documents=cleaned_docs)

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
        )

    finally:
        # Clean up the temporary directory and its contents
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post("/pdf/update", response_model=UpdatePDFResponse, status_code=200)
async def update_pdf_document(
    pdf_file: UploadFile,
    rate: None = Depends(RateLimiter(times=3, seconds=10)),
):
    """Demo route to load a PDF and return metadata from the first document."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
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

        # Process the cleaned documents to create chunks and ids
        processor = DocumentsPreprocessing()
        chunks, ids = await processor(documents=cleaned_docs)

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
        )
    finally:
        # Clean up the temporary directory and its contents
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
