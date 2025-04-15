from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.schema import Document

from src.configs.env_config import config
from src.models.documents_models import WebUrlRequest
from src.routes.documents_router import _process_web_url, router


@pytest.fixture
def app():
    """Create a FastAPI app with the documents router for testing"""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def test_client(app):
    """Create a test client for the app"""
    return TestClient(app)


@pytest.fixture
def mock_pdf_file():
    """Create a mock PDF file for testing"""
    pdf_mock = Mock()
    pdf_mock.filename = "test_document.pdf"
    pdf_mock.file = Mock()
    return pdf_mock


@pytest.fixture
def sample_document():
    """Create a sample document for testing"""
    return Document(
        page_content="This is a test document",
        metadata={
            "source": "test_document.pdf",
            "title": "Test Document",
            "author": "Test Author",
            "pages": 5,
        },
    )


@pytest.fixture
def sample_chunks():
    """Create sample document chunks"""
    return [
        Document(
            page_content="Chunk 1 content",
            metadata={
                "source": "test_document.pdf",
                "page": 1,
                "chunk": 0,
            },
        ),
        Document(
            page_content="Chunk 2 content",
            metadata={
                "source": "test_document.pdf",
                "page": 2,
                "chunk": 1,
            },
        ),
    ]


@pytest.fixture
def sample_chunk_ids():
    """Create sample document chunk IDs"""
    return ["test_document-0-12345678", "test_document-1-87654321"]


@pytest.fixture
def sample_web_document():
    """Create a sample web document for testing"""
    return Document(
        page_content="This is a test web page content",
        metadata={
            "source": "https://example.com/page",
            "url": "https://example.com/page",
            "title": "Test Web Page",
            "description": "A web page for testing",
        },
    )


@pytest.fixture
def sample_web_request():
    """Create a sample web URL request"""
    return WebUrlRequest(web_url="https://example.com/page", with_images=False)


@pytest.fixture
def sample_web_request_with_images():
    """Create a sample web URL request with images"""
    return WebUrlRequest(web_url="https://example.com/page", with_images=True)


class TestDocumentsRouter:
    @pytest.mark.asyncio
    @patch("src.routes.documents_router.BlobStorage")
    @patch("src.routes.documents_router.PdfLoader")
    @patch("src.routes.documents_router.PdfDocumentCleaner")
    @patch("src.routes.documents_router.DocumentsPreprocessing")
    async def test_blob_storage_process_pdf_file(
        self,
        mock_processor_class,
        mock_cleaner_class,
        mock_loader_class,
        mock_blob_storage_class,
        mock_pdf_file,
        sample_document,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test the internal _blob_storage_process_pdf_file function"""
        from src.routes.documents_router import _blob_storage_process_pdf_file

        # Setup BlobStorage mock
        mock_blob_storage = AsyncMock()
        mock_blob_storage.__aenter__.return_value = mock_blob_storage
        mock_blob_storage.download_blob.return_value = "/tmp/test_dir/test_document.pdf"
        mock_blob_storage_class.return_value = mock_blob_storage

        # Setup PDF loader mock
        mock_loader = AsyncMock()
        mock_loader.load_document.return_value = [sample_document]
        mock_loader_class.return_value.__aenter__.return_value = mock_loader

        # Setup cleaner mock
        mock_cleaner = AsyncMock()
        mock_cleaner.clean_documents.return_value = [sample_document]
        mock_cleaner_class.return_value = mock_cleaner

        # Setup processor mock
        mock_processor = AsyncMock()
        mock_processor.return_value = (sample_chunks, sample_chunk_ids)
        mock_processor_class.return_value = mock_processor

        # Call function
        result = await _blob_storage_process_pdf_file(
            "test_document.pdf", "/tmp/test_dir"
        )

        # Verify results
        chunks, ids, metadata = result
        assert chunks == sample_chunks
        assert ids == sample_chunk_ids
        assert metadata == sample_document.metadata

        # Verify BlobStorage download called
        mock_blob_storage.download_blob.assert_called_once_with(
            blob_name="test_document.pdf",
            temp_dir="/tmp/test_dir",
        )

        # Verify service calls
        mock_loader.load_document.assert_called_once_with(
            "/tmp/test_dir/test_document.pdf"
        )
        mock_cleaner.clean_documents.assert_called_once_with(
            documents=[sample_document]
        )
        mock_processor.assert_called_once_with(documents=[sample_document])

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._blob_storage_process_pdf_file")
    @patch("src.routes.documents_router.ChromaStore")
    @patch("src.routes.documents_router.tempfile.mkdtemp")
    @patch("src.routes.documents_router.os.path.exists")
    @patch("src.routes.documents_router.shutil.rmtree")
    async def test_add_pdf_document_success(
        self,
        mock_rmtree,
        mock_exists,
        mock_mkdtemp,
        mock_chroma_store_class,
        mock_blob_storage_process_pdf,
        test_client,
        mock_pdf_file,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test successful PDF document addition"""
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True
        mock_blob_storage_process_pdf.return_value = (
            sample_chunks,
            sample_chunk_ids,
            {"source": "test_document.pdf", "title": "Test Document"},
        )

        mock_store = AsyncMock()
        mock_store.add_documents.return_value = (2, 0, [])
        mock_store.store_metadata = {
            "nb_collections": 1,
            "details": {"pdf_documents": {"count": 2}},
        }
        mock_chroma_store_class.return_value = mock_store

        with patch("fastapi.Depends", return_value=None):
            response = test_client.post(
                "/v1/documents/pdf/add",
                json={"blob_name": "test_document.pdf"},
            )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["filename"] == "test_document.pdf"
        assert response_data["added_count"] == 2
        assert response_data["skipped_count"] == 0

        mock_store.add_documents.assert_called_once_with(
            documents=sample_chunks,
            ids=sample_chunk_ids,
            collection_name=config.COLLECTION_NAME,
        )
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._blob_storage_process_pdf_file")
    @patch("src.routes.documents_router.ChromaStore")
    @patch("src.routes.documents_router.tempfile.mkdtemp")
    @patch("src.routes.documents_router.os.path.exists")
    @patch("src.routes.documents_router.shutil.rmtree")
    async def test_update_pdf_document_success(
        self,
        mock_rmtree,
        mock_exists,
        mock_mkdtemp,
        mock_chroma_store_class,
        mock_blob_storage_process_pdf,
        test_client,
        mock_pdf_file,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test successful PDF document update"""
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True
        mock_blob_storage_process_pdf.return_value = (
            sample_chunks,
            sample_chunk_ids,
            {"source": "test_document.pdf", "title": "Test Document"},
        )

        mock_store = AsyncMock()
        mock_store.replace_documents.return_value = (2, 3, 1)
        mock_store.store_metadata = {
            "nb_collections": 1,
            "details": {"pdf_documents": {"count": 2}},
        }
        mock_chroma_store_class.return_value = mock_store

        with patch("fastapi.Depends", return_value=None):
            response = test_client.post(
                "/v1/documents/pdf/update",
                json={"blob_name": "test_document.pdf"},
            )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["filename"] == "test_document.pdf"
        assert response_data["added_count"] == 2
        assert response_data["docs_replaced"] == 3
        assert response_data["sources_updated"] == 1

        mock_store.replace_documents.assert_called_once_with(
            documents=sample_chunks,
            ids=sample_chunk_ids,
            collection_name=config.COLLECTION_NAME,
        )
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._blob_storage_process_pdf_file")
    @patch("src.routes.documents_router.ChromaStore")
    @patch("src.routes.documents_router.tempfile.mkdtemp")
    @patch("src.routes.documents_router.os.path.exists")
    @patch("src.routes.documents_router.shutil.rmtree")
    async def test_add_pdf_document_error_handling(
        self,
        mock_rmtree,
        mock_exists,
        mock_mkdtemp,
        mock_chroma_store_class,
        mock_blob_storage_process_pdf,
        test_client,
    ):
        """Test error handling in add_pdf_document"""
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True

        mock_blob_storage_process_pdf.side_effect = Exception("PDF processing failed")

        with patch("fastapi.Depends", return_value=None):
            response = test_client.post(
                "/v1/documents/pdf/add",
                json={"blob_name": "test_document.pdf"},
            )
        assert response.status_code == 503
        assert "PDF processing failed" in response.json()["detail"]
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._blob_storage_process_pdf_file")
    @patch("src.routes.documents_router.ChromaStore")
    @patch("src.routes.documents_router.tempfile.mkdtemp")
    @patch("src.routes.documents_router.os.path.exists")
    @patch("src.routes.documents_router.shutil.rmtree")
    async def test_update_pdf_document_error_handling(
        self,
        mock_rmtree,
        mock_exists,
        mock_mkdtemp,
        mock_chroma_store_class,
        mock_blob_storage_process_pdf,
        test_client,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test error handling in update_pdf_document"""
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True
        mock_blob_storage_process_pdf.return_value = (
            sample_chunks,
            sample_chunk_ids,
            {"source": "test_document.pdf", "title": "Test Document"},
        )

        mock_store = AsyncMock()
        mock_store.replace_documents.side_effect = Exception("Vector store error")
        mock_chroma_store_class.return_value = mock_store

        with patch("fastapi.Depends", return_value=None):
            response = test_client.post(
                "/v1/documents/pdf/update",
                json={"blob_name": "test_document.pdf"},
            )
        assert response.status_code == 503
        assert "Vector store error" in response.json()["detail"]
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._blob_storage_process_pdf_file")
    @patch("src.routes.documents_router.ChromaStore")
    @patch("src.routes.documents_router.tempfile.mkdtemp")
    @patch("src.routes.documents_router.os.path.exists")
    @patch("src.routes.documents_router.shutil.rmtree")
    async def test_add_pdf_document_partially_processed(
        self,
        mock_rmtree,
        mock_exists,
        mock_mkdtemp,
        mock_chroma_store_class,
        mock_blob_storage_process_pdf,
        test_client,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test PDF document addition with some chunks skipped"""
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True
        mock_blob_storage_process_pdf.return_value = (
            sample_chunks,
            sample_chunk_ids,
            {"source": "test_document.pdf", "title": "Test Document"},
        )

        mock_store = AsyncMock()
        mock_store.add_documents.return_value = (1, 1, ["other_document.pdf"])
        mock_store.store_metadata = {
            "nb_collections": 1,
            "details": {"pdf_documents": {"count": 1}},
        }
        mock_chroma_store_class.return_value = mock_store

        with patch("fastapi.Depends", return_value=None):
            response = test_client.post(
                "/v1/documents/pdf/add",
                json={"blob_name": "test_document.pdf"},
            )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["added_count"] == 1
        assert response_data["skipped_count"] == 1
        assert response_data["skipped_sources"] == ["other_document.pdf"]

    @pytest.mark.asyncio
    @patch("src.routes.documents_router.PublicLoader")
    @patch("src.routes.documents_router.WebDocumentCleaner")
    @patch("src.routes.documents_router.DocumentsPreprocessing")
    async def test_process_web_url(
        self,
        mock_processor_class,
        mock_cleaner_class,
        mock_loader_class,
        sample_web_document,
        sample_chunks,
        sample_chunk_ids,
        sample_web_request,
    ):
        """Test the internal _process_web_url function"""
        # Setup loader mock
        mock_loader = AsyncMock()
        mock_loader.load_single_document.return_value = sample_web_document
        mock_loader.load_single_document_with_images.return_value = [
            sample_web_document
        ]
        mock_loader_class.return_value = mock_loader

        # Setup cleaner mock
        mock_cleaner = AsyncMock()
        mock_cleaner.clean_documents.return_value = [sample_web_document]
        mock_cleaner_class.return_value = mock_cleaner

        # Setup processor mock
        mock_processor = AsyncMock()
        mock_processor.return_value = (sample_chunks, sample_chunk_ids)
        mock_processor_class.return_value = mock_processor

        # Call function
        result = await _process_web_url(sample_web_request)

        # Verify results
        chunks, ids, url, metadata = result
        assert chunks == sample_chunks
        assert ids == sample_chunk_ids
        assert url == "https://example.com/page"
        assert metadata == sample_web_document.metadata

        # Verify service calls
        mock_loader.load_single_document.assert_called_once_with(
            url="https://example.com/page"
        )
        mock_cleaner.clean_documents.assert_called_once_with(
            documents=[sample_web_document]
        )
        mock_processor.assert_called_once_with(documents=[sample_web_document])

        # Verify load_single_document_with_images was not called
        mock_loader.load_single_document_with_images.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.routes.documents_router.PublicLoader")
    @patch("src.routes.documents_router.WebDocumentCleaner")
    @patch("src.routes.documents_router.DocumentsPreprocessing")
    async def test_process_web_url_with_images(
        self,
        mock_processor_class,
        mock_cleaner_class,
        mock_loader_class,
        sample_web_document,
        sample_chunks,
        sample_chunk_ids,
        sample_web_request_with_images,
    ):
        """Test the internal _process_web_url function with images enabled"""
        # Setup loader mock for multiple documents (page + images)
        mock_loader = AsyncMock()
        mock_loader.load_single_document.return_value = sample_web_document
        mock_loader.load_single_document_with_images.return_value = [
            sample_web_document,
            Document(
                page_content="Image description",
                metadata={"source": "https://example.com/image1.jpg", "type": "image"},
            ),
        ]
        mock_loader_class.return_value = mock_loader

        # Setup cleaner mock
        mock_cleaner = AsyncMock()
        mock_cleaner.clean_documents.return_value = [
            sample_web_document,
            Document(
                page_content="Cleaned image description",
                metadata={"source": "https://example.com/image1.jpg", "type": "image"},
            ),
        ]
        mock_cleaner_class.return_value = mock_cleaner

        # Setup processor mock
        mock_processor = AsyncMock()
        mock_processor.return_value = (sample_chunks, sample_chunk_ids)
        mock_processor_class.return_value = mock_processor

        # Call function
        result = await _process_web_url(sample_web_request_with_images)

        # Verify results
        chunks, ids, url, metadata = result
        assert chunks == sample_chunks
        assert ids == sample_chunk_ids
        assert url == "https://example.com/page"
        assert metadata == sample_web_document.metadata

        # Verify service calls
        mock_loader.load_single_document_with_images.assert_called_once_with(
            url="https://example.com/page"
        )
        mock_cleaner.clean_documents.assert_called_once()
        mock_processor.assert_called_once()

        # Verify load_single_document was not called (since we're using with_images=True)
        mock_loader.load_single_document.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._process_web_url")
    @patch("src.routes.documents_router.ChromaStore")
    async def test_add_web_document_success(
        self,
        mock_chroma_store_class,
        mock_process_web,
        test_client,
        sample_chunks,
        sample_chunk_ids,
        sample_web_request,
    ):
        """Test successful web document addition"""
        # Setup mocks
        mock_process_web.return_value = (
            sample_chunks,
            sample_chunk_ids,
            "https://example.com/page",
            {"source": "https://example.com/page", "title": "Test Web Page"},
        )

        # Setup ChromaStore mock
        mock_store = AsyncMock()
        mock_store.add_documents.return_value = (2, 0, [])
        mock_store.store_metadata = {
            "nb_collections": 1,
            "details": {"web_documents": {"count": 2}},
        }
        mock_chroma_store_class.return_value = mock_store

        # Use test client to call the endpoint directly
        with patch("fastapi.Depends", return_value=None):
            # Need to pass json data for the WebUrlRequest
            response = test_client.post(
                "/v1/documents/web/add",
                json={"web_url": "https://example.com/page", "with_images": False},
            )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["filename"] == "https://example.com/page"
        assert response_data["added_count"] == 2
        assert response_data["skipped_count"] == 0

        # Verify service calls
        mock_store.add_documents.assert_called_once_with(
            documents=sample_chunks,
            ids=sample_chunk_ids,
            collection_name=config.COLLECTION_NAME,
            is_web=True,
        )

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._process_web_url")
    @patch("src.routes.documents_router.ChromaStore")
    async def test_update_web_document_success(
        self,
        mock_chroma_store_class,
        mock_process_web,
        test_client,
        sample_chunks,
        sample_chunk_ids,
        sample_web_request,
    ):
        """Test successful web document update"""
        # Setup mocks
        mock_process_web.return_value = (
            sample_chunks,
            sample_chunk_ids,
            "https://example.com/page",
            {"source": "https://example.com/page", "title": "Test Web Page"},
        )

        # Setup ChromaStore mock
        mock_store = AsyncMock()
        mock_store.replace_documents.return_value = (2, 3, 1)
        mock_store.store_metadata = {
            "nb_collections": 1,
            "details": {"web_documents": {"count": 2}},
        }
        mock_chroma_store_class.return_value = mock_store

        # Use test client to call the endpoint directly
        with patch("fastapi.Depends", return_value=None):
            # Need to pass json data for the WebUrlRequest
            response = test_client.post(
                "/v1/documents/web/update",
                json={"web_url": "https://example.com/page", "with_images": False},
            )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["filename"] == "https://example.com/page"
        assert response_data["added_count"] == 2
        assert response_data["docs_replaced"] == 3
        assert response_data["sources_updated"] == 1

        # Verify service calls
        mock_store.replace_documents.assert_called_once_with(
            documents=sample_chunks,
            ids=sample_chunk_ids,
            collection_name=config.COLLECTION_NAME,
            is_web=True,
        )

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._process_web_url")
    @patch("src.routes.documents_router.ChromaStore")
    async def test_add_web_document_error_handling(
        self, mock_chroma_store_class, mock_process_web, test_client
    ):
        """Test error handling in add_web_document"""
        # Setup process_web to raise an exception
        mock_process_web.side_effect = Exception("Web processing failed")

        # Use test client to call the endpoint directly
        with patch("fastapi.Depends", return_value=None):
            # Need to pass json data for the WebUrlRequest
            response = test_client.post(
                "/v1/documents/web/add", json={"web_url": "https://example.com/page"}
            )

        # Verify response indicates failure
        assert response.status_code == 503
        assert "Web processing failed" in response.json()["detail"]

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._process_web_url")
    @patch("src.routes.documents_router.ChromaStore")
    async def test_update_web_document_error_handling(
        self,
        mock_chroma_store_class,
        mock_process_web,
        test_client,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test error handling in update_web_document"""
        # Setup mocks
        mock_process_web.return_value = (
            sample_chunks,
            sample_chunk_ids,
            "https://example.com/page",
            {"source": "https://example.com/page", "title": "Test Web Page"},
        )

        # Setup ChromaStore mock to raise an exception
        mock_store = AsyncMock()
        mock_store.replace_documents.side_effect = Exception("Vector store error")
        mock_chroma_store_class.return_value = mock_store

        # Use test client to call the endpoint directly
        with patch("fastapi.Depends", return_value=None):
            # Need to pass json data for the WebUrlRequest
            response = test_client.post(
                "/v1/documents/web/update", json={"web_url": "https://example.com/page"}
            )

        # Verify response indicates failure
        assert response.status_code == 503
        assert "Vector store error" in response.json()["detail"]
