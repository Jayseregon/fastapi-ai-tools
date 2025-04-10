from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.schema import Document

from src.configs.env_config import config
from src.routes.documents_router import _process_pdf_file, router


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


class TestDocumentsRouter:
    @pytest.mark.asyncio
    @patch("src.routes.documents_router.tempfile.mkdtemp")
    @patch("src.routes.documents_router.shutil.copyfileobj")
    @patch("src.routes.documents_router.open")
    @patch("src.routes.documents_router.shutil.rmtree")
    @patch("src.routes.documents_router.PdfLoader")
    @patch("src.routes.documents_router.PdfDocumentCleaner")
    @patch("src.routes.documents_router.DocumentsPreprocessing")
    async def test_process_pdf_file(
        self,
        mock_processor_class,
        mock_cleaner_class,
        mock_loader_class,
        mock_rmtree,
        mock_open,
        mock_copyfileobj,
        mock_mkdtemp,
        mock_pdf_file,
        sample_document,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test the internal _process_pdf_file function"""
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_open.return_value = MagicMock()

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
        result = await _process_pdf_file(mock_pdf_file, "/tmp/test_dir")

        # Verify results
        chunks, ids, filename, metadata = result
        assert chunks == sample_chunks
        assert ids == sample_chunk_ids
        assert filename == "test_document.pdf"
        assert metadata == sample_document.metadata

        # Verify file operations
        mock_open.assert_called_once_with("/tmp/test_dir/test_document.pdf", "wb")
        mock_copyfileobj.assert_called_once_with(
            mock_pdf_file.file, mock_open.return_value.__enter__.return_value
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
    @patch("src.routes.documents_router._process_pdf_file")
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
        mock_process_pdf,
        test_client,
        mock_pdf_file,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test successful PDF document addition"""
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True
        mock_process_pdf.return_value = (
            sample_chunks,
            sample_chunk_ids,
            "test_document.pdf",
            {"source": "test_document.pdf", "title": "Test Document"},
        )

        # Setup ChromaStore mock
        mock_store = AsyncMock()
        mock_store.add_documents.return_value = (2, 0, [])
        mock_store.store_metadata = {
            "nb_collections": 1,
            "details": {"pdf_documents": {"count": 2}},
        }
        mock_chroma_store_class.return_value = mock_store

        # Use test client to call the endpoint directly
        with patch("fastapi.Depends", return_value=None):
            response = test_client.post(
                "/v1/documents/pdf/add",
                files={
                    "pdf_file": (
                        "test_document.pdf",
                        b"test content",
                        "application/pdf",
                    )
                },
            )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["filename"] == "test_document.pdf"
        assert response_data["added_count"] == 2
        assert response_data["skipped_count"] == 0

        # Verify service calls
        mock_store.add_documents.assert_called_once_with(
            documents=sample_chunks,
            ids=sample_chunk_ids,
            collection_name=config.COLLECTION_NAME,
        )

        # Verify cleanup
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._process_pdf_file")
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
        mock_process_pdf,
        test_client,
        mock_pdf_file,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test successful PDF document update"""
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True
        mock_process_pdf.return_value = (
            sample_chunks,
            sample_chunk_ids,
            "test_document.pdf",
            {"source": "test_document.pdf", "title": "Test Document"},
        )

        # Setup ChromaStore mock
        mock_store = AsyncMock()
        mock_store.replace_documents.return_value = (2, 3, 1)
        mock_store.store_metadata = {
            "nb_collections": 1,
            "details": {"pdf_documents": {"count": 2}},
        }
        mock_chroma_store_class.return_value = mock_store

        # Use test client to call the endpoint
        with patch("fastapi.Depends", return_value=None):
            response = test_client.post(
                "/v1/documents/pdf/update",
                files={
                    "pdf_file": (
                        "test_document.pdf",
                        b"test content",
                        "application/pdf",
                    )
                },
            )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["filename"] == "test_document.pdf"
        assert response_data["added_count"] == 2
        assert response_data["docs_replaced"] == 3
        assert response_data["sources_updated"] == 1

        # Verify service calls
        mock_store.replace_documents.assert_called_once_with(
            documents=sample_chunks,
            ids=sample_chunk_ids,
            collection_name=config.COLLECTION_NAME,
        )

        # Verify cleanup
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._process_pdf_file")
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
        mock_process_pdf,
        test_client,
    ):
        """Test error handling in add_pdf_document"""
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True

        # Setup process_pdf to raise an exception
        mock_process_pdf.side_effect = Exception("PDF processing failed")

        # Send request using test_client and assert exception is raised
        with pytest.raises(Exception, match="PDF processing failed"):
            with patch("fastapi.Depends", return_value=None):
                test_client.post(
                    "/v1/documents/pdf/add",
                    files={
                        "pdf_file": (
                            "test_document.pdf",
                            b"test content",
                            "application/pdf",
                        )
                    },
                )

        # Verify cleanup still occurs - this is the key thing we're testing
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._process_pdf_file")
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
        mock_process_pdf,
        test_client,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test error handling in update_pdf_document"""
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True
        mock_process_pdf.return_value = (
            sample_chunks,
            sample_chunk_ids,
            "test_document.pdf",
            {"source": "test_document.pdf", "title": "Test Document"},
        )

        # Setup ChromaStore mock to raise an exception
        mock_store = AsyncMock()
        mock_store.replace_documents.side_effect = Exception("Vector store error")
        mock_chroma_store_class.return_value = mock_store

        # Send request using test_client and assert exception is raised
        with pytest.raises(Exception, match="Vector store error"):
            with patch("fastapi.Depends", return_value=None):
                test_client.post(
                    "/v1/documents/pdf/update",
                    files={
                        "pdf_file": (
                            "test_document.pdf",
                            b"test content",
                            "application/pdf",
                        )
                    },
                )

        # Verify cleanup still occurs - this is the key thing we're testing
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    @pytest.mark.asyncio
    @patch("src.routes.documents_router._process_pdf_file")
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
        mock_process_pdf,
        test_client,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test PDF document addition with some chunks skipped"""
        # Setup mocks
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_exists.return_value = True
        mock_process_pdf.return_value = (
            sample_chunks,
            sample_chunk_ids,
            "test_document.pdf",
            {"source": "test_document.pdf", "title": "Test Document"},
        )

        # Setup ChromaStore mock for partial addition (some docs skipped)
        mock_store = AsyncMock()
        mock_store.add_documents.return_value = (1, 1, ["other_document.pdf"])
        mock_store.store_metadata = {
            "nb_collections": 1,
            "details": {"pdf_documents": {"count": 1}},
        }
        mock_chroma_store_class.return_value = mock_store

        # Use test client to call the endpoint
        with patch("fastapi.Depends", return_value=None):
            response = test_client.post(
                "/v1/documents/pdf/add",
                files={
                    "pdf_file": (
                        "test_document.pdf",
                        b"test content",
                        "application/pdf",
                    )
                },
            )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["added_count"] == 1
        assert response_data["skipped_count"] == 1
        assert response_data["skipped_sources"] == ["other_document.pdf"]
