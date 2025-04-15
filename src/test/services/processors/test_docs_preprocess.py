from unittest.mock import patch

import pytest
from langchain.schema import Document

from src.services.processors.docs_preprocess import DocumentsPreprocessing


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(page_content="Document 1 content", metadata={"source": "file1.pdf"}),
        Document(page_content="Document 2 content", metadata={"source": "file2.pdf"}),
        Document(page_content="Document 3 content", metadata={"source": "file3.pdf"}),
    ]


@pytest.fixture
def sample_chunks():
    """Create sample document chunks after splitting"""
    return [
        Document(page_content="Chunk 1", metadata={"source": "file1.pdf"}),
        Document(page_content="Chunk 2", metadata={"source": "file1.pdf"}),
        Document(page_content="Chunk 3", metadata={"source": "file2.pdf"}),
        Document(page_content="Chunk 4", metadata={"source": "file3.pdf"}),
        Document(page_content="Chunk 5", metadata={"source": "file3.pdf"}),
    ]


@pytest.fixture
def sample_chunk_ids():
    """Create sample document chunk IDs"""
    return [
        "file1-0-12345678",
        "file1-1-23456789",
        "file2-2-34567890",
        "file3-3-45678901",
        "file3-4-56789012",
    ]


class TestDocumentsPreprocessing:

    @pytest.mark.asyncio
    @patch("src.services.processors.docs_preprocess.text_splitter_recursive_char")
    @patch("src.services.processors.docs_preprocess.create_chunk_ids")
    async def test_basic_preprocessing(
        self,
        mock_create_ids,
        mock_text_splitter,
        sample_documents,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test basic document preprocessing with default parameters"""
        # Configure mocks
        mock_text_splitter.return_value = sample_chunks
        mock_create_ids.return_value = sample_chunk_ids

        # Create processor and process documents
        processor = DocumentsPreprocessing()
        result = await processor(sample_documents)

        # Check results
        chunks, ids = result
        assert chunks == sample_chunks
        assert ids == sample_chunk_ids

        # Verify correct calls to dependencies
        mock_text_splitter.assert_called_once_with(sample_documents, 1000, 200)
        mock_create_ids.assert_called_once_with(sample_chunks, None)

    @pytest.mark.asyncio
    @patch("src.services.processors.docs_preprocess.text_splitter_recursive_char")
    @patch("src.services.processors.docs_preprocess.create_chunk_ids")
    async def test_custom_chunk_parameters(
        self,
        mock_create_ids,
        mock_text_splitter,
        sample_documents,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test document preprocessing with custom chunk size and overlap"""
        # Configure mocks
        mock_text_splitter.return_value = sample_chunks
        mock_create_ids.return_value = sample_chunk_ids

        # Custom parameters
        chunk_size = 500
        chunk_overlap = 100

        # Create processor and process documents
        processor = DocumentsPreprocessing()
        result = await processor(
            sample_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Check results
        chunks, ids = result
        assert chunks == sample_chunks
        assert ids == sample_chunk_ids

        # Verify custom parameters were passed correctly
        mock_text_splitter.assert_called_once_with(
            sample_documents, chunk_size, chunk_overlap
        )
        mock_create_ids.assert_called_once_with(sample_chunks, None)

    @pytest.mark.asyncio
    @patch("src.services.processors.docs_preprocess.text_splitter_recursive_char")
    @patch("src.services.processors.docs_preprocess.create_chunk_ids")
    async def test_custom_prefix(
        self,
        mock_create_ids,
        mock_text_splitter,
        sample_documents,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test document preprocessing with custom prefix for IDs"""
        # Configure mocks
        mock_text_splitter.return_value = sample_chunks
        mock_create_ids.return_value = sample_chunk_ids

        # Custom prefix
        prefix = "test-prefix"

        # Create processor and process documents
        processor = DocumentsPreprocessing()
        result = await processor(sample_documents, prefix=prefix)

        # Check results
        chunks, ids = result
        assert chunks == sample_chunks
        assert ids == sample_chunk_ids

        # Verify prefix was passed correctly
        mock_text_splitter.assert_called_once_with(sample_documents, 1000, 200)
        mock_create_ids.assert_called_once_with(sample_chunks, prefix)

    @pytest.mark.asyncio
    @patch("src.services.processors.docs_preprocess.text_splitter_recursive_char")
    @patch("src.services.processors.docs_preprocess.create_chunk_ids")
    async def test_empty_documents(self, mock_create_ids, mock_text_splitter):
        """Test processing with empty document list"""
        # Configure mocks for empty results
        mock_text_splitter.return_value = []
        mock_create_ids.return_value = []

        # Create processor and process empty list
        processor = DocumentsPreprocessing()
        chunks, ids = await processor([])

        # Check results
        assert chunks == []
        assert ids == []

        # Verify calls with empty list
        mock_text_splitter.assert_called_once_with([], 1000, 200)
        mock_create_ids.assert_called_once_with([], None)

    @pytest.mark.asyncio
    @patch("src.services.processors.docs_preprocess.asyncio.to_thread")
    async def test_asyncio_thread_usage(
        self, mock_to_thread, sample_documents, sample_chunks, sample_chunk_ids
    ):
        """Test that operations are properly executed in threads using asyncio.to_thread"""
        # Configure mock to return expected results for both calls
        mock_to_thread.side_effect = [sample_chunks, sample_chunk_ids]

        # Create processor and process documents
        processor = DocumentsPreprocessing()
        result = await processor(sample_documents)

        # Check results
        chunks, ids = result
        assert chunks == sample_chunks
        assert ids == sample_chunk_ids

        # Verify asyncio.to_thread was called twice (once for splitting, once for IDs)
        assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    @patch("src.services.processors.docs_preprocess.text_splitter_recursive_char")
    @patch("src.services.processors.docs_preprocess.create_chunk_ids")
    async def test_all_parameters_combined(
        self,
        mock_create_ids,
        mock_text_splitter,
        sample_documents,
        sample_chunks,
        sample_chunk_ids,
    ):
        """Test document preprocessing with all custom parameters combined"""
        # Configure mocks
        mock_text_splitter.return_value = sample_chunks
        mock_create_ids.return_value = sample_chunk_ids

        # Custom parameters
        chunk_size = 750
        chunk_overlap = 150
        prefix = "combined-test"

        # Create processor and process documents
        processor = DocumentsPreprocessing()
        result = await processor(
            sample_documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            prefix=prefix,
        )

        # Check results
        chunks, ids = result
        assert chunks == sample_chunks
        assert ids == sample_chunk_ids

        # Verify all parameters were passed correctly
        mock_text_splitter.assert_called_once_with(
            sample_documents, chunk_size, chunk_overlap
        )
        mock_create_ids.assert_called_once_with(sample_chunks, prefix)
