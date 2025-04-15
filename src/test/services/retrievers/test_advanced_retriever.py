from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.schema import Document

from src.services.retrievers.advanced_retriever import MultiQRerankedRetriever


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(page_content="Document 1 content", metadata={"source": "source1"}),
        Document(page_content="Document 2 content", metadata={"source": "source2"}),
        Document(page_content="Document 3 content", metadata={"source": "source3"}),
    ]


class TestMultiQRerankedRetriever:
    @patch("src.services.retrievers.advanced_retriever.chroma_service")
    @patch("src.services.retrievers.advanced_retriever.ChatOpenAI")
    def test_init(self, mock_chat_openai, mock_chroma_service):
        """Test initialization of MultiQRerankedRetriever"""
        mock_client = MagicMock()
        mock_chroma_service.return_value = mock_client
        mock_chat_openai.return_value = MagicMock()

        retriever = MultiQRerankedRetriever()

        assert retriever.client == mock_client
        assert retriever.llm == mock_chat_openai.return_value
        mock_chat_openai.assert_called_once()
        mock_chroma_service.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.retrievers.advanced_retriever.chroma_retriever")
    @patch("src.services.retrievers.advanced_retriever.ContextualCompressionRetriever")
    @patch("src.services.retrievers.advanced_retriever.FlashrankRerank")
    @patch("src.services.retrievers.advanced_retriever.MultiQueryRetriever")
    @patch("src.services.retrievers.advanced_retriever.chroma_service")
    @patch("src.services.retrievers.advanced_retriever.ChatOpenAI")
    async def test_retriever_creation(
        self,
        mock_chat_openai,
        mock_chroma_service,
        mock_multi_query_retriever,
        mock_flashrank_rerank,
        mock_contextual_compression_retriever,
        mock_chroma_retriever,
    ):
        """Test creation of advanced retriever"""
        mock_client = MagicMock()
        mock_chroma_service.return_value = mock_client
        mock_chat_openai.return_value = MagicMock()

        mock_base_retriever = MagicMock()
        mock_chroma_retriever.return_value = mock_base_retriever

        mock_compressor = MagicMock()
        mock_flashrank_rerank.return_value = mock_compressor

        mock_multi_query = MagicMock()
        mock_multi_query_retriever.from_llm.return_value = mock_multi_query

        mock_contextual = MagicMock()
        mock_contextual_compression_retriever.return_value = mock_contextual

        retriever = MultiQRerankedRetriever()
        result = await retriever._retriever("test_collection", top_n=5)

        # Check that all the components were created correctly
        mock_chroma_retriever.assert_called_once_with(collection_name="test_collection")
        mock_flashrank_rerank.assert_called_once_with(top_n=5)
        mock_multi_query_retriever.from_llm.assert_called_once_with(
            retriever=mock_base_retriever,
            llm=retriever.llm,
            include_original=True,
        )
        mock_contextual_compression_retriever.assert_called_once_with(
            base_compressor=mock_compressor, base_retriever=mock_multi_query
        )
        assert result == mock_contextual

    @pytest.mark.asyncio
    @patch("src.services.retrievers.advanced_retriever.chroma_service")
    @patch("src.services.retrievers.advanced_retriever.ChatOpenAI")
    async def test_call_method(
        self,
        mock_chat_openai,
        mock_chroma_service,
        sample_documents,
    ):
        """Test __call__ method for document retrieval"""
        mock_client = MagicMock()
        mock_chroma_service.return_value = mock_client
        mock_chat_openai.return_value = MagicMock()

        mock_contextual_retriever = MagicMock()
        mock_contextual_retriever.ainvoke = AsyncMock(return_value=sample_documents)

        retriever = MultiQRerankedRetriever()
        # Mock the _retriever method
        retriever._retriever = AsyncMock(return_value=mock_contextual_retriever)

        results = await retriever("test query", "test_collection", top_n=4)

        # Check results and method calls
        assert results == sample_documents
        retriever._retriever.assert_called_once_with(
            collection_name="test_collection", top_n=4
        )
        mock_contextual_retriever.ainvoke.assert_called_once_with("test query")

    @pytest.mark.asyncio
    @patch("src.services.retrievers.advanced_retriever.chroma_service")
    @patch("src.services.retrievers.advanced_retriever.ChatOpenAI")
    @patch("src.services.retrievers.advanced_retriever.chroma_retriever")
    @patch("src.services.retrievers.advanced_retriever.ContextualCompressionRetriever")
    @patch("src.services.retrievers.advanced_retriever.FlashrankRerank")
    @patch("src.services.retrievers.advanced_retriever.MultiQueryRetriever")
    async def test_end_to_end_retrieval(
        self,
        mock_multi_query_retriever,
        mock_flashrank_rerank,
        mock_contextual_compression_retriever,
        mock_chroma_retriever,
        mock_chat_openai,
        mock_chroma_service,
        sample_documents,
    ):
        """Test end-to-end document retrieval process"""
        mock_client = MagicMock()
        mock_chroma_service.return_value = mock_client
        mock_chat_openai.return_value = MagicMock()

        # Set up the retriever chain
        mock_base_retriever_instance = MagicMock()
        mock_multi_query_instance = MagicMock()
        mock_contextual_instance = MagicMock()

        mock_chroma_retriever.return_value = mock_base_retriever_instance
        mock_flashrank_rerank.return_value = MagicMock()
        mock_multi_query_retriever.from_llm.return_value = mock_multi_query_instance
        mock_contextual_compression_retriever.return_value = mock_contextual_instance

        # Configure the final retriever to return sample documents
        mock_contextual_instance.ainvoke = AsyncMock(return_value=sample_documents)

        # Create retriever and test
        retriever = MultiQRerankedRetriever()
        results = await retriever("complex query", "test_collection", top_n=3)

        # Verify results and correct call chain
        assert results == sample_documents
        assert len(results) == 3

        mock_chroma_retriever.assert_called_once_with(collection_name="test_collection")
        mock_flashrank_rerank.assert_called_once_with(top_n=3)
        mock_multi_query_retriever.from_llm.assert_called_once()
        mock_contextual_compression_retriever.assert_called_once()
        mock_contextual_instance.ainvoke.assert_called_once_with("complex query")

    @pytest.mark.asyncio
    @patch("src.services.retrievers.advanced_retriever.chroma_service")
    @patch("src.services.retrievers.advanced_retriever.ChatOpenAI")
    async def test_retriever_with_empty_results(
        self,
        mock_chat_openai,
        mock_chroma_service,
    ):
        """Test retriever when there are no results found"""
        mock_client = MagicMock()
        mock_chroma_service.return_value = mock_client
        mock_chat_openai.return_value = MagicMock()

        # Set up empty retriever
        empty_retriever = MagicMock()
        empty_retriever.ainvoke = AsyncMock(return_value=[])

        retriever = MultiQRerankedRetriever()
        # Mock the _retriever method
        retriever._retriever = AsyncMock(return_value=empty_retriever)

        results = await retriever("empty query", "empty_collection")

        # Should return empty list
        assert results == []
        empty_retriever.ainvoke.assert_called_once_with("empty query")

    @pytest.mark.asyncio
    @patch("src.services.retrievers.advanced_retriever.chroma_service")
    @patch("src.services.retrievers.advanced_retriever.ChatOpenAI")
    @patch("src.services.retrievers.advanced_retriever.chroma_retriever")
    async def test_retriever_handles_errors(
        self,
        mock_chroma_retriever,
        mock_chat_openai,
        mock_chroma_service,
    ):
        """Test error handling in retriever creation"""
        mock_client = MagicMock()
        mock_chroma_service.return_value = mock_client
        mock_chat_openai.return_value = MagicMock()
        mock_chroma_retriever.side_effect = Exception("Connection error")

        retriever = MultiQRerankedRetriever()

        # Error should be raised when calling _retriever due to error in chroma_retriever
        with pytest.raises(Exception) as excinfo:
            await retriever._retriever("test_collection")

        assert "Connection error" in str(excinfo.value)
        mock_chroma_retriever.assert_called_once()
