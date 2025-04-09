from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.schema import Document

from src.services.vectorstore.chroma_store import ChromaStore, chroma_retriever


@pytest.fixture
def mock_client():
    """Mock ChromaDB client"""
    client = MagicMock()
    client.heartbeat.return_value = 1  # Healthy heartbeat
    client.count_collections.return_value = 2
    client.list_collections.return_value = ["test_collection", "default_collection"]

    # Create a mock collection that client.get_collection will return
    mock_collection = MagicMock()
    mock_collection.name = "test_collection"
    mock_collection.count.return_value = 10
    client.get_collection.return_value = mock_collection
    client.get_or_create_collection.return_value = mock_collection

    return client


@pytest.fixture
def mock_collection():
    """Mock ChromaDB collection"""
    collection = MagicMock()
    collection.name = "test_collection"
    collection.count.return_value = 10

    # Configure get method to return different results based on calls
    collection.get.return_value = {
        "ids": ["id1", "id2", "id3"],
        "metadatas": [
            {"source": "/path/to/doc1.pdf"},
            {"source": "/path/to/doc2.pdf"},
            {"source": "/path/to/doc3.pdf"},
        ],
    }

    return collection


@pytest.fixture
def mock_empty_collection():
    """Mock empty ChromaDB collection"""
    collection = MagicMock()
    collection.name = "empty_collection"
    collection.count.return_value = 0
    collection.get.return_value = {"ids": [], "metadatas": []}

    return collection


@pytest.fixture
def mock_large_collection():
    """Mock large ChromaDB collection for pagination testing"""
    collection = MagicMock()
    collection.name = "large_collection"
    collection.count.return_value = 2000

    # Configure the get method for pagination testing
    def side_effect(include=None, limit=None, offset=None):
        if offset >= 2000:
            return {"ids": [], "metadatas": []}

        # Create a subset of results based on limit and offset
        ids = [f"id{i}" for i in range(offset, min(offset + limit, 2000))]
        metadatas = [
            {"source": f"/path/to/doc{i}.pdf"}
            for i in range(offset, min(offset + limit, 2000))
        ]
        return {"ids": ids, "metadatas": metadatas}

    collection.get.side_effect = side_effect
    return collection


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(
            page_content="Test content 1", metadata={"source": "/path/to/doc1.pdf"}
        ),
        Document(
            page_content="Test content 2", metadata={"source": "/path/to/doc2.pdf"}
        ),
        Document(
            page_content="Test content 3", metadata={"source": "/path/to/doc3.pdf"}
        ),
        Document(page_content="No source", metadata={}),
    ]


@pytest.fixture
def sample_ids():
    """Create sample document IDs for testing"""
    return ["id1", "id2", "id3", "id4"]


class TestChromaStore:

    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    def test_init(self, mock_embeddings, mock_chroma_service, mock_client):
        """Test ChromaStore initialization"""
        mock_chroma_service.return_value = mock_client
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()

        assert store.client == mock_client
        mock_chroma_service.assert_called_once()
        mock_embeddings.assert_called_once()

    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    def test_store_metadata(self, mock_embeddings, mock_chroma_service, mock_client):
        """Test store_metadata property"""
        mock_chroma_service.return_value = mock_client
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()
        metadata = store.store_metadata

        assert metadata["nb_collections"] == 2
        assert "test_collection" in metadata["details"]
        assert "default_collection" in metadata["details"]
        assert metadata["details"]["test_collection"]["count"] == 10

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    async def test_check_connection_success(
        self, mock_embeddings, mock_chroma_service, mock_client
    ):
        """Test successful connection check"""
        mock_chroma_service.return_value = mock_client
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()
        # Should not raise exception
        await store._check_connection()

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    async def test_check_connection_failure(
        self, mock_embeddings, mock_chroma_service, mock_client
    ):
        """Test connection check failure"""
        mock_client.heartbeat.return_value = 0  # Failed heartbeat
        mock_chroma_service.return_value = mock_client
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()
        with pytest.raises(Exception) as excinfo:
            await store._check_connection()
        assert "ChromaDB is not available" in str(excinfo.value)

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    async def test_get_collection(
        self, mock_embeddings, mock_chroma_service, mock_client, mock_collection
    ):
        """Test getting a collection"""
        mock_chroma_service.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()
        collection = await store._get_collection("test_collection")

        assert collection == mock_collection
        mock_client.get_or_create_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    @patch("src.services.vectorstore.chroma_store.Chroma")
    async def test_get_vector_store(
        self,
        mock_chroma,
        mock_embeddings,
        mock_chroma_service,
        mock_client,
        mock_collection,
    ):
        """Test getting a vector store"""
        mock_chroma_service.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embeddings.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()

        store = ChromaStore()
        vector_store = await store._get_vector_store("test_collection")

        assert vector_store == mock_chroma.return_value
        mock_chroma.assert_called_once_with(
            client=mock_client,
            collection_name=mock_collection.name,
            embedding_function=store.embedding_function,
        )

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    async def test_get_source_tracker(
        self, mock_embeddings, mock_chroma_service, mock_client, mock_collection
    ):
        """Test source tracker with basic collection"""
        mock_chroma_service.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()
        sources = await store._get_source_tracker("test_collection")

        assert isinstance(sources, set)
        assert len(sources) == 3
        assert "doc1.pdf" in sources
        assert "doc2.pdf" in sources
        assert "doc3.pdf" in sources

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    async def test_get_source_tracker_empty(
        self, mock_embeddings, mock_chroma_service, mock_client, mock_empty_collection
    ):
        """Test source tracker with empty collection"""
        mock_chroma_service.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_empty_collection
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()
        sources = await store._get_source_tracker("empty_collection")

        assert isinstance(sources, set)
        assert len(sources) == 0

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    async def test_get_source_tracker_pagination(
        self, mock_embeddings, mock_chroma_service, mock_client, mock_large_collection
    ):
        """Test source tracker pagination with large collection"""
        mock_chroma_service.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_large_collection
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()
        sources = await store._get_source_tracker("large_collection")

        assert isinstance(sources, set)
        assert len(sources) == 2000  # All documents should be tracked

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    @patch("src.services.vectorstore.chroma_store.Chroma")
    async def test_get_retriever(
        self, mock_chroma, mock_embeddings, mock_chroma_service, mock_client
    ):
        """Test getting a retriever"""
        mock_chroma_service.return_value = mock_client
        mock_embeddings.return_value = MagicMock()

        # Configure mock vector store and retriever
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vector_store

        store = ChromaStore()
        retriever = await store.get_retriever("test_collection", k=5)

        assert retriever == mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    @patch("src.services.vectorstore.chroma_store.Chroma")
    async def test_add_documents_empty(
        self,
        mock_chroma,
        mock_embeddings,
        mock_chroma_service,
        mock_client,
        sample_documents,
        sample_ids,
    ):
        """Test adding documents with empty input"""
        mock_chroma_service.return_value = mock_client
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()
        with pytest.raises(ValueError) as excinfo:
            await store.add_documents([], [])

        assert "No documents provided for storage" in str(excinfo.value)

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    @patch("src.services.vectorstore.chroma_store.Chroma")
    async def test_add_documents_new(
        self,
        mock_chroma,
        mock_embeddings,
        mock_chroma_service,
        mock_client,
        mock_empty_collection,
        sample_documents,
        sample_ids,
    ):
        """Test adding documents to an empty collection"""
        mock_chroma_service.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_empty_collection
        mock_embeddings.return_value = MagicMock()

        # Configure mock vector store
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store

        store = ChromaStore()
        result = await store.add_documents(
            sample_documents, sample_ids, "empty_collection", batch_size=2
        )

        # Should add all documents
        assert result[0] == len(sample_documents)  # Added count
        assert result[1] == 0  # Skipped count
        assert len(result[2]) == 0  # No skipped sources

        # Check that vector_store.add_documents was called twice (2 batches of 2)
        assert mock_vector_store.add_documents.call_count == 2

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    @patch("src.services.vectorstore.chroma_store.Chroma")
    async def test_add_documents_skip_existing(
        self,
        mock_chroma,
        mock_embeddings,
        mock_chroma_service,
        mock_client,
        mock_collection,
        sample_documents,
        sample_ids,
    ):
        """Test adding documents with some existing sources"""
        mock_chroma_service.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embeddings.return_value = MagicMock()

        # Configure mock vector store
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store

        store = ChromaStore()
        result = await store.add_documents(
            sample_documents, sample_ids, "test_collection", batch_size=2
        )

        # Should skip the first 3 documents which have existing sources, add only the 4th with no source
        assert result[0] == 1  # Added count (only the one without source)
        assert result[1] == 3  # Skipped count
        assert len(result[2]) == 3  # Three skipped sources

        # Check that vector_store.add_documents was called once (1 batch for the single document)
        assert mock_vector_store.add_documents.call_count == 1

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    @patch("src.services.vectorstore.chroma_store.Chroma")
    async def test_add_documents_force_add(
        self,
        mock_chroma,
        mock_embeddings,
        mock_chroma_service,
        mock_client,
        mock_collection,
        sample_documents,
        sample_ids,
    ):
        """Test adding documents with force adding existing sources"""
        mock_chroma_service.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embeddings.return_value = MagicMock()

        # Configure mock vector store
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store

        store = ChromaStore()
        result = await store.add_documents(
            sample_documents,
            sample_ids,
            "test_collection",
            batch_size=2,
            skip_existing=False,
        )

        # Should add all documents
        assert result[0] == 4  # All documents added
        assert result[1] == 0  # No documents skipped
        assert len(result[2]) == 0  # No skipped sources

        # Check that vector_store.add_documents was called twice (2 batches of 2)
        assert mock_vector_store.add_documents.call_count == 2

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    @patch("src.services.vectorstore.chroma_store.Chroma")
    async def test_replace_documents_empty(
        self, mock_chroma, mock_embeddings, mock_chroma_service, mock_client
    ):
        """Test replacing documents with empty input"""
        mock_chroma_service.return_value = mock_client
        mock_embeddings.return_value = MagicMock()

        store = ChromaStore()
        with pytest.raises(ValueError) as excinfo:
            await store.replace_documents([], [])

        assert "No documents provided for replacement" in str(excinfo.value)

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.chroma_service")
    @patch("src.services.vectorstore.chroma_store.OpenAIEmbeddings")
    @patch("src.services.vectorstore.chroma_store.Chroma")
    async def test_replace_documents(
        self,
        mock_chroma,
        mock_embeddings,
        mock_chroma_service,
        mock_client,
        mock_collection,
        sample_documents,
        sample_ids,
    ):
        """Test replacing documents"""
        mock_chroma_service.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_embeddings.return_value = MagicMock()

        # Configure mock vector store
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store

        store = ChromaStore()
        # Mock the add_documents method to return a specific result
        store.add_documents = AsyncMock(return_value=(4, 0, []))

        result = await store.replace_documents(
            sample_documents, sample_ids, "test_collection"
        )

        assert result[0] == 4  # Added count
        assert result[1] == 3  # Replaced count (3 existing ids)
        assert result[2] == 3  # Updated sources (3 unique sources)

        # Verify the collection's delete method was called
        mock_collection.delete.assert_called()
        store.add_documents.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.vectorstore.chroma_store.ChromaStore")
    async def test_chroma_retriever_function(self, mock_chroma_store_class):
        """Test the standalone chroma_retriever function"""
        # Configure the mock
        mock_store_instance = AsyncMock()
        mock_retriever = MagicMock()
        mock_store_instance.get_retriever.return_value = mock_retriever
        mock_chroma_store_class.return_value = mock_store_instance

        # Call the function
        retriever = await chroma_retriever("custom_collection", k=7)

        # Verify results
        assert retriever == mock_retriever
        mock_chroma_store_class.assert_called_once()
        mock_store_instance.get_retriever.assert_called_once_with(
            "custom_collection", 7
        )
