from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routes.chroma_infos_router import router


@pytest.fixture
def app():
    """Create a FastAPI app with the chroma_infos_router for testing"""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def test_client(app):
    """Create a test client for the app"""
    return TestClient(app)


@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client"""
    client = MagicMock()
    client.heartbeat.return_value = 42

    # Setup mock collection behavior
    collection1 = MagicMock()
    collection1.count.return_value = 10
    collection2 = MagicMock()
    collection2.count.return_value = 25

    client.list_collections.return_value = ["collection1", "collection2"]
    client.get_collection.side_effect = lambda name: {
        "collection1": collection1,
        "collection2": collection2,
    }.get(name)

    return client


class TestChromaInfosRouter:
    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_ping_success(
        self, mock_chroma_service, test_client, mock_chroma_client, auth_headers
    ):
        """Test successful ping to ChromaDB"""
        # Setup mock
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.heartbeat.return_value = 42

        # Send request to the endpoint
        response = test_client.get("/v1/chroma-infos/ping", headers=auth_headers)

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["message"] == "ChromaDB is available"
        assert data["heartbeat"] == 42

        # Verify mocks called
        mock_chroma_service.assert_called_once()
        mock_chroma_client.heartbeat.assert_called_once()

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_ping_warning(
        self, mock_chroma_service, test_client, mock_chroma_client, auth_headers
    ):
        """Test warning when ChromaDB returns zero heartbeat"""
        # Setup mock
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.heartbeat.return_value = 0

        # Send request
        response = test_client.get("/v1/chroma-infos/ping", headers=auth_headers)

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "warning"
        assert data["message"] == "ChromaDB returned zero heartbeat"
        assert data["heartbeat"] == 0

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_ping_error(self, mock_chroma_service, test_client, auth_headers):
        """Test error handling when ChromaDB is unavailable"""
        # Setup mock to raise exception
        mock_chroma_service.return_value = MagicMock()
        mock_chroma_service.return_value.heartbeat.side_effect = Exception(
            "Connection refused"
        )

        # Send request
        response = test_client.get("/v1/chroma-infos/ping", headers=auth_headers)

        # Check response for error status
        assert response.status_code == 503
        data = response.json()
        assert "Connection refused" in data["detail"]

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_list_collections_success(
        self, mock_chroma_service, test_client, mock_chroma_client, auth_headers
    ):
        """Test successful collection listing"""
        # Setup mock
        mock_chroma_service.return_value = mock_chroma_client

        # Send request
        response = test_client.get("/v1/chroma-infos/collections", headers=auth_headers)

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "collection1" in data
        assert "collection2" in data
        assert data["collection1"] == 10
        assert data["collection2"] == 25

        # Verify mocks called
        mock_chroma_service.assert_called_once()
        mock_chroma_client.list_collections.assert_called_once()
        assert mock_chroma_client.get_collection.call_count == 2

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_list_collections_empty(
        self, mock_chroma_service, test_client, mock_chroma_client, auth_headers
    ):
        """Test when no collections exist"""
        # Setup mock
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = []

        # Send request
        response = test_client.get("/v1/chroma-infos/collections", headers=auth_headers)

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data == {}  # Empty collections dict

        # Verify mocks called
        mock_chroma_service.assert_called_once()
        mock_chroma_client.list_collections.assert_called_once()
        mock_chroma_client.get_collection.assert_not_called()

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_list_collections_error(
        self, mock_chroma_service, test_client, auth_headers
    ):
        """Test error handling when listing collections fails"""
        # Setup mock to raise exception
        mock_chroma_service.return_value = MagicMock()
        mock_chroma_service.return_value.list_collections.side_effect = Exception(
            "Database error"
        )

        # Send request
        response = test_client.get("/v1/chroma-infos/collections", headers=auth_headers)

        # Check response for error status
        assert response.status_code == 500
        data = response.json()
        assert "Database error" in data["detail"]

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_delete_collection_success(
        self, mock_chroma_service, test_client, mock_chroma_client, auth_headers
    ):
        """Test successful deletion of a collection"""
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = [
            "collection1",
            "collection2",
        ]

        response = test_client.post(
            "/v1/chroma-infos/delete-collection",
            json={"collection_name": "collection1"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "deleted successfully" in data["message"]
        mock_chroma_client.delete_collection.assert_called_once_with("collection1")

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_delete_collection_not_found(
        self, mock_chroma_service, test_client, mock_chroma_client, auth_headers
    ):
        """Test deletion of a non-existent collection"""
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = [
            "collection1",
            "collection2",
        ]

        response = test_client.post(
            "/v1/chroma-infos/delete-collection",
            json={"collection_name": "doesnotexist"},
            headers=auth_headers,
        )

        assert response.status_code == 404
        data = response.json()
        assert "does not exist" in data["detail"]
        mock_chroma_client.delete_collection.assert_not_called()

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_delete_collection_error(
        self, mock_chroma_service, test_client, mock_chroma_client, auth_headers
    ):
        """Test error handling when deletion fails"""
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = ["collection1"]
        mock_chroma_client.delete_collection.side_effect = Exception("Internal error")

        response = test_client.post(
            "/v1/chroma-infos/delete-collection",
            json={"collection_name": "collection1"},
            headers=auth_headers,
        )

        assert response.status_code == 500
        data = response.json()
        assert "Internal error" in data["detail"]

    @patch("src.routes.chroma_infos_router.ChromaStore")
    def test_get_collections_with_sources_success(
        self, mock_chroma_store_class, test_client, auth_headers
    ):
        """Test successful retrieval of collections with their sources"""
        # Set up mock response
        mock_store_instance = MagicMock()
        # Use AsyncMock for the async method
        mock_store_instance.get_collections_with_sources = AsyncMock(
            return_value={
                "collection1": ["doc1.pdf", "doc2.pdf", "https://example.com/page1"],
                "collection2": ["doc3.pdf", "http://example.org/page2"],
            }
        )
        mock_chroma_store_class.return_value = mock_store_instance

        # Send request
        response = test_client.get(
            "/v1/chroma-infos/collections/list-sources", headers=auth_headers
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert len(data["collections"]) == 2
        assert "collection1" in data["collections"]
        assert "collection2" in data["collections"]
        assert "doc1.pdf" in data["collections"]["collection1"]
        assert "https://example.com/page1" in data["collections"]["collection1"]
        assert "http://example.org/page2" in data["collections"]["collection2"]

        # Verify mocks called
        mock_chroma_store_class.assert_called_once()
        mock_store_instance.get_collections_with_sources.assert_called_once()

    @patch("src.routes.chroma_infos_router.ChromaStore")
    def test_get_collections_with_sources_empty(
        self, mock_chroma_store_class, test_client, auth_headers
    ):
        """Test when no collections or sources exist"""
        # Set up mock response
        mock_store_instance = MagicMock()
        # Use AsyncMock for the async method
        mock_store_instance.get_collections_with_sources = AsyncMock(return_value={})
        mock_chroma_store_class.return_value = mock_store_instance

        # Send request
        response = test_client.get(
            "/v1/chroma-infos/collections/list-sources", headers=auth_headers
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert data["collections"] == {}

    @patch("src.routes.chroma_infos_router.ChromaStore")
    def test_get_collections_with_sources_error(
        self, mock_chroma_store_class, test_client, auth_headers
    ):
        """Test error handling when listing collections with sources fails"""
        # Set up mock to raise exception
        mock_store_instance = MagicMock()
        mock_store_instance.get_collections_with_sources = AsyncMock(
            side_effect=Exception("Failed to retrieve sources")
        )
        mock_chroma_store_class.return_value = mock_store_instance

        # Send request
        response = test_client.get(
            "/v1/chroma-infos/collections/list-sources", headers=auth_headers
        )

        # Check response for error status
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve sources" in data["detail"]

    @patch("src.routes.chroma_infos_router.chroma_service")
    @patch("src.routes.chroma_infos_router.ChromaStore")
    def test_delete_source_success(
        self,
        mock_chroma_store_class,
        mock_chroma_service,
        test_client,
        mock_chroma_client,
        auth_headers,
    ):
        """Test successful deletion of documents from a source"""
        # Set up mocks
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = [
            "collection1",
            "collection2",
        ]

        mock_store_instance = MagicMock()
        # Use AsyncMock for the async method
        mock_store_instance.delete_source_documents = AsyncMock(
            return_value=5
        )  # 5 documents deleted
        mock_chroma_store_class.return_value = mock_store_instance

        # Send request
        response = test_client.post(
            "/v1/chroma-infos/collections/delete-source",
            json={"collection_name": "collection1", "source_name": "doc1.pdf"},
            headers=auth_headers,
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Successfully deleted source" in data["message"]
        assert data["documents_deleted"] == 5

        # Verify mocks called
        mock_chroma_client.list_collections.assert_called_once()
        mock_store_instance.delete_source_documents.assert_called_once_with(
            collection_name="collection1", source_name="doc1.pdf"
        )

    @patch("src.routes.chroma_infos_router.chroma_service")
    @patch("src.routes.chroma_infos_router.ChromaStore")
    def test_delete_source_web_url_success(
        self,
        mock_chroma_store_class,
        mock_chroma_service,
        test_client,
        mock_chroma_client,
        auth_headers,
    ):
        """Test successful deletion of documents from a web URL source"""
        # Set up mocks
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = [
            "collection1",
            "collection2",
        ]

        mock_store_instance = MagicMock()
        # Use AsyncMock for the async method
        mock_store_instance.delete_source_documents = AsyncMock(
            return_value=3
        )  # 3 documents deleted
        mock_chroma_store_class.return_value = mock_store_instance

        # Send request with a web URL
        response = test_client.post(
            "/v1/chroma-infos/collections/delete-source",
            json={
                "collection_name": "collection1",
                "source_name": "https://example.com/page1",
            },
            headers=auth_headers,
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Successfully deleted source" in data["message"]
        assert data["documents_deleted"] == 3

        # Verify mocks called with correct parameters
        mock_store_instance.delete_source_documents.assert_called_once_with(
            collection_name="collection1", source_name="https://example.com/page1"
        )

    @patch("src.routes.chroma_infos_router.chroma_service")
    @patch("src.routes.chroma_infos_router.ChromaStore")
    def test_delete_source_not_found(
        self,
        mock_chroma_store_class,
        mock_chroma_service,
        test_client,
        mock_chroma_client,
        auth_headers,
    ):
        """Test when no documents found for the source"""
        # Set up mocks
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = [
            "collection1",
            "collection2",
        ]

        mock_store_instance = MagicMock()
        # Use AsyncMock for the async method
        mock_store_instance.delete_source_documents = AsyncMock(
            return_value=0
        )  # No documents found
        mock_chroma_store_class.return_value = mock_store_instance

        # Send request
        response = test_client.post(
            "/v1/chroma-infos/collections/delete-source",
            json={"collection_name": "collection1", "source_name": "nonexistent.pdf"},
            headers=auth_headers,
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "warning"
        assert "No documents found" in data["message"]
        assert data["documents_deleted"] == 0

    @patch("src.routes.chroma_infos_router.chroma_service")
    @patch("src.routes.chroma_infos_router.ChromaStore")
    def test_delete_source_collection_not_found(
        self,
        mock_chroma_store_class,
        mock_chroma_service,
        test_client,
        mock_chroma_client,
        auth_headers,
    ):
        """Test when the collection doesn't exist"""
        # Set up mocks
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = [
            "collection1",
            "collection2",
        ]

        # Send request with nonexistent collection
        response = test_client.post(
            "/v1/chroma-infos/collections/delete-source",
            json={"collection_name": "nonexistent", "source_name": "doc1.pdf"},
            headers=auth_headers,
        )

        # Check response
        assert response.status_code == 404
        data = response.json()
        assert "does not exist" in data["detail"]

        # Verify delete_source_documents was not called
        mock_chroma_store_class.assert_not_called()

    @patch("src.routes.chroma_infos_router.chroma_service")
    @patch("src.routes.chroma_infos_router.ChromaStore")
    def test_delete_source_error(
        self,
        mock_chroma_store_class,
        mock_chroma_service,
        test_client,
        mock_chroma_client,
        auth_headers,
    ):
        """Test error handling when source deletion fails"""
        # Set up mocks
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = [
            "collection1",
            "collection2",
        ]

        mock_store_instance = MagicMock()
        mock_store_instance.delete_source_documents = AsyncMock(
            side_effect=Exception("Database connection error")
        )
        mock_chroma_store_class.return_value = mock_store_instance

        # Send request
        response = test_client.post(
            "/v1/chroma-infos/collections/delete-source",
            json={"collection_name": "collection1", "source_name": "doc1.pdf"},
            headers=auth_headers,
        )

        # Check response for error status
        assert response.status_code == 500
        data = response.json()
        assert "Database connection error" in data["detail"]
