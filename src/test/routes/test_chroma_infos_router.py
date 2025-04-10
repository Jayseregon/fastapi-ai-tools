from unittest.mock import MagicMock, patch

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
    def test_ping_success(self, mock_chroma_service, test_client, mock_chroma_client):
        """Test successful ping to ChromaDB"""
        # Setup mock
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.heartbeat.return_value = 42

        # Send request to the endpoint
        response = test_client.get("/v1/chroma-infos/ping")

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
    def test_ping_warning(self, mock_chroma_service, test_client, mock_chroma_client):
        """Test warning when ChromaDB returns zero heartbeat"""
        # Setup mock
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.heartbeat.return_value = 0

        # Send request
        response = test_client.get("/v1/chroma-infos/ping")

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "warning"
        assert data["message"] == "ChromaDB returned zero heartbeat"
        assert data["heartbeat"] == 0

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_ping_error(self, mock_chroma_service, test_client):
        """Test error handling when ChromaDB is unavailable"""
        # Setup mock to raise exception
        mock_chroma_service.return_value = MagicMock()
        mock_chroma_service.return_value.heartbeat.side_effect = Exception(
            "Connection refused"
        )

        # Send request
        response = test_client.get("/v1/chroma-infos/ping")

        # Check response for error status
        assert response.status_code == 503
        data = response.json()
        assert "Connection refused" in data["detail"]

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_list_collections_success(
        self, mock_chroma_service, test_client, mock_chroma_client
    ):
        """Test successful collection listing"""
        # Setup mock
        mock_chroma_service.return_value = mock_chroma_client

        # Send request
        response = test_client.get("/v1/chroma-infos/collections")

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
        self, mock_chroma_service, test_client, mock_chroma_client
    ):
        """Test when no collections exist"""
        # Setup mock
        mock_chroma_service.return_value = mock_chroma_client
        mock_chroma_client.list_collections.return_value = []

        # Send request
        response = test_client.get("/v1/chroma-infos/collections")

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data == {}  # Empty collections dict

        # Verify mocks called
        mock_chroma_service.assert_called_once()
        mock_chroma_client.list_collections.assert_called_once()
        mock_chroma_client.get_collection.assert_not_called()

    @patch("src.routes.chroma_infos_router.chroma_service")
    def test_list_collections_error(self, mock_chroma_service, test_client):
        """Test error handling when listing collections fails"""
        # Setup mock to raise exception
        mock_chroma_service.return_value = MagicMock()
        mock_chroma_service.return_value.list_collections.side_effect = Exception(
            "Database error"
        )

        # Send request
        response = test_client.get("/v1/chroma-infos/collections")

        # Check response for error status
        assert response.status_code == 500
        data = response.json()
        assert "Database error" in data["detail"]
