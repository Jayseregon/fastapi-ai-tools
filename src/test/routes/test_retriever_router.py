from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.schema import Document

from src.configs.env_config import config
from src.models.retriever_models import (
    DocumentMetadata,
    RetrievedDocument,
    RetrieverResponse,
)
from src.routes.retriever_router import router


@pytest.fixture
def app():
    """Create a FastAPI app with the retriever router for testing"""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def test_client(app):
    """Create a test client for the app"""
    return TestClient(app)


@pytest.fixture
def query_request():
    """Create a sample query request"""
    return {"query": "test query about AI"}


@pytest.fixture
def sample_langchain_documents():
    """Create sample langchain documents for testing"""
    return [
        Document(
            page_content="AI is a field of computer science that focuses on creating intelligent machines.",
            metadata={
                "source": "ai_textbook.pdf",
                "page": 42,
                "author": "AI Researcher",
                "title": "Introduction to AI",
                "id": "doc1",
                "relevance_score": 0.95,
                "document_type": "textbook",
                "description": "An introduction to AI concepts",
            },
        ),
        Document(
            page_content="Machine learning is a subset of AI that enables systems to learn from data.",
            metadata={
                "source": "ai_textbook.pdf",
                "page": 43,
                "author": "AI Researcher",
                "title": "Introduction to AI",
                "id": "doc2",
                "relevance_score": 0.85,
                "document_type": "textbook",
                "description": "An introduction to machine learning",
            },
        ),
    ]


@pytest.fixture
def mock_retrieved_documents():
    """Create sample retrieved documents that match the RetrieverResponse model"""
    return RetrieverResponse(
        documents=[
            RetrievedDocument(
                metadata=DocumentMetadata(
                    id="doc1",
                    relevance_score=0.95,
                    description="An introduction to AI concepts",
                    title="Introduction to AI",
                    document_type="textbook",
                    source="ai_textbook.pdf",
                ),
                page_content="AI is a field of computer science that focuses on creating intelligent machines.",
            ),
            RetrievedDocument(
                metadata=DocumentMetadata(
                    id="doc2",
                    relevance_score=0.85,
                    description="An introduction to machine learning",
                    title="Introduction to AI",
                    document_type="textbook",
                    source="ai_textbook.pdf",
                ),
                page_content="Machine learning is a subset of AI that enables systems to learn from data.",
            ),
        ]
    )


class TestRetrieverRouter:
    @pytest.mark.asyncio
    @patch("src.routes.retriever_router.MultiQRerankedRetriever")
    async def test_query_base_collection_success(
        self,
        mock_retriever_class,
        test_client,
        query_request,
        sample_langchain_documents,
        auth_headers,
    ):
        """Test successful vector store querying (base_collection)"""
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.return_value = sample_langchain_documents
        mock_retriever_class.return_value = mock_retriever_instance

        response = test_client.post(
            "/v1/retriever/base_collection/invoke",
            json=query_request,
            headers=auth_headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["documents"]) == 2
        assert (
            response_data["documents"][0]["page_content"]
            == "AI is a field of computer science that focuses on creating intelligent machines."
        )
        assert response_data["documents"][0]["metadata"]["source"] == "ai_textbook.pdf"
        assert response_data["documents"][0]["metadata"]["document_type"] == "textbook"
        assert response_data["documents"][0]["metadata"]["id"] == "doc1"
        mock_retriever_instance.assert_called_once_with(
            query=query_request["query"], collection_name=config.COLLECTION_NAME
        )

    @pytest.mark.asyncio
    @patch("src.routes.retriever_router.MultiQRerankedRetriever")
    async def test_query_base_collection_error(
        self, mock_retriever_class, test_client, query_request, auth_headers
    ):
        """Test error handling in vector store querying (base_collection)"""
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.side_effect = Exception("Vector store query failed")
        mock_retriever_class.return_value = mock_retriever_instance

        response = test_client.post(
            "/v1/retriever/base_collection/invoke",
            json=query_request,
            headers=auth_headers,
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Error querying vector store"

    @pytest.mark.asyncio
    @patch("src.routes.retriever_router.MultiQRerankedRetriever")
    async def test_query_base_collection_with_empty_results(
        self, mock_retriever_class, test_client, query_request, auth_headers
    ):
        """Test when retriever returns no documents (base_collection)"""
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.return_value = []
        mock_retriever_class.return_value = mock_retriever_instance

        response = test_client.post(
            "/v1/retriever/base_collection/invoke",
            json=query_request,
            headers=auth_headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["documents"]) == 0

    @pytest.mark.asyncio
    async def test_base_collection_invalid_query_format(
        self, test_client, auth_headers
    ):
        """Test with invalid query format (base_collection)"""
        response = test_client.post(
            "/v1/retriever/base_collection/invoke", json={}, headers=auth_headers
        )
        assert response.status_code == 422
        assert "field required" in response.text.lower()

    # --- SETICS COLLECTION TESTS ---

    @pytest.mark.asyncio
    @patch("src.routes.retriever_router.MultiQRerankedRetriever")
    async def test_query_setics_collection_success(
        self,
        mock_retriever_class,
        test_client,
        query_request,
        sample_langchain_documents,
        auth_headers,
    ):
        """Test successful vector store querying (setics_collection)"""
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.return_value = sample_langchain_documents
        mock_retriever_class.return_value = mock_retriever_instance

        response = test_client.post(
            "/v1/retriever/setics_collection/invoke",
            json=query_request,
            headers=auth_headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["documents"]) == 2
        assert (
            response_data["documents"][0]["page_content"]
            == "AI is a field of computer science that focuses on creating intelligent machines."
        )
        assert response_data["documents"][0]["metadata"]["source"] == "ai_textbook.pdf"
        assert response_data["documents"][0]["metadata"]["document_type"] == "textbook"
        assert response_data["documents"][0]["metadata"]["id"] == "doc1"
        mock_retriever_instance.assert_called_once_with(
            query=query_request["query"], collection_name=config.SETICS_COLLECTION
        )

    @pytest.mark.asyncio
    @patch("src.routes.retriever_router.MultiQRerankedRetriever")
    async def test_query_setics_collection_error(
        self, mock_retriever_class, test_client, query_request, auth_headers
    ):
        """Test error handling in vector store querying (setics_collection)"""
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.side_effect = Exception("Vector store query failed")
        mock_retriever_class.return_value = mock_retriever_instance

        response = test_client.post(
            "/v1/retriever/setics_collection/invoke",
            json=query_request,
            headers=auth_headers,
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Error querying vector store"

    @pytest.mark.asyncio
    @patch("src.routes.retriever_router.MultiQRerankedRetriever")
    async def test_query_setics_collection_with_empty_results(
        self, mock_retriever_class, test_client, query_request, auth_headers
    ):
        """Test when retriever returns no documents (setics_collection)"""
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.return_value = []
        mock_retriever_class.return_value = mock_retriever_instance

        response = test_client.post(
            "/v1/retriever/setics_collection/invoke",
            json=query_request,
            headers=auth_headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["documents"]) == 0

    @pytest.mark.asyncio
    async def test_setics_collection_invalid_query_format(
        self, test_client, auth_headers
    ):
        """Test with invalid query format (setics_collection)"""
        response = test_client.post(
            "/v1/retriever/setics_collection/invoke", json={}, headers=auth_headers
        )
        assert response.status_code == 422
        assert "field required" in response.text.lower()
