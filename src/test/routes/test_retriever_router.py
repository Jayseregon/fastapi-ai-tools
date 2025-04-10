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
    async def test_query_vector_store_success(
        self,
        mock_retriever_class,
        test_client,
        query_request,
        sample_langchain_documents,
    ):
        """Test successful vector store querying"""
        # Setup retriever mock
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.return_value = sample_langchain_documents
        mock_retriever_class.return_value = mock_retriever_instance

        # Use test client to call the endpoint
        response = test_client.post("/v1/retriever/invoke", json=query_request)

        # Verify response
        assert response.status_code == 200
        response_data = response.json()

        # Check that documents are returned correctly
        assert len(response_data["documents"]) == 2

        # Verify first document
        assert (
            response_data["documents"][0]["page_content"]
            == "AI is a field of computer science that focuses on creating intelligent machines."
        )
        assert response_data["documents"][0]["metadata"]["source"] == "ai_textbook.pdf"
        assert response_data["documents"][0]["metadata"]["document_type"] == "textbook"
        assert response_data["documents"][0]["metadata"]["id"] == "doc1"

        # Verify retriever was called with correct parameters
        mock_retriever_instance.assert_called_once_with(
            query=query_request["query"], collection_name=config.COLLECTION_NAME
        )

    @pytest.mark.asyncio
    @patch("src.routes.retriever_router.MultiQRerankedRetriever")
    async def test_query_vector_store_error(
        self, mock_retriever_class, test_client, query_request
    ):
        """Test error handling in vector store querying"""
        # Setup retriever mock to raise exception
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.side_effect = Exception("Vector store query failed")
        mock_retriever_class.return_value = mock_retriever_instance

        # Use test client to call the endpoint
        response = test_client.post("/v1/retriever/invoke", json=query_request)

        # Verify error response
        assert response.status_code == 500
        assert response.json()["detail"] == "Error querying vector store"

    @pytest.mark.asyncio
    @patch("src.routes.retriever_router.MultiQRerankedRetriever")
    async def test_query_with_empty_results(
        self, mock_retriever_class, test_client, query_request
    ):
        """Test when retriever returns no documents"""
        # Setup retriever mock to return empty list
        mock_retriever_instance = AsyncMock()
        mock_retriever_instance.return_value = []
        mock_retriever_class.return_value = mock_retriever_instance

        # Use test client to call the endpoint
        response = test_client.post("/v1/retriever/invoke", json=query_request)

        # Verify response has empty documents list
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["documents"]) == 0

    @pytest.mark.asyncio
    async def test_invalid_query_format(self, test_client):
        """Test with invalid query format"""
        # Use test client to call the endpoint with invalid JSON
        response = test_client.post("/v1/retriever/invoke", json={})

        # Verify response indicates validation error
        assert response.status_code == 422
        assert "field required" in response.text.lower()
