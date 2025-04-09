from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel

from src.services.loaders.files.base_document_loader import BaseDocumentLoader


class ConcreteDocumentLoader(BaseDocumentLoader):
    """Concrete implementation of BaseDocumentLoader for testing."""

    async def initialize(self, llm_model=None, **kwargs):
        self._llm_model = llm_model
        self._initialized = True
        return self

    async def load_document(self, file_path):
        if not self._initialized:
            raise RuntimeError("Loader not initialized")
        return [
            Document(page_content="Test content", metadata={"source": str(file_path)})
        ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM model for testing."""
    mock_model = MagicMock(spec=BaseChatModel)
    return mock_model


@pytest.mark.asyncio
async def test_initialization():
    """Test the base state of the document loader after initialization."""
    loader = ConcreteDocumentLoader()
    assert loader._initialized is False
    assert loader._llm_model is None


@pytest.mark.asyncio
async def test_initialize(mock_llm):
    """Test initialization with and without LLM model."""
    # Test with LLM model
    loader = ConcreteDocumentLoader()
    initialized_loader = await loader.initialize(llm_model=mock_llm)
    assert initialized_loader is loader  # Should return self
    assert loader._initialized is True
    assert loader._llm_model is mock_llm

    # Test without LLM model
    loader = ConcreteDocumentLoader()
    initialized_loader = await loader.initialize()
    assert loader._initialized is True
    assert loader._llm_model is None


@pytest.mark.asyncio
async def test_context_manager():
    """Test the async context manager functionality."""
    loader = ConcreteDocumentLoader()

    # Mock the initialize and close methods
    loader.initialize = AsyncMock(return_value=loader)
    loader.close = AsyncMock()

    async with loader as context_loader:
        assert context_loader is loader
        loader.initialize.assert_called_once()

    # After exiting context, close should be called
    loader.close.assert_called_once()


@pytest.mark.asyncio
async def test_load_document():
    """Test loading a document."""
    loader = ConcreteDocumentLoader()
    await loader.initialize()

    test_path = Path("/test/file.txt")
    documents = await loader.load_document(test_path)

    assert len(documents) == 1
    assert documents[0].page_content == "Test content"
    assert documents[0].metadata["source"] == str(test_path)


@pytest.mark.asyncio
async def test_load_document_not_initialized():
    """Test loading a document before initialization."""
    loader = ConcreteDocumentLoader()

    # Override the concrete implementation to test the behavior
    async def failing_load(self, file_path):
        if not self._initialized:
            raise RuntimeError("Loader not initialized")
        return []

    with patch.object(ConcreteDocumentLoader, "load_document", failing_load):
        with pytest.raises(RuntimeError, match="Loader not initialized"):
            await loader.load_document(Path("/test/file.txt"))


@pytest.mark.asyncio
async def test_close():
    """Test resource cleanup."""
    loader = ConcreteDocumentLoader()
    await loader.initialize(llm_model=MagicMock())

    assert loader._initialized is True
    assert loader._llm_model is not None

    await loader.close()

    assert loader._initialized is False
    assert loader._llm_model is None
