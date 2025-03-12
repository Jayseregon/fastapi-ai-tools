import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.services.loaders.files.pdf_loader import PdfLoader, create_pdf_loader


@pytest.fixture
def mock_llm():
    """Create a mock LLM model for testing."""
    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.__class__.__name__ = "MockLLM"
    return mock_model


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Page 1 content", metadata={"page": 1, "source": "test.pdf"}
        ),
        Document(
            page_content="Page 2 content", metadata={"page": 2, "source": "test.pdf"}
        ),
    ]


@pytest.mark.asyncio
async def test_initialization_with_default_model():
    """Test initialization with default model."""
    with patch("src.services.loaders.files.pdf_loader.ChatOpenAI") as mock_chat:
        mock_chat.return_value = MagicMock(spec=ChatOpenAI)

        loader = PdfLoader()
        await loader.initialize()

        assert loader._initialized is True
        assert loader._llm_model is not None
        mock_chat.assert_called_once()


@pytest.mark.asyncio
async def test_initialization_with_custom_model(mock_llm):
    """Test initialization with custom model."""
    loader = PdfLoader()
    await loader.initialize(llm_model=mock_llm)

    assert loader._initialized is True
    assert loader._llm_model is mock_llm


@pytest.mark.asyncio
async def test_load_document_with_valid_pdf(mock_llm, sample_documents):
    """Test loading a valid PDF document."""
    # Mock the PyMuPDFLoader
    mock_pymupdf_loader = AsyncMock(spec=PyMuPDFLoader)
    mock_pymupdf_loader.aload.return_value = sample_documents

    with (
        patch(
            "src.services.loaders.files.pdf_loader.PyMuPDFLoader",
            return_value=mock_pymupdf_loader,
        ) as mock_loader,
        patch.object(PdfLoader, "_is_valid_pdf", return_value=True),
    ):

        loader = PdfLoader(llm_model=mock_llm)
        await loader.initialize()

        test_path = Path("/test/file.pdf")
        documents = await loader.load_document(test_path)

        assert documents == sample_documents
        mock_loader.assert_called_once()
        mock_pymupdf_loader.aload.assert_called_once()


@pytest.mark.asyncio
async def test_load_document_auto_initializes(mock_llm, sample_documents):
    """Test that load_document auto-initializes if not already initialized."""
    mock_pymupdf_loader = AsyncMock(spec=PyMuPDFLoader)
    mock_pymupdf_loader.aload.return_value = sample_documents

    # Fix: Use a simpler approach to test auto-initialization
    with (
        patch(
            "src.services.loaders.files.pdf_loader.PyMuPDFLoader",
            return_value=mock_pymupdf_loader,
        ),
        patch.object(PdfLoader, "_is_valid_pdf", return_value=True),
    ):

        loader = PdfLoader(llm_model=mock_llm)
        assert loader._initialized is False  # Verify not initialized

        test_path = Path("/test/file.pdf")
        documents = await loader.load_document(test_path)

        assert documents == sample_documents
        assert loader._initialized is True  # Should be initialized after call


@pytest.mark.asyncio
async def test_load_document_with_invalid_pdf(mock_llm):
    """Test loading an invalid PDF document."""
    with patch.object(PdfLoader, "_is_valid_pdf", return_value=False):
        loader = PdfLoader(llm_model=mock_llm)
        await loader.initialize()

        test_path = Path("/test/invalid.pdf")

        with pytest.raises(
            ValueError, match=f"Invalid or inaccessible PDF file: {test_path}"
        ):
            await loader.load_document(test_path)


@pytest.mark.asyncio
async def test_is_valid_pdf_for_existing_pdf():
    """Test PDF validation for an existing PDF file."""
    # Mock file existence and PDF header check
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.suffix = ".pdf"
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = b"%PDF-1.7 test content"

    # Mock fitz.open functionality
    mock_doc = MagicMock()
    mock_doc.page_count = 2
    mock_doc.is_encrypted = False
    mock_doc.metadata = {"title": "Test PDF"}
    mock_doc.__getitem__.return_value = "test page"

    with (
        patch("builtins.open", return_value=mock_file),
        patch("fitz.open", return_value=mock_doc),
    ):

        loader = PdfLoader()
        result = await loader._is_valid_pdf(mock_path)

        assert result is True
        mock_path.exists.assert_called_once()
        mock_doc.close.assert_called()


@pytest.mark.asyncio
async def test_is_valid_pdf_for_nonexistent_file():
    """Test PDF validation for a nonexistent file."""
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = False

    loader = PdfLoader()
    result = await loader._is_valid_pdf(mock_path)

    assert result is False
    mock_path.exists.assert_called_once()


@pytest.mark.asyncio
async def test_is_valid_pdf_for_invalid_signature():
    """Test PDF validation for a file with invalid PDF signature."""
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.suffix = ".pdf"
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = b"Not a PDF file"

    with patch("builtins.open", return_value=mock_file):
        loader = PdfLoader()
        result = await loader._is_valid_pdf(mock_path)

        assert result is False


@pytest.mark.asyncio
async def test_is_valid_pdf_for_encrypted_pdf():
    """Test PDF validation for an encrypted PDF file."""
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.suffix = ".pdf"
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = b"%PDF-1.7 test content"

    mock_doc = MagicMock()
    mock_doc.page_count = 2
    mock_doc.is_encrypted = True

    with (
        patch("builtins.open", return_value=mock_file),
        patch("fitz.open", return_value=mock_doc),
    ):

        loader = PdfLoader()
        result = await loader._is_valid_pdf(mock_path)

        assert result is False
        mock_doc.close.assert_called_once()


@pytest.mark.asyncio
async def test_documents_to_json(mock_llm, sample_documents, tmp_path):
    """Test exporting documents to JSON file."""
    test_file = tmp_path / "test_output.json"

    # Fix: Create a real file and verify its contents instead of mocking
    loader = PdfLoader(llm_model=mock_llm)
    await loader.initialize()
    await loader.documents_to_json(sample_documents, test_file)

    # Verify file was created and contains expected content
    assert test_file.exists()
    with open(test_file, "r") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["page_content"] == "Page 1 content"
        assert data[0]["metadata"]["page"] == 1
        assert data[1]["page_content"] == "Page 2 content"
        assert data[1]["metadata"]["page"] == 2


@pytest.mark.asyncio
async def test_documents_to_json_permission_error(mock_llm, sample_documents, tmp_path):
    """Test error handling during JSON export."""
    # Fix: Use tmp_path and mock a permission error using a side effect
    test_file = tmp_path / "test_output.json"

    # Mock the Path.open method to raise PermissionError
    with patch.object(Path, "open", side_effect=PermissionError("Permission denied")):
        loader = PdfLoader(llm_model=mock_llm)
        await loader.initialize()

        with pytest.raises(RuntimeError, match="Error while exporting json file"):
            await loader.documents_to_json(sample_documents, test_file)


@pytest.mark.asyncio
async def test_json_to_documents(mock_llm, tmp_path):
    """Test importing documents from JSON file."""
    # Fix: Create actual JSON file for testing
    test_file = tmp_path / "test_input.json"

    # Create expected JSON content
    json_content = [
        {
            "page_content": "Page 1 content",
            "metadata": {"page": 1, "source": "test.pdf"},
        },
        {
            "page_content": "Page 2 content",
            "metadata": {"page": 2, "source": "test.pdf"},
        },
    ]

    # Write test data to a real file
    with open(test_file, "w") as f:
        json.dump(json_content, f)

    loader = PdfLoader(llm_model=mock_llm)
    await loader.initialize()

    documents = await loader.json_to_documents(test_file)

    assert len(documents) == 2
    assert documents[0].page_content == "Page 1 content"
    assert documents[0].metadata["page"] == 1
    assert documents[1].page_content == "Page 2 content"
    assert documents[1].metadata["page"] == 2


@pytest.mark.asyncio
async def test_json_to_documents_file_error(mock_llm, tmp_path):
    """Test error handling during JSON import."""
    # Fix: Use tmp_path and non-existent subpath
    test_file = tmp_path / "nonexistent" / "file.json"

    loader = PdfLoader(llm_model=mock_llm)
    await loader.initialize()

    with pytest.raises(RuntimeError, match="Error while importing json file"):
        await loader.json_to_documents(test_file)


@pytest.mark.asyncio
async def test_create_pdf_loader_factory():
    """Test the PDF loader factory function."""
    with patch.object(PdfLoader, "initialize") as mock_init:
        mock_init.return_value = AsyncMock()

        loader = await create_pdf_loader()

        assert isinstance(loader, PdfLoader)
        mock_init.assert_called_once()
