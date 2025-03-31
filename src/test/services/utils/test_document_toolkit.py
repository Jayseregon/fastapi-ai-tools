import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain.schema import Document

from src.services.utils.document_toolkit import DocumentJsonToolkit


class TestDocumentJsonToolkit:

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Test content 1", metadata={"source": "test1"}),
            Document(page_content="Test content 2", metadata={"source": "test2"}),
        ]

    def test_documents_to_json(self, tmp_path, sample_documents):
        """Test converting documents to JSON file."""
        # Arrange
        file_path = tmp_path / "test_output.json"

        # Act
        DocumentJsonToolkit.documents_to_json(sample_documents, file_path)

        # Assert
        assert file_path.exists()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["page_content"] == "Test content 1"
            assert data[0]["metadata"]["source"] == "test1"
            assert data[1]["page_content"] == "Test content 2"
            assert data[1]["metadata"]["source"] == "test2"

    def test_documents_to_json_string_path(self, tmp_path, sample_documents):
        """Test converting documents to JSON using string path."""
        # Arrange
        file_path = str(tmp_path / "test_output.json")

        # Act
        DocumentJsonToolkit.documents_to_json(sample_documents, file_path)

        # Assert
        assert Path(file_path).exists()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 2

    def test_documents_to_json_creates_directories(self, tmp_path, sample_documents):
        """Test that documents_to_json creates parent directories if needed."""
        # Arrange
        nested_dir = tmp_path / "nested" / "path"
        file_path = nested_dir / "test_output.json"

        # Act
        DocumentJsonToolkit.documents_to_json(sample_documents, file_path)

        # Assert
        assert file_path.exists()

    def test_documents_to_json_permission_error(self, sample_documents):
        """Test handling of permission error while writing JSON file."""
        # Arrange
        with patch(
            "pathlib.Path.open", side_effect=PermissionError("Permission denied")
        ):
            # Act & Assert
            with pytest.raises(
                RuntimeError,
                match="Error while exporting JSON file:.*Permission denied",
            ):
                DocumentJsonToolkit.documents_to_json(
                    sample_documents, "test_file.json"
                )

    def test_documents_to_json_io_error(self, sample_documents):
        """Test handling of IO error while writing JSON file."""
        # Arrange
        with patch("pathlib.Path.open", side_effect=IOError("IO Error")):
            # Act & Assert
            with pytest.raises(
                RuntimeError, match="Error while exporting JSON file:.*IO Error"
            ):
                DocumentJsonToolkit.documents_to_json(
                    sample_documents, "test_file.json"
                )

    def test_json_to_documents(self, tmp_path, sample_documents):
        """Test converting JSON file to documents."""
        # Arrange
        file_path = tmp_path / "test_input.json"
        DocumentJsonToolkit.documents_to_json(sample_documents, file_path)

        # Act
        results = DocumentJsonToolkit.json_to_documents(file_path)

        # Assert
        assert len(results) == 2
        assert results[0].page_content == "Test content 1"
        assert results[0].metadata["source"] == "test1"
        assert results[1].page_content == "Test content 2"
        assert results[1].metadata["source"] == "test2"

    def test_json_to_documents_string_path(self, tmp_path, sample_documents):
        """Test converting JSON file to documents using string path."""
        # Arrange
        file_path = tmp_path / "test_input.json"
        DocumentJsonToolkit.documents_to_json(sample_documents, file_path)

        # Act
        results = DocumentJsonToolkit.json_to_documents(str(file_path))

        # Assert
        assert len(results) == 2

    def test_json_to_documents_permission_error(self):
        """Test handling of permission error while reading JSON file."""
        # Arrange
        with patch(
            "pathlib.Path.open", side_effect=PermissionError("Permission denied")
        ):
            # Act & Assert
            with pytest.raises(
                RuntimeError,
                match="Error while importing JSON file:.*Permission denied",
            ):
                DocumentJsonToolkit.json_to_documents("test_file.json")

    def test_json_to_documents_io_error(self):
        """Test handling of IO error while reading JSON file."""
        # Arrange
        with patch("pathlib.Path.open", side_effect=IOError("IO Error")):
            # Act & Assert
            with pytest.raises(
                RuntimeError, match="Error while importing JSON file:.*IO Error"
            ):
                DocumentJsonToolkit.json_to_documents("test_file.json")
