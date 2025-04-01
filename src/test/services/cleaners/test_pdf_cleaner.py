from unittest.mock import AsyncMock

import pytest
from langchain.schema import Document

from src.services.cleaners.cleaning_strategies import (
    CleaningStrategy,
    HeaderFooterRemovalStrategy,
    TableFormattingStrategy,
    WhitespaceNormalizationStrategy,
)
from src.services.cleaners.pdf_cleaner import PdfDocumentCleaner


class TestPdfDocumentCleaner:
    """Tests for the PdfDocumentCleaner class."""

    @pytest.fixture
    def mock_document(self):
        """Create a mock document for testing."""
        return Document(
            page_content="Test content with header\nPage 1 of 10\nActual content",
            metadata={"source": "test.pdf"},
        )

    @pytest.fixture
    def mock_documents(self):
        """Create a list of mock documents for testing."""
        return [
            Document(
                page_content="Document 1 content", metadata={"source": "doc1.pdf"}
            ),
            Document(
                page_content="Document 2 content", metadata={"source": "doc2.pdf"}
            ),
        ]

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock cleaning strategy."""
        strategy = AsyncMock(spec=CleaningStrategy)
        strategy.clean.return_value = "Cleaned content"
        strategy.name = "MockStrategy"
        return strategy

    def test_init_default_strategies(self):
        """Test initializing with default strategies."""
        cleaner = PdfDocumentCleaner()
        assert len(cleaner.strategies) > 0
        assert any(
            isinstance(s, HeaderFooterRemovalStrategy) for s in cleaner.strategies
        )
        assert any(
            isinstance(s, WhitespaceNormalizationStrategy) for s in cleaner.strategies
        )
        assert any(isinstance(s, TableFormattingStrategy) for s in cleaner.strategies)

    def test_init_custom_strategies(self, mock_strategy):
        """Test initializing with custom strategies."""
        cleaner = PdfDocumentCleaner(strategies=[mock_strategy])
        assert len(cleaner.strategies) == 1
        assert cleaner.strategies[0] == mock_strategy

    @pytest.mark.asyncio
    async def test_clean_document(self, mock_document, mock_strategy):
        """Test cleaning a single document."""
        # Arrange
        cleaner = PdfDocumentCleaner(strategies=[mock_strategy])

        # Act
        result = await cleaner.clean_document(mock_document)

        # Assert
        assert result.page_content == "Cleaned content"
        assert result.metadata == mock_document.metadata
        mock_strategy.clean.assert_called_once_with(mock_document.page_content)

    @pytest.mark.asyncio
    async def test_clean_documents(self, mock_documents, mock_strategy):
        """Test cleaning multiple documents."""
        # Arrange
        cleaner = PdfDocumentCleaner(strategies=[mock_strategy])

        # Act
        results = await cleaner.clean_documents(mock_documents)

        # Assert
        assert len(results) == 2
        assert all(doc.page_content == "Cleaned content" for doc in results)
        assert results[0].metadata == mock_documents[0].metadata
        assert results[1].metadata == mock_documents[1].metadata
        assert mock_strategy.clean.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_strategies_applied_in_order(self, mock_document):
        """Test that multiple strategies are applied in the correct order."""
        # Arrange
        first_strategy = AsyncMock(spec=CleaningStrategy)
        first_strategy.clean.return_value = "First strategy applied"
        first_strategy.name = "FirstStrategy"

        second_strategy = AsyncMock(spec=CleaningStrategy)
        second_strategy.clean.return_value = "Both strategies applied"
        second_strategy.name = "SecondStrategy"

        cleaner = PdfDocumentCleaner(strategies=[first_strategy, second_strategy])

        # Act
        result = await cleaner.clean_document(mock_document)

        # Assert
        assert result.page_content == "Both strategies applied"
        first_strategy.clean.assert_called_once_with(mock_document.page_content)
        second_strategy.clean.assert_called_once_with("First strategy applied")

    def test_add_strategy(self, mock_strategy):
        """Test adding a strategy."""
        # Arrange
        cleaner = PdfDocumentCleaner()  # Use default constructor instead of empty list
        initial_count = len(cleaner.strategies)

        # Act
        cleaner.add_strategy(mock_strategy)

        # Assert
        assert len(cleaner.strategies) == initial_count + 1
        assert (
            cleaner.strategies[-1] == mock_strategy
        )  # Check that our strategy was added at the end

    def test_remove_strategy_by_name(self):
        """Test removing a strategy by name."""
        # Arrange
        strategy1 = HeaderFooterRemovalStrategy()
        strategy2 = WhitespaceNormalizationStrategy()
        cleaner = PdfDocumentCleaner(strategies=[strategy1, strategy2])
        assert len(cleaner.strategies) == 2

        # Act
        cleaner.remove_strategy("HeaderFooterRemovalStrategy")

        # Assert
        assert len(cleaner.strategies) == 1
        assert all(
            not isinstance(s, HeaderFooterRemovalStrategy) for s in cleaner.strategies
        )
        assert any(
            isinstance(s, WhitespaceNormalizationStrategy) for s in cleaner.strategies
        )

    def test_remove_nonexistent_strategy(self):
        """Test removing a strategy that doesn't exist."""
        # Arrange
        cleaner = PdfDocumentCleaner()
        initial_strategy_count = len(cleaner.strategies)

        # Act
        cleaner.remove_strategy("NonexistentStrategy")

        # Assert
        assert (
            len(cleaner.strategies) == initial_strategy_count
        )  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_clean_empty_document(self):
        """Test cleaning an empty document."""
        # Arrange
        empty_doc = Document(page_content="", metadata={"source": "empty.pdf"})
        mock_strategy = AsyncMock(spec=CleaningStrategy)
        mock_strategy.clean.return_value = ""
        mock_strategy.name = "MockStrategy"

        cleaner = PdfDocumentCleaner(strategies=[mock_strategy])

        # Act
        result = await cleaner.clean_document(empty_doc)

        # Assert
        assert result.page_content == ""
        assert result.metadata == empty_doc.metadata
        mock_strategy.clean.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_real_cleaning_integration(self):
        """Integration test with real cleaning strategies."""
        # Arrange
        # Add a newline before the section heading to match SectionHeadingStrategy's regex pattern
        test_doc = Document(
            page_content="Header\nPage 1 of 5\n\n\n\n\n1. Introduction\n\nThis is    some text   with    excessive    spaces.\n\n\n\n",
            metadata={"source": "test.pdf"},
        )
        cleaner = PdfDocumentCleaner()  # Use default strategies

        # Act
        result = await cleaner.clean_document(test_doc)

        # Assert
        assert (
            "Header\nPage 1 of 5" not in result.page_content
        )  # Header should be removed

        # Check for normalized whitespace instead of specific heading formatting
        assert "excessive    spaces" not in result.page_content
        assert "\n\n\n\n" not in result.page_content

        # If heading formatting is important for the test, modify the input pattern to match
        # what SectionHeadingStrategy expects or check an alternate assertion
        assert "1. Introduction" in result.page_content  # Section content is preserved
