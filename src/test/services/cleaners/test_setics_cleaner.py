from unittest.mock import AsyncMock

import pytest
from langchain.schema import Document

from src.services.cleaners.cleaning_strategies import (
    CleaningStrategy,
    SeticsHeadingCleanupStrategy,
    SeticsTableFormattingStrategy,
    SeticsWebCleanupStrategy,
    WhitespaceNormalizationStrategy,
)
from src.services.cleaners.setics_cleaner import SeticsDocumentCleaner


class TestSeticsDocumentCleaner:
    """Tests for the SeticsDocumentCleaner class."""

    @pytest.fixture
    def mock_document(self):
        """Create a mock document for testing."""
        return Document(
            page_content="Setics User Manual - Version 1.2\nSetics Sttar Advanced Designer | User Manual Version 1.2\nActual content",
            metadata={"source": "setics_manual.html"},
        )

    @pytest.fixture
    def mock_documents(self):
        """Create a list of mock documents for testing."""
        return [
            Document(
                page_content="Document 1 content\nEnglish\n\n\nFrançais\n\nMore content",
                metadata={"source": "setics1.html"},
            ),
            Document(
                page_content="Document 2 content\nTable of Contents\n\nEntry 1\nEntry 2",
                metadata={"source": "setics2.html"},
            ),
        ]

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock cleaning strategy."""
        strategy = AsyncMock(spec=CleaningStrategy)
        strategy.clean.return_value = "Cleaned Setics content"
        strategy.name = "MockSeticsStrategy"
        return strategy

    def test_init_default_strategies(self):
        """Test initializing with default strategies."""
        cleaner = SeticsDocumentCleaner()
        assert len(cleaner.strategies) > 0
        assert any(isinstance(s, SeticsWebCleanupStrategy) for s in cleaner.strategies)
        assert any(
            isinstance(s, WhitespaceNormalizationStrategy) for s in cleaner.strategies
        )

    def test_init_custom_strategies(self, mock_strategy):
        """Test initializing with custom strategies."""
        cleaner = SeticsDocumentCleaner(strategies=[mock_strategy])
        assert len(cleaner.strategies) == 1
        assert cleaner.strategies[0] == mock_strategy

    @pytest.mark.asyncio
    async def test_clean_document(self, mock_document, mock_strategy):
        """Test cleaning a single document."""
        # Arrange
        cleaner = SeticsDocumentCleaner(strategies=[mock_strategy])

        # Act
        result = await cleaner.clean_document(mock_document)

        # Assert
        assert result.page_content == "Cleaned Setics content"
        assert result.metadata == mock_document.metadata
        mock_strategy.clean.assert_called_once_with(mock_document.page_content)

    @pytest.mark.asyncio
    async def test_clean_documents(self, mock_documents, mock_strategy):
        """Test cleaning multiple documents."""
        # Arrange
        cleaner = SeticsDocumentCleaner(strategies=[mock_strategy])

        # Act
        results = await cleaner.clean_documents(mock_documents)

        # Assert
        assert len(results) == 2
        assert all(doc.page_content == "Cleaned Setics content" for doc in results)
        assert results[0].metadata == mock_documents[0].metadata
        assert results[1].metadata == mock_documents[1].metadata
        assert mock_strategy.clean.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_strategies_applied_in_order(self, mock_document):
        """Test that multiple strategies are applied in the correct order."""
        # Arrange
        first_strategy = AsyncMock(spec=CleaningStrategy)
        first_strategy.clean.return_value = "First Setics strategy applied"
        first_strategy.name = "FirstSeticsStrategy"

        second_strategy = AsyncMock(spec=CleaningStrategy)
        second_strategy.clean.return_value = "Both Setics strategies applied"
        second_strategy.name = "SecondSeticsStrategy"

        cleaner = SeticsDocumentCleaner(strategies=[first_strategy, second_strategy])

        # Act
        result = await cleaner.clean_document(mock_document)

        # Assert
        assert result.page_content == "Both Setics strategies applied"
        first_strategy.clean.assert_called_once_with(mock_document.page_content)
        second_strategy.clean.assert_called_once_with("First Setics strategy applied")

    def test_add_strategy(self, mock_strategy):
        """Test adding a strategy."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        initial_count = len(cleaner.strategies)

        # Act
        cleaner.add_strategy(mock_strategy)

        # Assert
        assert len(cleaner.strategies) == initial_count + 1
        assert cleaner.strategies[-1] == mock_strategy

    def test_remove_strategy_by_name(self):
        """Test removing a strategy by name."""
        # Arrange
        strategy1 = SeticsWebCleanupStrategy()
        strategy2 = WhitespaceNormalizationStrategy()
        cleaner = SeticsDocumentCleaner(strategies=[strategy1, strategy2])
        assert len(cleaner.strategies) == 2

        # Act
        cleaner.remove_strategy("SeticsWebCleanupStrategy")

        # Assert
        assert len(cleaner.strategies) == 1
        assert all(
            not isinstance(s, SeticsWebCleanupStrategy) for s in cleaner.strategies
        )
        assert any(
            isinstance(s, WhitespaceNormalizationStrategy) for s in cleaner.strategies
        )

    def test_remove_nonexistent_strategy(self):
        """Test removing a strategy that doesn't exist."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        initial_strategy_count = len(cleaner.strategies)

        # Act
        cleaner.remove_strategy("NonexistentStrategy")

        # Assert
        assert len(cleaner.strategies) == initial_strategy_count

    @pytest.mark.asyncio
    async def test_clean_empty_document(self):
        """Test cleaning an empty document."""
        # Arrange
        empty_doc = Document(page_content="", metadata={"source": "empty_setics.html"})
        mock_strategy = AsyncMock(spec=CleaningStrategy)
        mock_strategy.clean.return_value = ""
        mock_strategy.name = "MockStrategy"

        cleaner = SeticsDocumentCleaner(strategies=[mock_strategy])

        # Act
        result = await cleaner.clean_document(empty_doc)

        # Assert
        assert result.page_content == ""
        assert result.metadata == empty_doc.metadata
        mock_strategy.clean.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_uncomment_additional_strategies(self):
        """Test adding commented-out strategies from the source file."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        initial_count = len(cleaner.strategies)

        # Act
        cleaner.add_strategy(SeticsTableFormattingStrategy())
        cleaner.add_strategy(SeticsHeadingCleanupStrategy())

        # Assert
        assert len(cleaner.strategies) == initial_count + 2
        assert any(
            isinstance(s, SeticsTableFormattingStrategy) for s in cleaner.strategies
        )
        assert any(
            isinstance(s, SeticsHeadingCleanupStrategy) for s in cleaner.strategies
        )

    @pytest.mark.asyncio
    async def test_real_cleaning_integration(self):
        """Integration test with real cleaning strategies."""
        # Arrange
        test_doc = Document(
            page_content=(
                "User Manual - Version 1.2\n"
                "Setics Sttar Advanced Designer | User Manual Version 1.2\n\n"
                "Content with    excessive    spaces\n\n\n\n"
                "English\n\n\nFrançais\n\n"
                "Need more help with this? Support & Assistance"
            ),
            metadata={"source": "setics_test.html"},
        )
        cleaner = SeticsDocumentCleaner()  # Use default strategies

        # Act
        result = await cleaner.clean_document(test_doc)

        # Assert
        # Check that whitespace is normalized
        assert "excessive    spaces" not in result.page_content
        assert "\n\n\n\n" not in result.page_content

        # Content is preserved but cleaned
        assert "Content" in result.page_content
