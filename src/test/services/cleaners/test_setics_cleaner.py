from unittest.mock import AsyncMock

import pytest
from langchain.schema import Document

from src.services.cleaners.cleaning_strategies import (
    CleaningStrategy,
    SeticsWebCleanupStrategy,
    SeticsWebCleanupStrategyFR,
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
    def mock_document_fr(self):
        """Create a mock French document for testing."""
        return Document(
            page_content="Setics Manuel Utilisateur - Version 1.2\nSetics Sttar Advanced Designer | Manuel Utilisateur Version 1.2\nContenu actuel",
            metadata={"source": "setics_manual_fr.html", "language": "fr"},
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
                metadata={"source": "setics2.html", "language": "fr"},
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

        # Check default language strategies
        assert "default" in cleaner.language_strategies
        assert "fr" in cleaner.language_strategies

        # Check default strategies
        default_strategies = cleaner.language_strategies["default"]
        assert len(default_strategies) > 0
        assert any(isinstance(s, SeticsWebCleanupStrategy) for s in default_strategies)
        assert any(
            isinstance(s, WhitespaceNormalizationStrategy) for s in default_strategies
        )

        # Check FR strategies
        fr_strategies = cleaner.language_strategies["fr"]
        assert len(fr_strategies) > 0
        assert any(isinstance(s, SeticsWebCleanupStrategyFR) for s in fr_strategies)
        assert any(
            isinstance(s, WhitespaceNormalizationStrategy) for s in fr_strategies
        )

        # Check custom strategies
        assert len(cleaner.custom_strategies) == 0

    def test_get_strategies_for_language(self):
        """Test getting strategies for specific languages."""
        cleaner = SeticsDocumentCleaner()

        # Default language
        default_strategies = cleaner.get_strategies_for_language("default")
        assert any(isinstance(s, SeticsWebCleanupStrategy) for s in default_strategies)

        # French language
        fr_strategies = cleaner.get_strategies_for_language("fr")
        assert any(isinstance(s, SeticsWebCleanupStrategyFR) for s in fr_strategies)

        # Unknown language should return default
        unknown_strategies = cleaner.get_strategies_for_language("unknown")
        assert any(isinstance(s, SeticsWebCleanupStrategy) for s in unknown_strategies)

    @pytest.mark.asyncio
    async def test_clean_document_default_language(self, mock_document, mock_strategy):
        """Test cleaning a single document with default language."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        cleaner.add_strategy(mock_strategy)  # Add to custom strategies

        # Act
        result = await cleaner.clean_document(mock_document)

        # Assert
        assert result.metadata == mock_document.metadata
        mock_strategy.clean.assert_called_once()

    @pytest.mark.asyncio
    async def test_clean_document_french(self, mock_document_fr, mock_strategy):
        """Test cleaning a document with French language."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        cleaner.add_strategy(mock_strategy, language="fr")

        # Act
        result = await cleaner.clean_document(mock_document_fr)

        # Assert
        assert result.metadata == mock_document_fr.metadata
        mock_strategy.clean.assert_called_once()

    @pytest.mark.asyncio
    async def test_clean_documents(self, mock_documents):
        """Test cleaning multiple documents with different languages."""
        # Arrange
        cleaner = SeticsDocumentCleaner()

        # Create mock strategies for tracking calls
        default_mock = AsyncMock(spec=CleaningStrategy)
        default_mock.clean.return_value = "Default cleaned"
        default_mock.name = "DefaultMock"

        fr_mock = AsyncMock(spec=CleaningStrategy)
        fr_mock.clean.return_value = "French cleaned"
        fr_mock.name = "FrenchMock"

        cleaner.add_strategy(default_mock, language="default")
        cleaner.add_strategy(fr_mock, language="fr")

        # Act
        results = await cleaner.clean_documents(mock_documents)

        # Assert
        assert len(results) == 2
        default_mock.clean.assert_called_once()
        fr_mock.clean.assert_called_once()

    def test_add_strategy_to_language(self, mock_strategy):
        """Test adding a strategy to a specific language."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        initial_fr_count = len(cleaner.language_strategies["fr"])

        # Act
        cleaner.add_strategy(mock_strategy, language="fr")

        # Assert
        assert len(cleaner.language_strategies["fr"]) == initial_fr_count + 1
        assert cleaner.language_strategies["fr"][-1] == mock_strategy

    def test_add_strategy_custom(self, mock_strategy):
        """Test adding a strategy to custom strategies."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        initial_count = len(cleaner.custom_strategies)

        # Act
        cleaner.add_strategy(mock_strategy)

        # Assert
        assert len(cleaner.custom_strategies) == initial_count + 1
        assert cleaner.custom_strategies[-1] == mock_strategy

    def test_add_strategy_new_language(self, mock_strategy):
        """Test adding a strategy to a new language."""
        # Arrange
        cleaner = SeticsDocumentCleaner()

        # Act
        cleaner.add_strategy(mock_strategy, language="es")

        # Assert
        assert "es" in cleaner.language_strategies
        assert len(cleaner.language_strategies["es"]) == 1
        assert cleaner.language_strategies["es"][0] == mock_strategy

    def test_remove_strategy_by_name_from_language(self):
        """Test removing a strategy by name from a specific language."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        initial_fr_count = len(cleaner.language_strategies["fr"])

        # Act
        cleaner.remove_strategy("SeticsWebCleanupStrategyFR", language="fr")

        # Assert
        assert len(cleaner.language_strategies["fr"]) < initial_fr_count
        assert all(
            not isinstance(s, SeticsWebCleanupStrategyFR)
            for s in cleaner.language_strategies["fr"]
        )

    def test_remove_strategy_by_name_from_custom(self, mock_strategy):
        """Test removing a strategy by name from custom strategies."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        cleaner.add_strategy(mock_strategy)
        initial_count = len(cleaner.custom_strategies)

        # Act
        cleaner.remove_strategy("MockSeticsStrategy")

        # Assert
        assert len(cleaner.custom_strategies) == initial_count - 1
        assert all(s.name != "MockSeticsStrategy" for s in cleaner.custom_strategies)

    def test_remove_nonexistent_strategy(self):
        """Test removing a strategy that doesn't exist."""
        # Arrange
        cleaner = SeticsDocumentCleaner()
        initial_default_count = len(cleaner.language_strategies["default"])
        initial_fr_count = len(cleaner.language_strategies["fr"])
        initial_custom_count = len(cleaner.custom_strategies)

        # Act
        cleaner.remove_strategy("NonexistentStrategy")
        cleaner.remove_strategy("NonexistentStrategy", language="fr")

        # Assert
        assert len(cleaner.language_strategies["default"]) == initial_default_count
        assert len(cleaner.language_strategies["fr"]) == initial_fr_count
        assert len(cleaner.custom_strategies) == initial_custom_count

    @pytest.mark.asyncio
    async def test_clean_empty_document(self):
        """Test cleaning an empty document."""
        # Arrange
        empty_doc = Document(page_content="", metadata={"source": "empty_setics.html"})
        mock_strategy = AsyncMock(spec=CleaningStrategy)
        mock_strategy.clean.return_value = ""
        mock_strategy.name = "MockStrategy"

        cleaner = SeticsDocumentCleaner()
        cleaner.add_strategy(mock_strategy)

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
