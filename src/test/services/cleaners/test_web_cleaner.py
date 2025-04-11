from unittest.mock import AsyncMock

import pytest
from langchain.schema import Document

from src.services.cleaners.cleaning_strategies import (
    CleaningStrategy,
    MarkupRemovalStrategy,
    NavigationMenuRemovalStrategy,
    WebHeaderFooterRemovalStrategy,
    WhitespaceNormalizationStrategy,
)
from src.services.cleaners.web_cleaner import WebDocumentCleaner


class TestWebDocumentCleaner:
    """Tests for the WebDocumentCleaner class."""

    @pytest.fixture
    def mock_document(self):
        """Create a mock document for testing."""
        return Document(
            page_content="<nav>Menu</nav><header>Site Header</header><div>Actual content</div><footer>Site Footer</footer>",
            metadata={"source": "https://example.com"},
        )

    @pytest.fixture
    def mock_documents(self):
        """Create a list of mock documents for testing."""
        return [
            Document(
                page_content="<div>Document 1 content</div>",
                metadata={"source": "https://example.com/doc1"},
            ),
            Document(
                page_content="<div>Document 2 content</div>",
                metadata={"source": "https://example.com/doc2"},
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
        cleaner = WebDocumentCleaner()
        assert len(cleaner.strategies) > 0
        assert any(
            isinstance(s, NavigationMenuRemovalStrategy) for s in cleaner.strategies
        )
        assert any(
            isinstance(s, WebHeaderFooterRemovalStrategy) for s in cleaner.strategies
        )
        assert any(isinstance(s, MarkupRemovalStrategy) for s in cleaner.strategies)
        assert any(
            isinstance(s, WhitespaceNormalizationStrategy) for s in cleaner.strategies
        )

    def test_init_custom_strategies(self, mock_strategy):
        """Test initializing with custom strategies."""
        cleaner = WebDocumentCleaner(strategies=[mock_strategy])
        assert len(cleaner.strategies) == 1
        assert cleaner.strategies[0] == mock_strategy

    @pytest.mark.asyncio
    async def test_clean_document(self, mock_document, mock_strategy):
        """Test cleaning a single document."""
        # Arrange
        cleaner = WebDocumentCleaner(strategies=[mock_strategy])

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
        cleaner = WebDocumentCleaner(strategies=[mock_strategy])

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

        cleaner = WebDocumentCleaner(strategies=[first_strategy, second_strategy])

        # Act
        result = await cleaner.clean_document(mock_document)

        # Assert
        assert result.page_content == "Both strategies applied"
        first_strategy.clean.assert_called_once_with(mock_document.page_content)
        second_strategy.clean.assert_called_once_with("First strategy applied")

    def test_add_strategy(self, mock_strategy):
        """Test adding a strategy."""
        # Arrange
        cleaner = WebDocumentCleaner()
        initial_count = len(cleaner.strategies)

        # Act
        cleaner.add_strategy(mock_strategy)

        # Assert
        assert len(cleaner.strategies) == initial_count + 1
        assert cleaner.strategies[-1] == mock_strategy

    def test_remove_strategy_by_name(self):
        """Test removing a strategy by name."""
        # Arrange
        strategy1 = NavigationMenuRemovalStrategy()
        strategy2 = WhitespaceNormalizationStrategy()
        cleaner = WebDocumentCleaner(strategies=[strategy1, strategy2])
        assert len(cleaner.strategies) == 2

        # Act
        cleaner.remove_strategy("NavigationMenuRemovalStrategy")

        # Assert
        assert len(cleaner.strategies) == 1
        assert all(
            not isinstance(s, NavigationMenuRemovalStrategy) for s in cleaner.strategies
        )
        assert any(
            isinstance(s, WhitespaceNormalizationStrategy) for s in cleaner.strategies
        )

    def test_remove_nonexistent_strategy(self):
        """Test removing a strategy that doesn't exist."""
        # Arrange
        cleaner = WebDocumentCleaner()
        initial_strategy_count = len(cleaner.strategies)

        # Act
        cleaner.remove_strategy("NonexistentStrategy")

        # Assert
        assert len(cleaner.strategies) == initial_strategy_count

    @pytest.mark.asyncio
    async def test_clean_empty_document(self):
        """Test cleaning an empty document."""
        # Arrange
        empty_doc = Document(
            page_content="", metadata={"source": "https://example.com/empty"}
        )
        mock_strategy = AsyncMock(spec=CleaningStrategy)
        mock_strategy.clean.return_value = ""
        mock_strategy.name = "MockStrategy"

        cleaner = WebDocumentCleaner(strategies=[mock_strategy])

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
            page_content="""
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <nav>
                        <ul>
                            <li><a href="/">Home</a></li>
                            <li><a href="/products">Products</a></li>
                        </ul>
                    </nav>
                    <header>
                        <h1>Website Header</h1>
                    </header>
                    <div class="cookie-banner">Accept cookies</div>
                    <div class="sidebar">Sidebar content</div>
                    <div id="content">
                        <h2>Main Content</h2>
                        <p>This is the main content    with excessive    spaces.</p>
                        <div class="advertisement">Buy now!</div>
                        <div class="social-share">
                            <button>Share on Facebook</button>
                            <button>Tweet</button>
                        </div>
                        <table>
                            <tr><td>Data 1</td><td>Data 2</td></tr>
                        </table>
                        <img src="image.jpg" alt="An example image">
                    </div>
                    <footer>
                        <p>Copyright 2023</p>
                    </footer>
                </body>
            </html>
            """,
            metadata={"source": "https://example.com/test"},
        )
        cleaner = WebDocumentCleaner()  # Use default strategies

        # Act
        result = await cleaner.clean_document(test_doc)

        # Print for debugging - helpful to see actual output
        print(f"Cleaned content: {result.page_content}")

        # Assert
        # Navigation menu structure should be removed, though text might remain
        assert "<nav>" not in result.page_content

        # Header/footer tags should be removed
        assert "<header>" not in result.page_content
        assert "<footer>" not in result.page_content

        # Cookie banner class should be removed
        assert "cookie-banner" not in result.page_content

        # Sidebar class should be removed
        assert "sidebar" not in result.page_content

        # Advertisement should be removed
        assert "advertisement" not in result.page_content

        # Social share should be removed
        assert "social-share" not in result.page_content

        # Main content should be preserved
        assert "Main Content" in result.page_content

        # HTML structural tags should be removed
        assert "<html>" not in result.page_content
        assert "<head>" not in result.page_content
        assert "<body>" not in result.page_content

        # Check whitespace normalization
        assert "content    with excessive    spaces" not in result.page_content
        assert "content with excessive spaces" in result.page_content

        # Note: Table data may or may not be preserved depending on the implementation
        # of the TableFormattingStrategy - don't test for specific table content

        # Note: Image alt text may or may not be preserved depending on implementation
        # of the ImageDescriptionStrategy - don't test for specific image content
