from typing import List, Optional

from langchain.schema import Document

from src.services.cleaners.cleaning_strategies import (  # SeticsTableFormattingStrategy,; SeticsHeadingCleanupStrategy,
    CleaningStrategy,
    SeticsWebCleanupStrategy,
    WhitespaceNormalizationStrategy,
)


class SeticsDocumentCleaner:
    """Cleaner for Setics web document content using configurable strategies."""

    def __init__(self, strategies: Optional[List[CleaningStrategy]] = None):
        """Initialize with cleaning strategies."""
        self.strategies = strategies or [
            SeticsWebCleanupStrategy(),
            # SeticsHeadingCleanupStrategy(),
            # SeticsTableFormattingStrategy(),
            WhitespaceNormalizationStrategy(),
        ]

    async def clean_document(self, document: Document) -> Document:
        """Clean a document using all configured strategies."""
        content = document.page_content
        for strategy in self.strategies:
            content = await strategy.clean(content)
        return Document(page_content=content, metadata=document.metadata)

    async def clean_documents(self, documents: List[Document]) -> List[Document]:
        """Clean multiple documents."""
        return [await self.clean_document(doc) for doc in documents]

    def add_strategy(self, strategy: CleaningStrategy) -> None:
        """Add a cleaning strategy."""
        self.strategies.append(strategy)

    def remove_strategy(self, strategy_name: str) -> None:
        """Remove a cleaning strategy by name."""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
