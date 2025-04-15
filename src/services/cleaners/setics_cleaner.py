from typing import List, Optional

from langchain.schema import Document

from src.services.cleaners.cleaning_strategies import (
    CleaningStrategy,
    SeticsWebCleanupStrategy,
    SeticsWebCleanupStrategyFR,
    WhitespaceNormalizationStrategy,
)


class SeticsDocumentCleaner:
    """Cleaner for Setics web document content using configurable strategies."""

    def __init__(self):
        """Initialize the cleaner with language-specific strategy mappings."""
        self.language_strategies = {
            "default": [
                SeticsWebCleanupStrategy(),
                WhitespaceNormalizationStrategy(),
            ],
            "fr": [
                SeticsWebCleanupStrategyFR(),
                WhitespaceNormalizationStrategy(),
            ],
        }
        # Custom strategies that apply to all languages
        self.custom_strategies = []

    def get_strategies_for_language(self, language: str) -> List[CleaningStrategy]:
        """Get the appropriate strategies for the given language."""
        base_strategies = self.language_strategies.get(
            language, self.language_strategies["default"]
        )
        return base_strategies + self.custom_strategies

    async def clean_document(self, document: Document) -> Document:
        """Clean a document using strategies appropriate for its language."""
        content = document.page_content
        language = document.metadata.get("language", "default")

        strategies = self.get_strategies_for_language(language)

        for strategy in strategies:
            content = await strategy.clean(content)
        return Document(page_content=content, metadata=document.metadata)

    async def clean_documents(self, documents: List[Document]) -> List[Document]:
        """Clean multiple documents."""
        return [await self.clean_document(doc) for doc in documents]

    def add_strategy(
        self, strategy: CleaningStrategy, language: Optional[str] = None
    ) -> None:
        """
        Add a cleaning strategy.
        If language is specified, add to that language's strategies, otherwise add to custom strategies.
        """
        if language:
            if language not in self.language_strategies:
                self.language_strategies[language] = []
            self.language_strategies[language].append(strategy)
        else:
            self.custom_strategies.append(strategy)

    def remove_strategy(
        self, strategy_name: str, language: Optional[str] = None
    ) -> None:
        """Remove a cleaning strategy by name for the specified language or from custom strategies."""
        if language and language in self.language_strategies:
            self.language_strategies[language] = [
                s for s in self.language_strategies[language] if s.name != strategy_name
            ]
        elif language is None:
            self.custom_strategies = [
                s for s in self.custom_strategies if s.name != strategy_name
            ]
