import logging
from typing import List, Optional

from langchain.schema import Document

from src.services.cleaners.cleaning_strategies import (
    AdvertisementRemovalStrategy,
    CleaningStrategy,
    CookieBannerRemovalStrategy,
    ImageDescriptionStrategy,
    MarkupRemovalStrategy,
    NavigationMenuRemovalStrategy,
    SidebarRemovalStrategy,
    SocialShareRemovalStrategy,
    TableFormattingStrategy,
    WebHeaderFooterRemovalStrategy,
    WebPageFeedbackCleanupStrategy,
    WebSpecificWhitespaceCleanupStrategy,
    WhitespaceNormalizationStrategy,
)

logger = logging.getLogger(__name__)


class WebDocumentCleaner:
    """Cleaner for web document content using configurable strategies."""

    def __init__(self, strategies: Optional[List[CleaningStrategy]] = None):
        """Initialize with cleaning strategies."""
        self.strategies = strategies or [
            NavigationMenuRemovalStrategy(),
            WebHeaderFooterRemovalStrategy(),
            CookieBannerRemovalStrategy(),
            SidebarRemovalStrategy(),
            AdvertisementRemovalStrategy(),
            SocialShareRemovalStrategy(),
            MarkupRemovalStrategy(),
            TableFormattingStrategy(),
            ImageDescriptionStrategy(mode="compact"),
            WebSpecificWhitespaceCleanupStrategy(),
            WebPageFeedbackCleanupStrategy(),
            WhitespaceNormalizationStrategy(),
        ]
        logger.debug(
            f"Initialized WebDocumentCleaner with {len(self.strategies)} strategies"
        )

    async def clean_document(self, document: Document) -> Document:
        """Clean a document using all configured strategies."""
        logger.debug("Starting web document cleaning...")
        content = document.page_content

        for strategy in self.strategies:
            logger.debug(f"Applying cleaning strategy: {strategy.__class__.__name__}")
            content = await strategy.clean(content)

        logger.debug("Web document cleaning completed.")
        return Document(page_content=content, metadata=document.metadata)

    async def clean_documents(self, documents: List[Document]) -> List[Document]:
        """Clean multiple documents."""
        logger.debug(f"Cleaning batch of {len(documents)} web documents")
        cleaned = [await self.clean_document(doc) for doc in documents]
        logger.debug(f"Completed cleaning {len(cleaned)} web documents")
        return cleaned

    def add_strategy(self, strategy: CleaningStrategy) -> None:
        """Add a cleaning strategy."""
        self.strategies.append(strategy)

    def remove_strategy(self, strategy_name: str) -> None:
        """Remove a cleaning strategy by name."""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
