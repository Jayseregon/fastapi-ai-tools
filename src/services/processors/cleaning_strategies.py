import re
from abc import ABC, abstractmethod


class CleaningStrategy(ABC):
    """Abstract base class for document cleaning strategies."""

    @abstractmethod
    async def clean(self, text: str) -> str:
        """Clean the text using this strategy."""
        pass

    @property
    def name(self) -> str:
        """Return the name of this cleaning strategy."""
        return self.__class__.__name__


class HeaderFooterRemovalStrategy(CleaningStrategy):
    """Strategy to remove headers and footers from documents."""

    def __init__(self):
        self._pattern = re.compile(
            r"^(.*?(Page\s+\d+\s+of\s+\d+|Document Number:.*?|Confidential|Revision:.*?)){1,5}\s*$",
            re.MULTILINE,
        )

    async def clean(self, text: str) -> str:
        """Remove headers and footers from text."""
        return self._pattern.sub("", text)


class WhitespaceNormalizationStrategy(CleaningStrategy):
    """Strategy to normalize excessive whitespace."""

    def __init__(self, max_consecutive_newlines: int = 2):
        self._pattern = re.compile(r"\n{" + str(max_consecutive_newlines + 1) + r",}")
        self._replacement = "\n" * max_consecutive_newlines

    async def clean(self, text: str) -> str:
        """Normalize excessive whitespace."""
        return self._pattern.sub(self._replacement, text)


class TableFormattingStrategy(CleaningStrategy):
    """Strategy to improve table formatting."""

    async def clean(self, text: str) -> str:
        """Format tables better for NLP processing."""
        # Implementation for table formatting
        # This could detect markdown tables and ensure proper spacing
        return text  # Placeholder implementation
