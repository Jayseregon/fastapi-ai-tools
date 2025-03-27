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
    """Strategy to normalize excessive whitespace in PDF documents while preserving structure."""

    def __init__(self):
        """Initialize the whitespace normalization strategy."""
        # Match excessive consecutive newlines (3 or more) and normalize to double newline
        self._excessive_newlines = re.compile(r"\n{3,}")

        # Multiple spaces within lines
        self._multiple_spaces = re.compile(r" {2,}")

        # Whitespace before newlines (trailing whitespace on lines)
        self._trailing_whitespace = re.compile(r"[ \t]+\n")

        # Whitespace after newlines (leading whitespace on lines, but preserve indentation)
        self._leading_whitespace = re.compile(r"\n[ \t]{2,}")

        # Handle bullet points - updated with more aggressive pattern
        self._bullet_pattern = re.compile(r"\n\s*\uf0b7\s*")

        # Additional pattern to fix bullet points followed by newlines
        self._bullet_cleanup_pattern = re.compile(r"•\s*\n\s*")

    async def clean(self, text: str) -> str:
        """Clean whitespace while preserving document structure."""
        # Replace unicode bullet with standard bullet
        normalized = text.replace("\uf0b7", "•")

        # First pass: Convert bullets to standard format
        normalized = self._bullet_pattern.sub("\n• ", normalized)

        # Second pass: Fix any remaining cases of bullet followed by newline
        normalized = self._bullet_cleanup_pattern.sub("• ", normalized)

        # Remove trailing whitespace on lines
        normalized = self._trailing_whitespace.sub("\n", normalized)

        # Normalize leading whitespace on lines (preserve basic indentation)
        normalized = self._leading_whitespace.sub("\n  ", normalized)

        # Compress multiple spaces within lines
        normalized = self._multiple_spaces.sub(" ", normalized)

        # Normalize excessive newlines to double newlines
        normalized = self._excessive_newlines.sub("\n\n", normalized)

        # Clean up any remaining edge cases
        normalized = normalized.strip()

        return normalized


class TableFormattingStrategy(CleaningStrategy):
    """Strategy to improve table formatting."""

    async def clean(self, text: str) -> str:
        """Format tables better for NLP processing."""
        # Implementation for table formatting
        # This could detect markdown tables and ensure proper spacing
        return text  # Placeholder implementation
