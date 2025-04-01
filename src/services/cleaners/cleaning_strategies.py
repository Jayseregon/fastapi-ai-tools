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
    """Strategy to remove headers and footers from PDF documents."""

    def __init__(self):
        """Initialize with pattern to match headers ending with page numbering."""
        # Pattern to match from beginning of document to the page number line
        self._header_pattern = re.compile(
            r"^.*?Page\s+\d+\s+of\s+\d+.*?\n",
            re.DOTALL,  # Make dot match newlines to capture entire header block
        )

        # # Optional: Pattern for footers if needed
        # self._footer_pattern = re.compile(
        #     r"\n.*?Confidential.*?$",
        #     re.DOTALL
        # )

    async def clean(self, text: str) -> str:
        """Remove headers and footers from text."""
        # Remove header (everything from start to page numbering)
        cleaned_text = self._header_pattern.sub("", text)

        # Optionally remove footer if present
        # cleaned_text = self._footer_pattern.sub("", cleaned_text)

        return cleaned_text


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

        # New pattern for consistent spacing after bullet points
        self._bullet_spacing = re.compile(r"(•.+?)(\n{3,})", re.DOTALL)

        # New pattern to fix spacing around headings
        self._heading_whitespace = re.compile(r"(\n#+\s+.+?\n)\n+", re.DOTALL)

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

        # Ensure consistent spacing after headings
        normalized = self._heading_whitespace.sub(r"\1\n", normalized)

        # Fix spacing after bullet points
        normalized = self._bullet_spacing.sub(r"\1\n\n", normalized)

        # Clean up any remaining edge cases
        normalized = normalized.strip()

        return normalized


class TableFormattingStrategy(CleaningStrategy):
    """Strategy to normalize table formatting."""

    async def clean(self, text: str) -> str:
        """Clean up markdown tables for better processing."""
        # Locate tables (header + divider + rows)
        table_pattern = re.compile(r"(\|.*\|\n\|[-|]+\|\n(?:\|.*\|\n)+)", re.MULTILINE)

        def format_table(match):
            table = match.group(1)

            # Split the table into individual lines
            table_lines = table.strip().split("\n")

            if table_lines:
                header_line = table_lines[0]
                expected_cell_count = header_line.count("|") - 1

                # Fix each row to match the header cell count
                for i in range(len(table_lines)):
                    line = table_lines[i]
                    current_cell_count = line.count("|") - 1
                    if current_cell_count < expected_cell_count:
                        missing_cells = expected_cell_count - current_cell_count
                        table_lines[i] = line.rstrip("|") + ("|" * missing_cells) + "|"

            # Rejoin the fixed table lines
            formatted_table = "\n".join(table_lines)

            # Remove rows that are completely empty (only pipes and whitespace)
            formatted_table = re.sub(r"(\n\|(?:\s*\|)+\n)", "\n", formatted_table)

            # Convert HTML entities (e.g., &amp;#39;) into characters
            formatted_table = re.sub(
                r"&amp;#(\d+);", lambda m: chr(int(m.group(1))), formatted_table
            )
            if not formatted_table.endswith("\n"):
                formatted_table += "\n"

            return f"\n\nTABLE:\n{formatted_table}\n"

        text = table_pattern.sub(format_table, text)

        # Remove completely empty tables (those that collapse to nothing but whitespace)
        text = re.sub(r"\n\nTABLE:\n\s*\n", "", text)

        # Remove extra trailing empty rows inside tables
        text = re.sub(r"(\n\|\s*(?:\|\s*)+\n)(\s*\|\s*(?:\|\s*)+\n)+", r"\1", text)

        return text


class ImageDescriptionStrategy(CleaningStrategy):
    """Strategy to handle image descriptions from LLM parsing."""

    def __init__(self, mode="compact"):
        """
        Initialize image description handler.

        Args:
            mode: How to handle image descriptions
                - "compact": Convert to brief format
                - "remove": Remove entirely
                - "preserve": Keep as is (default)
        """
        self._mode = mode
        self._image_pattern = re.compile(r"!\[(.*?)\]\(#\)", re.DOTALL)

        # Pattern for normalizing whitespace within image descriptions
        self._internal_newlines = re.compile(r"\n{2,}")
        self._internal_spaces = re.compile(r" {2,}")

    async def clean(self, text: str) -> str:
        """Process image descriptions based on selected mode."""
        if self._mode == "remove":
            return self._image_pattern.sub("", text)
        elif self._mode == "compact":

            def normalize_description(match):
                desc = match.group(1)
                # Normalize internal whitespace
                desc = self._internal_newlines.sub(
                    " ", desc
                )  # Replace multiple newlines with single space
                desc = desc.replace("\n", " ")  # Replace remaining newlines with spaces
                desc = self._internal_spaces.sub(" ", desc)  # Compress multiple spaces
                desc = desc.strip()  # Remove leading/trailing whitespace
                return f"[IMAGE: {desc}]"

            return self._image_pattern.sub(normalize_description, text)

        return text  # preserve mode


class SectionHeadingStrategy(CleaningStrategy):
    """Strategy to format section headings with appropriate heading levels."""

    def __init__(self):
        """Initialize the section heading strategy."""
        # Pattern to match section headings with various numbering depths
        # Matches patterns like "1 Title", "1.2 Title", "1.2.3 Title" but not within words
        self._heading_pattern = re.compile(
            r"^(\s*)(\d+(?:\.\d+){0,4})\s+([A-Z][^.\n]+)(?:\s*)$", re.MULTILINE
        )

        # Exclude specific patterns that match the regex but aren't headings
        self._excluded_patterns = [
            r"Figure \d+:",
            r"Table \d+:",
            r"\d+ of \d+",  # Pagination references
            r"\d+/\w+/\d+",  # Dates
            r"\d+mm",  # Measurements
            r"\d+m",  # Measurements
        ]

    async def clean(self, text: str) -> str:
        """Format section headings with appropriate heading levels."""

        def determine_heading_level(section_number):
            """Determine heading level based on section number depth."""
            # Count dots to determine heading level (1.2.3 → 3 levels deep)
            level = section_number.count(".") + 1
            # Use appropriate markdown heading level (# for h1, ## for h2, etc.)
            return "#" * min(level, 6) + " "  # Maximum of 6 levels

        def format_heading(match):
            """Format a heading with the appropriate level."""
            indent = match.group(1)
            section_number = match.group(2)
            title = match.group(3)

            # Skip if this matches any excluded pattern
            full_match = match.group(0)
            for pattern in self._excluded_patterns:
                if re.search(pattern, full_match):
                    return match.group(0)

            # Format as heading
            heading_level = determine_heading_level(section_number)
            return f"\n{indent}{heading_level}{section_number} {title}\n"

        # Apply the heading formatting
        return self._heading_pattern.sub(format_heading, text)


class FigureReferenceStrategy(CleaningStrategy):
    """Strategy to improve formatting of figure references."""

    def __init__(self):
        # Pattern to detect consecutive figure references
        self._multiple_figures_pattern = re.compile(
            r"(\n\nFigure \d+:.+?\n\n)(?=Figure \d+:)", re.DOTALL
        )

        # Pattern to format standalone figure references
        self._single_figure_pattern = re.compile(
            r"(^|\n)(?!##)(?!#)(?!-)(Figure \d+:.+?)(\n)", re.MULTILINE
        )

        # Pattern to clean up "See Figure X" references
        self._see_figure_pattern = re.compile(r"(See Figure \d+)(\s+for\s+)", re.DOTALL)

    async def clean(self, text: str) -> str:
        # Group consecutive figures with less spacing
        text = self._multiple_figures_pattern.sub(r"\1", text)

        # Format standalone figures with proper emphasis
        text = self._single_figure_pattern.sub(r"\1\n**\2**\n\n", text)

        # Emphasize in-text figure references
        text = self._see_figure_pattern.sub(r"**\1**\2", text)

        return text


class TableOfContentsStrategy(CleaningStrategy):
    """Strategy to clean and format table of contents sections."""

    def __init__(self):
        # Pattern to locate the TOC header
        self._toc_header_pattern = re.compile(
            r"(^|\n)(Table of Contents)(\n)", re.MULTILINE
        )

    async def clean(self, text: str) -> str:
        # First, mark the TOC header as a markdown header
        text = self._toc_header_pattern.sub(r"\1## \2\n\n", text)

        # Extract the TOC block (from the header until the next double-newline)
        toc_match = re.search(r"(## Table of Contents\n\n)(.*?)(\n\n)", text, re.DOTALL)
        if toc_match:
            toc_header = toc_match.group(1)
            toc_body = toc_match.group(2)
            toc_end = toc_match.group(3)

            # Reassemble entries: merge lines that don't start with a number
            toc_lines = toc_body.splitlines()
            merged = []
            current = ""
            for line in toc_lines:
                if re.match(r"^\d", line.strip()):
                    if current:
                        merged.append(current.strip())
                    current = line.strip()
                else:
                    # Continuation of previous line if current not empty
                    if current:
                        current += " " + line.strip()
                    else:
                        current = line.strip()
            if current:
                merged.append(current.strip())

            # Now, format each entry using a regex that captures the entry number, text, and trailing page info or error text.
            formatted_entries = []
            entry_pattern = re.compile(
                r"^(\d+(?:\.\d+)*)(?:\s+)(.*?)(?:\.{3,}\s*)(.+)$"
            )
            for entry in merged:
                m = entry_pattern.match(entry)
                if m:
                    number, title, page = m.groups()
                    formatted_entries.append(
                        f"- **{number}** {title} (page {page.strip()})"
                    )
                else:
                    # If the pattern doesn't match, output the line as-is
                    formatted_entries.append(f"- {entry}")
            new_toc_body = "\n".join(formatted_entries)
            new_toc = toc_header + new_toc_body + toc_end
            # Replace the original TOC block
            text = text.replace(toc_match.group(0), new_toc)

        # Similarly, handle "Table of Figures" if present
        text = re.sub(
            r"(^|\n)(Table of Figures)(\n)", r"\1## \2\n\n", text, flags=re.MULTILINE
        )
        # Clean up Figure entries in TOC with a simple pattern
        text = re.sub(r"(Figure \d+:.+?)\.+\s*(\d+)", r"- \1 (page \2)", text)

        return text


class SeticsWebCleanupStrategy(CleaningStrategy):
    """Strategy to clean up Setics web documentation pages."""

    def __init__(self):
        """Initialize Setics web document cleanup patterns."""
        # Pattern to remove the entire repeated table of contents section
        self._toc_pattern = re.compile(
            r"Table of Contents\n\n+.*?(?=\n\n\n\n\n\d+\.\d+\.|\n\n\n\nRevision)",
            re.DOTALL,
        )

        # Pattern to remove header navigation and titles
        self._header_pattern = re.compile(
            r"^\s*\n+.*?User Manual - Version \d+\.\d+\n+.*?Setics Sttar Advanced Designer\s+\|"
            r"\s+User Manual\s+Version \d+\.\d+",
            re.DOTALL,
        )

        # Pattern to remove language selector
        self._lang_selector_pattern = re.compile(
            r"English\s+\n+\s*\n+Français\n+", re.DOTALL
        )

        # Pattern to remove footer sections
        self._footer_pattern = re.compile(
            r"Need more help with this\?\s+Support & Assistance.*?Copyright © \d{4} Setics",
            re.DOTALL,
        )

        # Pattern to remove version information at start of document
        self._version_pattern = re.compile(r"^Version \d+\.\d+\s*\n+", re.MULTILINE)

        # Pattern to remove revision info completely
        self._revision_pattern = re.compile(
            r"Revision:\s+\d+\s+Last modified:\s+\d+ \w+ \d{4}", re.DOTALL
        )

        # Pattern to remove feedback form
        self._feedback_pattern = re.compile(
            r"× Thanks for your feedback\.",
        )

        # New pattern to handle section navigation references
        self._section_nav_pattern = re.compile(
            r"(\n\n+\d+\.\d+(?:\.\d+)*\.\s+[^\n]+)\s+\n\n+\n\n+(\d+\.\d+(?:\.\d+)*\.\s+[^\n]+\s+\n\n+\n\n+\d+\.\d+(?:\.\d+)*\.\s+[^\n]+)",
            re.DOTALL,
        )

    async def clean(self, text: str) -> str:
        """Clean up Setics web documentation."""
        # Remove entire table of contents section
        text = self._toc_pattern.sub("", text)

        # Remove header boilerplate
        text = self._header_pattern.sub("", text)

        # Remove language selector
        text = self._lang_selector_pattern.sub("", text)

        # Remove version information at start
        text = self._version_pattern.sub("", text)

        # Remove revision info completely
        text = self._revision_pattern.sub("", text)

        # Remove footer sections
        text = self._footer_pattern.sub("", text)

        # Remove feedback form
        text = self._feedback_pattern.sub("", text)

        # Handle section navigation headers
        # First identify the primary section heading and format it properly
        primary_heading_match = re.search(
            r"\n(\d+\.\d+(?:\.\d+)*)\.\s+([^\n]+)\s+\n", text
        )

        if primary_heading_match:
            section_num = primary_heading_match.group(1)
            title = primary_heading_match.group(2)

            # Determine heading level based on section number depth
            level = section_num.count(".") + 1
            heading_marks = "#" * min(level, 6)

            # Create properly formatted heading
            formatted_heading = f"{heading_marks} {section_num}. {title}"

            # Find and remove the navigation section that contains multiple section references
            nav_section_pattern = re.compile(
                r"(\n\n+\d+\.\d+(?:\.\d+)*\.\s+[^\n]+\s+\n\n+\n\n+\d+\.\d+(?:\.\d+)*\.\s+[^\n]+\s+\n\n+\n\n+\d+\.\d+(?:\.\d+)*\.\s+[^\n]+)",
                re.DOTALL,
            )

            text = nav_section_pattern.sub(f"\n\n{formatted_heading}\n\n", text)

        # Collapse sequences of multiple newlines to no more than 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Trim leading/trailing whitespace
        return text.strip()


class SeticsTableFormattingStrategy(CleaningStrategy):
    """Strategy to normalize tables in Setics documentation."""

    async def clean(self, text: str) -> str:
        """Clean up tables in Setics documentation."""
        # The command tables have a unique format with columns separated by newlines
        table_pattern = re.compile(
            r"(\n\nCommand\s+\nDescription\s+\nKeyboard shortcut\s+\n\n\n)(.*?)(\n\n\n)",
            re.DOTALL,
        )

        def format_command_table(match):
            header = "| Command | Description | Keyboard shortcut |\n|---------|-------------|------------------|\n"
            content = match.group(2)

            # Process each row
            rows = re.findall(
                r"\s+([^\n]+)\s+\n\s+([^\n]+(?:\n[^\n]+)*)\s+\n\s+([^\n]*)\s+\n\n",
                content,
            )
            formatted_rows = []

            for command, desc, shortcut in rows:
                # Clean up description - replace internal newlines with spaces
                desc = re.sub(r"\n\s+", " ", desc)
                formatted_rows.append(
                    f"| {command.strip()} | {desc.strip()} | {shortcut.strip()} |"
                )

            return (
                "\n\n**Table: Commands**\n\n"
                + header
                + "\n".join(formatted_rows)
                + "\n\n"
            )

        # Apply table formatting
        return table_pattern.sub(format_command_table, text)


class SeticsHeadingCleanupStrategy(CleaningStrategy):
    """Strategy to format section headings in Setics documentation."""

    def __init__(self):
        # Pattern to match section references at the top/bottom of pages
        self._section_ref_pattern = re.compile(
            r"(\n\d+\.\d+(?:\.\d+)*\.\s+[^\n]+)\s+\n+\s+\n+\s+\n+\s+\n+(\d+\.\d+(?:\.\d+)*\.\s+[^\n]+)",
            re.DOTALL,
        )

        # Pattern to format actual section headings
        self._heading_pattern = re.compile(
            r"(?:\n|^)(\d+(?:\.\d+){0,4})\.\s+([A-Z][^\n]+)(?=\n)",
        )

    async def clean(self, text: str) -> str:
        """Format section headings and clean up section references."""
        # Remove redundant section references that appear at page transitions
        text = self._section_ref_pattern.sub(r"\1", text)

        # Format actual section headings with markdown
        def format_heading(match):
            section_num = match.group(1)
            title = match.group(2)

            # Determine heading level based on section number depth
            level = section_num.count(".") + 1
            heading_marks = "#" * min(level, 6)

            return f"\n\n{heading_marks} {section_num}. {title}\n\n"

        return self._heading_pattern.sub(format_heading, text)
