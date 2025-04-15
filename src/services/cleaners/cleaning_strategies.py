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

    async def clean(self, text: str) -> str:
        """Remove headers and footers from text."""
        # Remove header (everything from start to page numbering)
        cleaned_text = self._header_pattern.sub("", text)

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

        # Pattern to handle stray tab characters
        self._tab_pattern = re.compile(r"\t")

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

        # Replace tab characters with a space for better text quality
        text = self._tab_pattern.sub(" ", text)

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


class SeticsWebCleanupStrategyFR(CleaningStrategy):
    """Strategy to clean up Setics web documentation pages - FRENCH."""

    def __init__(self):
        """Initialize Setics web document cleanup patterns for FRENCH docs."""
        # Pattern to remove the entire repeated table of contents section
        self._toc_pattern = re.compile(
            r"Table des matières\n\n+.*?(?=\n\n\n\n\n\d+\.\d+\.|\n\n\n\nRevision)",
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
            r"Français\s+\n+\s*\n+English\n+", re.DOTALL
        )

        # Pattern to remove footer sections
        self._footer_pattern = re.compile(
            r"Besoin d'aide supplémentaire avec ce sujet\?\s+Support & Assistance.*?Copyright © \d{4} Setics",
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
            r"× Merci pour vos commentaires\.",
        )

        # New pattern to handle section navigation references
        self._section_nav_pattern = re.compile(
            r"(\n\n+\d+\.\d+(?:\.\d+)*\.\s+[^\n]+)\s+\n\n+\n\n+(\d+\.\d+(?:\.\d+)*\.\s+[^\n]+\s+\n\n+\n\n+\d+\.\d+(?:\.\d+)*\.\s+[^\n]+)",
            re.DOTALL,
        )

        # Pattern to handle tab characters
        self._tab_pattern = re.compile(r"\t")

    async def clean(self, text: str) -> str:
        """Clean up Setics web documentation - FRENCH."""
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

        # Replace tab characters with a space for better text quality
        text = self._tab_pattern.sub(" ", text)

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


class NavigationMenuRemovalStrategy(CleaningStrategy):
    """Strategy to remove navigation menus and breadcrumbs from web pages."""

    def __init__(self):
        """Initialize navigation menu removal patterns."""
        # Enhanced pattern to match various navigation sections
        self._nav_pattern = re.compile(
            r"(Navigation|Menu|Nav\s+bar|Breadcrumbs?|Main\s+menu|Site\s+menu"
            r"|Primary\s+menu|Header\s+menu|Top\s+menu|Navbar)[^\n]*\n+.*?(?=\n\n\n)",
            re.DOTALL | re.IGNORECASE,
        )

        # Enhanced pattern to match common site menu sections with more variations
        self._site_menu_pattern = re.compile(
            r"(Home|Search|About|Contact|Login|Sign up|Sign in|Register|FAQ|Help|Support"
            r"|Products|Services|Blog|News|Shop|Cart|Account|Profile|Dashboard)[^\n]*\n+"
            r"(Home|Search|About|Contact|Login|Sign up|Sign in|Register|FAQ|Help|Support"
            r"|Products|Services|Blog|News|Shop|Cart|Account|Profile|Dashboard)[^\n]*\n+",
            re.DOTALL | re.IGNORECASE,
        )

        # Enhanced pattern to match various pagination patterns
        self._pagination_pattern = re.compile(
            r"(Previous|Next|Page \d+ of \d+|\d+ - \d+ of \d+|First|Last"
            r"|Older|Newer|Back|Forward|Results per page)[^\n]*\n[^\n]*\n",
            re.DOTALL | re.IGNORECASE,
        )

    async def clean(self, text: str) -> str:
        """Remove navigation menus from web page text."""
        # Remove navigation elements
        text = self._nav_pattern.sub("", text)

        # Remove site menu sections
        text = self._site_menu_pattern.sub("", text)

        # Remove pagination elements
        text = self._pagination_pattern.sub("", text)

        return text


class WebHeaderFooterRemovalStrategy(CleaningStrategy):
    """Strategy to remove common web page headers and footers."""

    def __init__(self):
        """Initialize web header/footer removal patterns."""
        # Enhanced pattern to match various website headers
        self._header_pattern = re.compile(
            r"^.*?(Home\s+page|Main\s+page|Sign\s+in|Log\s+in|Register|Search"
            r"|Menu|Skip\s+to\s+content|Welcome|Subscribe|Newsletter"
            r"|Language|Login|Logout|My\s+Account)[^\n]*\n+",
            re.DOTALL | re.IGNORECASE,
        )

        # Enhanced pattern to match various footer elements
        self._footer_pattern = re.compile(
            r"\n+(Copyright|©|All\s+rights\s+reserved|Terms\s+of\s+Service"
            r"|Privacy\s+Policy|Contact\s+Us|About\s+Us|Powered\s+by"
            r"|Sitemap|Legal|Disclaimer|Accessibility|Cookies"
            r"|Newsletter|Subscribe)[^\n]+(\n.*?){0,5}$",
            re.DOTALL | re.IGNORECASE,
        )

        # Enhanced pattern to match various social media sections
        self._social_pattern = re.compile(
            r"(facebook|twitter|linkedin|youtube|github|instagram|tiktok|pinterest"
            r"|Follow\s+us|Connect\s+with\s+us|Share|Social\s+Media)"
            r"[^\n]*\n[^\n]*\n[^\n]*\n+",
            re.DOTALL | re.IGNORECASE,
        )

    async def clean(self, text: str) -> str:
        """Remove headers and footers from web page text."""
        # Remove header elements
        text = self._header_pattern.sub("", text)

        # Remove footer elements
        text = self._footer_pattern.sub("", text)

        # Remove social media sections
        text = self._social_pattern.sub("", text)

        return text


class CookieBannerRemovalStrategy(CleaningStrategy):
    """Strategy to remove cookie consent banners and popups."""

    def __init__(self):
        """Initialize cookie banner removal patterns."""
        # Enhanced pattern to match various cookie consent messages
        self._cookie_pattern = re.compile(
            r"(This\s+website\s+uses\s+cookies|Cookie\s+Policy|Accept\s+cookies|Allow\s+cookies"
            r"|We\s+use\s+cookies|Cookies?\s+Notice|Cookies?\s+settings|Manage\s+cookies"
            r"|By\s+continuing\s+to\s+browse|We\s+value\s+your\s+privacy)"
            r"[^\n]*\n+[^\n]*\n+[^\n]*\n+",
            re.DOTALL | re.IGNORECASE,
        )

        # Enhanced pattern to match various privacy/GDPR notices
        self._gdpr_pattern = re.compile(
            r"(GDPR|Privacy\s+settings|Your\s+privacy|Privacy\s+choices"
            r"|Data\s+protection|Privacy\s+preference|Cookie\s+preferences"
            r"|Privacy\s+Notice|Accept\s+all|Reject\s+all)"
            r"[^\n]*\n+[^\n]*\n+[^\n]*\n+",
            re.DOTALL | re.IGNORECASE,
        )

    async def clean(self, text: str) -> str:
        """Remove cookie banners from web page text."""
        # Remove cookie consent text
        text = self._cookie_pattern.sub("", text)

        # Remove GDPR notices
        text = self._gdpr_pattern.sub("", text)

        return text


class SidebarRemovalStrategy(CleaningStrategy):
    """Strategy to remove sidebars with related links, categories, etc."""

    def __init__(self):
        """Initialize sidebar removal patterns."""
        # Enhanced pattern to match various sidebar sections
        self._sidebar_pattern = re.compile(
            r"(Related\s+Links|Categories|Recent\s+Posts|Archives|Tags|Popular\s+Posts"
            r"|Most\s+Read|Trending|Featured|Related\s+Articles|Similar\s+Posts"
            r"|Recommended|You\s+might\s+also\s+like|Top\s+Stories|Latest\s+News"
            r"|Quick\s+Links|Menu|Topics|Sections)[^\n]*\n+(?:(?!\n\n).*\n)+",
            re.DOTALL | re.IGNORECASE,
        )

        # Enhanced pattern to match various table of contents
        self._toc_sidebar_pattern = re.compile(
            r"(On\s+this\s+page|Table\s+of\s+Contents|Contents|In\s+this\s+article"
            r"|Jump\s+to\s+section|Skip\s+to|Article\s+contents|Page\s+contents"
            r"|What's\s+in\s+this\s+guide|Article\s+sections)[^\n]*\n+(?:(?!\n\n).*\n)+",
            re.DOTALL | re.IGNORECASE,
        )

    async def clean(self, text: str) -> str:
        """Remove sidebar elements from web page text."""
        # Remove sidebar sections
        text = self._sidebar_pattern.sub("", text)

        # Remove table of contents in sidebar
        text = self._toc_sidebar_pattern.sub("", text)

        return text


class AdvertisementRemovalStrategy(CleaningStrategy):
    """Strategy to remove advertisements and promotional content."""

    def __init__(self):
        """Initialize advertisement removal patterns."""
        # Pattern to match common ad indicators
        self._ad_pattern = re.compile(
            r"(Advertisement|Sponsored|Promotion|Ad\s+by|Promoted\s+by"
            r"|Recommended\s+for\s+you|Special\s+offer|Limited\s+time\s+offer"
            r"|Subscribe\s+now|Sign\s+up\s+today|Try\s+for\s+free)[^\n]*\n+(?:(?!\n\n).*\n){0,5}",
            re.DOTALL | re.IGNORECASE,
        )

        # Pattern to match product promotions
        self._promo_pattern = re.compile(
            r"(Buy\s+now|Get\s+\d+%\s+off|Only\s+\$\d+\.\d+|Free\s+shipping"
            r"|Limited\s+stock|Sale\s+ends|Special\s+discount)[^\n]*\n+(?:(?!\n\n).*\n){0,3}",
            re.DOTALL | re.IGNORECASE,
        )

    async def clean(self, text: str) -> str:
        """Remove advertisements from web page text."""
        # Remove ad sections
        text = self._ad_pattern.sub("", text)

        # Remove promotional content
        text = self._promo_pattern.sub("", text)

        return text


class MarkupRemovalStrategy(CleaningStrategy):
    """Strategy to remove leftover HTML/CSS/JavaScript elements in text."""

    def __init__(self):
        """Initialize markup removal patterns."""
        # Enhanced pattern to match HTML tags
        self._html_tag_pattern = re.compile(r"<[^>]+>", re.DOTALL)

        # Enhanced pattern to match CSS class indicators
        self._css_pattern = re.compile(r"\.\w+(-\w+)*\s*{[^}]*}", re.DOTALL)

        # Enhanced pattern to match JavaScript fragments
        self._js_pattern = re.compile(
            r"(function\s*\([^)]*\)\s*{[^}]*}|var\s+\w+\s*=|const\s+\w+\s*=|let\s+\w+\s*=|"
            r"document\.getElementById|window\.|if\s*\([^)]*\)\s*{)",
            re.DOTALL,
        )

        # Pattern to match URL parameters
        self._url_params_pattern = re.compile(r"\?[a-zA-Z0-9_=&%-]+", re.DOTALL)

        # Pattern to match data attributes
        self._data_attr_pattern = re.compile(
            r"data-[a-z0-9-]+=\"[^\"]*\"", re.DOTALL | re.IGNORECASE
        )

    async def clean(self, text: str) -> str:
        """Remove markup elements from web page text."""
        # Remove HTML tags
        text = self._html_tag_pattern.sub("", text)

        # Remove CSS fragments
        text = self._css_pattern.sub("", text)

        # Remove JavaScript fragments
        text = self._js_pattern.sub("", text)

        # Remove URL parameters
        text = self._url_params_pattern.sub("", text)

        # Remove data attributes
        text = self._data_attr_pattern.sub("", text)

        return text


class WebSpecificWhitespaceCleanupStrategy(CleaningStrategy):
    """Strategy specifically for cleaning web-specific whitespace issues."""

    def __init__(self):
        """Initialize web whitespace cleanup patterns."""
        # Pattern to match excessive consecutive newlines (3 or more)
        self._excessive_newlines = re.compile(r"\n{3,}")

        # Pattern to match Unicode special spaces
        self._special_spaces = re.compile(
            r"[\u00A0\u2000-\u200F\u2028-\u202F\u205F\u3000]"
        )

        # Pattern to clean up list formatting
        self._list_cleanup = re.compile(
            r"(\n\s*[-•*]\s*[^\n]+)(\n+)(?=\s*[-•*]\s*)", re.DOTALL
        )

        # Clean up spacing at section boundaries
        self._section_boundary = re.compile(
            r"(\n#{1,6}\s+[^\n]+\n)(\n+)(#{1,6}\s+)", re.DOTALL
        )

        # Pattern to fix no-break spaces and other special characters
        self._special_chars = re.compile(
            r"[\u2013\u2014\u2018\u2019\u201C\u201D\u2026]"
        )

        # Pattern to fix inconsistent list markers
        self._list_markers = re.compile(r"\n\s*[•\-\*○●◦□■◆▪▫]\s*", re.DOTALL)

    async def clean(self, text: str) -> str:
        """Clean up web-specific whitespace issues."""
        # Replace special Unicode spaces with regular spaces
        text = self._special_spaces.sub(" ", text)

        # Normalize special characters to ASCII equivalents
        text = self._special_chars.sub(
            lambda m: {
                "\u2013": "-",
                "\u2014": "--",
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2026": "...",
            }.get(m.group(0), m.group(0)),
            text,
        )

        # Normalize list markers
        text = self._list_markers.sub("\n• ", text)

        # Fix list item spacing
        text = self._list_cleanup.sub(r"\1\n", text)

        # Fix section heading spacing
        text = self._section_boundary.sub(r"\1\n\3", text)

        # Normalize excessive newlines to double newlines
        text = self._excessive_newlines.sub("\n\n", text)

        # Clean trailing/leading whitespace
        text = text.strip()

        return text


class WebPageFeedbackCleanupStrategy(CleaningStrategy):
    """Strategy to remove feedback forms and rating sections."""

    def __init__(self):
        """Initialize feedback section removal patterns."""
        # Enhanced pattern to match various feedback sections
        self._feedback_pattern = re.compile(
            r"(Was\s+this\s+page\s+helpful|Rate\s+this\s+page|Give\s+feedback|Send\s+feedback"
            r"|How\s+useful\s+was\s+this|Did\s+you\s+find\s+this|Leave\s+a\s+comment"
            r"|Share\s+your\s+thoughts|What\s+do\s+you\s+think|Tell\s+us\s+what\s+you\s+think)"
            r"[^\n]*\n+(?:(?!\n\n).*\n)+",
            re.DOTALL | re.IGNORECASE,
        )

        # Enhanced pattern to match various rating elements
        self._rating_pattern = re.compile(
            r"(Yes|No|Maybe|[1-5]\s+stars?|Rating:\s*\d+/\d+|Helpful|Not\s+helpful"
            r"|Like|Dislike|Thumbs\s+up|Thumbs\s+down|Recommend|Would\s+not\s+recommend)"
            r"[^\n]*\n+[^\n]*\n+",
            re.DOTALL | re.IGNORECASE,
        )

    async def clean(self, text: str) -> str:
        """Remove feedback and rating elements from web page text."""
        # Remove feedback sections
        text = self._feedback_pattern.sub("", text)

        # Remove rating elements
        text = self._rating_pattern.sub("", text)

        return text


class SocialShareRemovalStrategy(CleaningStrategy):
    """Strategy to remove social sharing buttons and sections."""

    def __init__(self):
        """Initialize social share removal patterns."""
        # Pattern to match share sections
        self._share_pattern = re.compile(
            r"(Share\s+this|Share\s+on|Share\s+via|Share\s+with|Share\s+to"
            r"|Tweet|Pin\s+it|Email\s+this|Send\s+to|Forward\s+to)"
            r"[^\n]*\n+(?:(?!\n\n).*\n){0,3}",
            re.DOTALL | re.IGNORECASE,
        )

        # Pattern to match social media icon groups
        self._social_icons_pattern = re.compile(
            r"(facebook|twitter|linkedin|youtube|pinterest|instagram|whatsapp|telegram|reddit)"
            r"[^\n]*\n+(facebook|twitter|linkedin|youtube|pinterest|instagram|whatsapp|telegram|reddit)",
            re.DOTALL | re.IGNORECASE,
        )

    async def clean(self, text: str) -> str:
        """Remove social sharing elements from web page text."""
        # Remove share sections
        text = self._share_pattern.sub("", text)

        # Remove social media icon groups
        text = self._social_icons_pattern.sub("", text)

        return text
