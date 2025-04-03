import re
import uuid
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document


class EmbeddingToolkit:
    """
    A toolkit for generating safe identifiers and slugs from text and document metadata.
    """

    @staticmethod
    def make_safe_slug(text: str) -> str:
        """
        Convert the given text to a safe slug.

        Transforms text to lowercase, removes special characters, replaces whitespace and hyphens
        with a single hyphen, and trims leading/trailing hyphens.

        Args:
            text (str): The input text.

        Returns:
            str: A slugified version of the text.
        """
        if not text:
            return ""
        text = re.sub(r"[^\w\s-]", "", text.lower())
        text = re.sub(r"[-\s]+", "-", text)
        return text.strip("-")

    @staticmethod
    def generate_safe_name(
        prefix: str, chunk: Optional[Document] = None, source: Optional[str] = None
    ) -> str:
        """
        Generate a safe name from a chunk's source metadata or a provided source string.

        Args:
            prefix (str): A default prefix if chunk or source is not provided.
            chunk (Optional[Document]): A document chunk with metadata including 'source'.
            source (Optional[str]): A source file string.

        Returns:
            str: A safe slugified name derived from the file name.
        """
        if chunk:
            file_path = chunk.metadata.get("source", prefix)
        else:
            file_path = source

        # Use prefix as fallback if no file_path provided
        file_name = Path(file_path).stem if file_path else prefix
        safe_name = EmbeddingToolkit.make_safe_slug(file_name)
        return safe_name

    @staticmethod
    def create_chunk_ids(
        chunks: List[Document], prefix: Optional[str] = None
    ) -> List[str]:
        """
        Create human-readable unique IDs for each document chunk based on its source filename.

        Each ID consists of a safe name derived from the metadata, the chunk's index, and a random UUID segment.
        If the safe name exceeds 50 characters, it is truncated to 42 characters.

        Args:
            chunks (List[Document]): List of document chunks.
            prefix (Optional[str]): Default prefix used if chunk metadata is missing.

        Returns:
            List[str]: A list of unique chunk IDs.
        """
        chunk_ids: List[str] = []
        if not chunks:
            return chunk_ids

        prefix = prefix or "unknown"
        for i, chunk in enumerate(chunks):
            safe_name = EmbeddingToolkit.generate_safe_name(prefix=prefix, chunk=chunk)

            # Truncate safe name to mitigate excessive length
            if len(safe_name) > 50:
                safe_name = safe_name[:42]

            unique_suffix = str(uuid.uuid4())[:8]
            chunk_id = f"{safe_name}-{i}-{unique_suffix}"
            chunk.metadata["id"] = chunk_id
            chunk_ids.append(chunk_id)

        return chunk_ids

    @staticmethod
    def create_image_id(source: str, index: int, prefix: Optional[str] = None) -> str:
        """
        Generate a unique identifier for an image based on its source.

        The identifier consists of a safe name derived from the image source, the literal "img", and a random UUID segment.
        If the safe name exceeds 50 characters, it is truncated to 42 characters.

        Args:
            source (str): The source path of the image.
            prefix (Optional[str]): Default prefix if necessary.

        Returns:
            str: A unique image identifier.
        """
        prefix = prefix or "unknown"
        safe_name = EmbeddingToolkit.generate_safe_name(prefix=prefix, source=source)
        if len(safe_name) > 50:
            safe_name = safe_name[:42]

        unique_suffix = str(uuid.uuid4())[:8]
        return f"{safe_name}-img{index}-{unique_suffix}"
