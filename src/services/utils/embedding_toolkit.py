import re
import uuid
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants to avoid magic numbers
MAX_NAME_LENGTH = 50
TRUNCATED_NAME_LENGTH = 42
UUID_SUFFIX_LENGTH = 8


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
    return text.strip("-_ ")


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
    safe_name = make_safe_slug(file_name)

    # Truncate safe name to mitigate excessive length
    if len(safe_name) > MAX_NAME_LENGTH:
        safe_name = safe_name[:TRUNCATED_NAME_LENGTH]

    return safe_name.strip("-_ ")


def create_chunk_ids(chunks: List[Document], prefix: Optional[str] = None) -> List[str]:
    """
    Create human-readable unique IDs for each document chunk based on its source filename.
    Also updates each chunk's metadata with the generated ID.

    Each ID consists of a safe name derived from the metadata, the chunk's index, and a random UUID segment.
    If the safe name exceeds MAX_NAME_LENGTH characters, it is truncated.

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
        safe_name = generate_safe_name(prefix=prefix, chunk=chunk)

        unique_suffix = str(uuid.uuid4())[:UUID_SUFFIX_LENGTH]
        chunk_id = f"{safe_name}-{i}-{unique_suffix}"
        chunk.metadata["id"] = chunk_id
        chunk_ids.append(chunk_id)

    return chunk_ids


def create_image_id(source: str, index: int, prefix: Optional[str] = None) -> str:
    """
    Generate a unique identifier for an image based on its source.

    The identifier consists of a safe name derived from the image source, the literal "img",
    the image index, and a random UUID segment.

    Args:
        source (str): The source path of the image.
        index (int): The index of the image.
        prefix (Optional[str]): Default prefix if necessary.

    Returns:
        str: A unique image identifier.
    """
    prefix = prefix or "unknown"
    safe_name = generate_safe_name(prefix=prefix, source=source)

    unique_suffix = str(uuid.uuid4())[:UUID_SUFFIX_LENGTH]
    return f"{safe_name}-img{index}-{unique_suffix}"


def text_splitter_recursive_char(
    data: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller chunks using recursive character splitting.

    Args:
        data (List[Document]): List of documents to split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        List[Document]: List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(data)
    return chunks
