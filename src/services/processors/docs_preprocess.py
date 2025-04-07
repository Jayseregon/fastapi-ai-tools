import asyncio
from typing import List, Optional, Tuple

from langchain.schema import Document

from src.services.utils import create_chunk_ids, text_splitter_recursive_char


class DocumentsPreprocessing:
    """Service to preprocess documents before embedding."""

    async def __call__(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        prefix: Optional[str] = None,
    ) -> Tuple[List[Document], List[str]]:
        """
        Preprocess documents by splitting into chunks and creating IDs.

        Args:
            documents: List of documents to process
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            prefix: Optional prefix for document IDs

        Returns:
            Tuple containing (document_chunks, document_ids)
        """
        doc_chunks = await asyncio.to_thread(
            text_splitter_recursive_char, documents, chunk_size, chunk_overlap
        )
        doc_ids = await asyncio.to_thread(create_chunk_ids, doc_chunks, prefix)

        return doc_chunks, doc_ids
