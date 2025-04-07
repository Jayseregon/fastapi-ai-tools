import asyncio
from typing import Dict, List, Set, Tuple

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.configs.env_config import config
from src.services.db import chroma_service


class ChromaStore:
    """Service to manage vector database operations with ChromaDB."""

    def __init__(self) -> None:
        """
        Initialize the ChromaStore service with Chroma client and embedding function.
        """
        self.client: ClientAPI = chroma_service()
        self.embedding_function: OpenAIEmbeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", openai_api_key=config.OPENAI_API_KEY
        )

    async def _check_connection(self) -> None:
        """
        Check ChromaDB connection and raise an exception if unavailable.

        Raises:
            Exception: If ChromaDB is not available or connection fails.
        """
        heartbeat: int = await asyncio.to_thread(self.client.heartbeat)
        if not heartbeat > 0:
            raise Exception("ChromaDB is not available")

    async def _get_collection(
        self, collection_name: str = "default_collection"
    ) -> Collection:
        """
        Get or create a ChromaDB collection.

        Args:
            collection_name: Name of the collection to retrieve or create.

        Returns:
            Collection: The requested ChromaDB collection.

        Raises:
            Exception: If ChromaDB connection fails.
        """
        await self._check_connection()
        return await asyncio.to_thread(
            self.client.get_or_create_collection, collection_name
        )

    async def _get_vector_store(
        self, collection_name: str = "default_collection"
    ) -> Chroma:
        """
        Get a Chroma vector store instance for a collection.

        Args:
            collection_name: Name of the collection to use.

        Returns:
            Chroma: A configured Chroma vector store instance.
        """
        collection: Collection = await self._get_collection(collection_name)
        return Chroma(
            client=self.client,
            collection_name=collection.name,
            embedding_function=self.embedding_function,
        )

    async def get_retriever(
        self, collection_name: str = "default_collection", k: int = 10
    ) -> BaseRetriever:
        """
        Get a retriever for the vector store with specified parameters.

        Args:
            collection_name: Name of the collection to use.
            k: Number of documents to retrieve in searches.

        Returns:
            BaseRetriever: A configured retriever for the vector store.
        """
        vector_store: Chroma = await self._get_vector_store(collection_name)
        return await asyncio.to_thread(
            vector_store.as_retriever, search_kwargs={"k": k}
        )

    async def _get_source_tracker(self, collection_name: str) -> Set[str]:
        """
        Get set of sources already stored in a collection.

        Args:
            collection_name: Name of the collection to check

        Returns:
            Set of document sources already stored in the collection
        """
        collection = await self._get_collection(collection_name)

        # Get all metadata
        # Paginate to handle large collections
        sources = set()
        limit = 1000  # Fetch in batches
        offset = 0

        while True:
            # Get a batch of records with their metadata
            results = await asyncio.to_thread(
                collection.get, include=["metadatas"], limit=limit, offset=offset
            )

            # Break if no more records
            if (
                not results
                or not results["metadatas"]
                or len(results["metadatas"]) == 0
            ):
                break

            # Process this batch of metadatas
            for metadata in results["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])

            # If we got fewer results than the limit, we're done
            if len(results["metadatas"]) < limit:
                break

            # Otherwise continue with next batch
            offset += limit

        return sources

    async def add_documents(
        self,
        documents: List[Document],
        ids: List[str],
        collection_name: str = "default_collection",
        batch_size: int = 50,
        skip_existing: bool = True,
    ) -> Tuple[int, int, List[str]]:
        """
        Store documents with their embeddings in the vector store, skipping documents
        from sources that already exist in the collection.
        Uses batching to avoid payload size limitations.

        Args:
            documents: List of documents to store.
            ids: List of IDs for the documents.
            collection_name: Name of the collection to store in.
            batch_size: Number of documents per batch.
            skip_existing: If True, skip documents from sources that already exist.

        Returns:
            Tuple containing:
            - Number of documents added
            - Number of documents skipped
            - List of skipped sources
        """
        if not documents:
            raise ValueError("No documents provided for storage.")

        # First, get existing sources
        existing_sources = await self._get_source_tracker(collection_name)

        # Group documents by source
        docs_by_source: Dict[str, List[int]] = {}
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source")
            if source:
                if source not in docs_by_source:
                    docs_by_source[source] = []
                docs_by_source[source].append(i)

        # Determine which documents to add
        filtered_docs = []
        filtered_ids = []
        skipped_sources = []

        for source, indices in docs_by_source.items():
            if source in existing_sources and skip_existing:
                # Skip this source as it already exists
                skipped_sources.append(source)
            else:
                # Add all documents from this source
                for idx in indices:
                    filtered_docs.append(documents[idx])
                    filtered_ids.append(ids[idx])

        # Handle documents with no source
        for i, doc in enumerate(documents):
            if not doc.metadata.get("source"):
                filtered_docs.append(doc)
                filtered_ids.append(ids[i])

        added_count = 0
        if filtered_docs:
            vector_store: Chroma = await self._get_vector_store(collection_name)

            # Process documents in batches
            total_docs = len(filtered_docs)
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = filtered_docs[i:batch_end]
                batch_ids = filtered_ids[i:batch_end]

                try:
                    await asyncio.to_thread(
                        vector_store.add_documents, documents=batch_docs, ids=batch_ids
                    )
                    added_count += len(batch_docs)
                except Exception as e:
                    raise Exception(f"Error adding batch documents to ChromaDB: {e}")

        skipped_count = len(documents) - added_count
        return (added_count, skipped_count, skipped_sources)

    async def replace_documents(
        self,
        documents: List[Document],
        ids: List[str],
        collection_name: str = "default_collection",
        batch_size: int = 50,
    ) -> Tuple[int, int, int]:
        """
        Replace existing documents with new versions by deleting old chunks and adding new ones.

        Args:
            documents: List of document chunks to store
            ids: List of IDs for the documents
            collection_name: Name of collection to store in
            batch_size: Number of documents per batch

        Returns:
            Tuple containing (docs_added, docs_replaced, sources_updated)
        """
        if not documents:
            raise ValueError("No documents provided for storage.")

        collection = await self._get_collection(collection_name)

        # Extract unique sources from the documents
        document_sources = set()
        for doc in documents:
            source = doc.metadata.get("source")
            if source:
                document_sources.add(source)

        # Count metrics
        sources_updated = len(document_sources)
        docs_replaced = 0

        # For each source, remove existing chunks
        for source in document_sources:
            # Find all document IDs with this source
            results = await asyncio.to_thread(
                collection.get, where={"source": source}, include=["metadatas"]
            )

            if results and results["ids"]:
                # Count how many docs we're replacing
                docs_replaced += len(results["ids"])

                # Delete all chunks with this source
                await asyncio.to_thread(collection.delete, ids=results["ids"])

        # Add the new chunks - force add because old chunks are deleted
        added_count = await self.add_documents(
            documents=documents,
            ids=ids,
            collection_name=collection_name,
            batch_size=batch_size,
            skip_existing=False,  # Force add new chunks
        )

        return (added_count[0], docs_replaced, sources_updated)


# Standalone functions for external use
async def chroma_retriever(
    collection_name: str = "default_collection", k: int = 10
) -> BaseRetriever:
    """
    A convenience function that creates a ChromaStore instance and returns a retriever.

    Args:
        collection_name: Name of the collection to retrieve from.
        k: Number of documents to retrieve in searches.

    Returns:
        BaseRetriever: A configured retriever for the specified collection.
    """
    store: ChromaStore = ChromaStore()
    return await store.get_retriever(collection_name, k)
