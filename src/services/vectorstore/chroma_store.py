import asyncio
import logging
import os
from typing import Dict, List, Set, Tuple, Union

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.configs.env_config import config
from src.services.db import chroma_service

logger = logging.getLogger(__name__)


class ChromaStore:
    """Service to manage vector database operations with ChromaDB."""

    def __init__(self) -> None:
        """
        Initialize the ChromaStore service with Chroma client and embedding function.
        """
        logger.debug("Initializing ChromaStore")
        self.client: ClientAPI = chroma_service()
        self.embedding_function: OpenAIEmbeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", openai_api_key=config.OPENAI_API_KEY
        )

    @property
    def store_metadata(self) -> Dict[str, Union[int, Dict[str, Dict[str, int]]]]:
        """
        Get metadata about the vector store.

        Retrieves information about all collections in the Chroma store,
        including the total number of collections and details about each
        individual collection.

        Returns:
            Dict[str, Union[int, Dict[str, Dict[str, int]]]]: A dictionary containing:
                - "nb_collections": The total number of collections in the store
                - "details": A dictionary mapping collection names to their details,
                            where each detail contains the count of items in that collection
        """
        nb_collection: int = self.client.count_collections()
        collections: List[str] = self.client.list_collections()
        collections_details: Dict[str, Dict[str, int]] = {
            coll: {"count": self.client.get_collection(coll).count()}
            for coll in collections
        }
        return {"nb_collections": nb_collection, "details": collections_details}

    async def _check_connection(self) -> None:
        """
        Check ChromaDB connection and raise an exception if unavailable.

        Raises:
            Exception: If ChromaDB is not available or connection fails.
        """
        heartbeat: int = await asyncio.to_thread(self.client.heartbeat)
        if not heartbeat > 0:
            logger.error("ChromaDB heartbeat check failed")
            raise Exception("ChromaDB is not available")
        logger.debug(f"ChromaDB connection verified with heartbeat: {heartbeat}")

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
        collection = await asyncio.to_thread(
            self.client.get_or_create_collection, collection_name
        )
        logger.debug(f"Retrieved collection: '{collection.name}'")
        return collection

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
        logger.debug(f"Creating vector store for collection: '{collection_name}'")
        collection: Collection = await self._get_collection(collection_name)
        return Chroma(
            client=self.client,
            collection_name=collection.name,
            embedding_function=self.embedding_function,
        )

    async def _get_source_tracker(
        self, collection_name: str, is_web: bool = False
    ) -> Set[str]:
        """
        Get set of sources already stored in a collection.

        Args:
            collection_name: Name of the collection to check

        Returns:
            Set of document sources already stored in the collection
        """
        logger.debug(f"Getting source tracker for collection: '{collection_name}'")
        collection = await self._get_collection(collection_name)

        # Get all metadata
        # Paginate to handle large collections
        sources = set()
        limit = 1000  # Fetch in batches
        offset = 0

        while True:
            logger.debug(f"Fetching metadata batch: limit={limit}, offset={offset}")
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
                logger.debug("No more metadata records found")
                break

            # Process this batch of metadatas
            batch_sources = 0
            for metadata in results["metadatas"]:
                if metadata and "source" in metadata:
                    # Extract only the filename from the full path
                    source_path = metadata["source"]
                    source_filename = (
                        os.path.basename(source_path) if not is_web else source_path
                    )
                    sources.add(source_filename)
                    batch_sources += 1

            logger.debug(f"Processed batch with {batch_sources} sources found")

            # If we got fewer results than the limit, we're done
            if len(results["metadatas"]) < limit:
                logger.debug("Reached end of collection metadata")
                break

            # Otherwise continue with next batch
            offset += limit
            logger.debug(f"Moving to next batch, offset={offset}")

        logger.debug(f"Found {len(sources)} unique document sources in collection")
        return sources

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
        retriever = await asyncio.to_thread(
            vector_store.as_retriever, search_kwargs={"k": k}
        )
        logger.debug(f"Retriever created successfully for '{collection_name}'")
        return retriever

    async def add_documents(
        self,
        documents: List[Document],
        ids: List[str],
        collection_name: str = "default_collection",
        batch_size: int = 50,
        skip_existing: bool = True,
        is_web: bool = False,
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
        logger.debug(
            f"Adding {len(documents)} documents to collection '{collection_name}' "
            f"(batch_size={batch_size}, skip_existing={skip_existing})"
        )

        if not documents:
            logger.warning("No documents provided for storage")
            raise ValueError("No documents provided for storage.")

        # First, get existing sources (now contains just filenames)
        existing_sources = await self._get_source_tracker(
            collection_name=collection_name, is_web=is_web
        )
        logger.debug(f"Found {len(existing_sources)} existing sources in collection")
        logger.debug(f"EXISTING SOURCES: {existing_sources}")

        # Group documents by source filename (not full path)
        docs_by_source: Dict[str, List[int]] = {}
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source")
            if source:
                # Extract the filename from the full path
                source_filename = os.path.basename(source) if not is_web else source
                if source_filename not in docs_by_source:
                    docs_by_source[source_filename] = []
                docs_by_source[source_filename].append(i)

        logger.debug(
            f"Documents grouped into {len(docs_by_source)} unique source files"
        )

        # Determine which documents to add
        filtered_docs = []
        filtered_ids = []
        skipped_sources = []

        for source_filename, indices in docs_by_source.items():
            if source_filename in existing_sources and skip_existing:
                # Skip this source as it already exists
                logger.debug(
                    f"Skipping {len(indices)} documents from existing source: '{source_filename}'"
                )
                skipped_sources.append(source_filename)
            else:
                # Add all documents from this source
                logger.debug(
                    f"Adding {len(indices)} documents from source: '{source_filename}'"
                )
                for idx in indices:
                    filtered_docs.append(documents[idx])
                    filtered_ids.append(ids[idx])

        # Handle documents with no source
        no_source_count = 0
        for i, doc in enumerate(documents):
            if not doc.metadata.get("source"):
                filtered_docs.append(doc)
                filtered_ids.append(ids[i])
                no_source_count += 1

        if no_source_count > 0:
            logger.debug(f"Found {no_source_count} documents with no source")

        added_count = 0
        if filtered_docs:
            logger.debug(
                f"Adding {len(filtered_docs)} filtered documents to vector store"
            )
            vector_store: Chroma = await self._get_vector_store(collection_name)

            # Process documents in batches
            total_docs = len(filtered_docs)
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = filtered_docs[i:batch_end]
                batch_ids = filtered_ids[i:batch_end]

                logger.debug(
                    f"Processing batch {i // batch_size + 1}: documents {i + 1}-{batch_end} of {total_docs}"
                )

                try:
                    await asyncio.to_thread(
                        vector_store.add_documents, documents=batch_docs, ids=batch_ids
                    )
                    added_count += len(batch_docs)
                    logger.debug(
                        f"Successfully added batch of {len(batch_docs)} documents"
                    )
                except Exception as e:
                    logger.error(f"Error adding documents batch: {str(e)}")
                    raise Exception(f"Error adding batch documents to ChromaDB: {e}")

        skipped_count = len(documents) - added_count
        logger.debug(
            f"Documents added: {added_count}, skipped: {skipped_count}, skipped sources: {len(skipped_sources)}"
        )
        return (added_count, skipped_count, skipped_sources)

    async def replace_documents(
        self,
        documents: List[Document],
        ids: List[str],
        collection_name: str = "default_collection",
        batch_size: int = 50,
        is_web: bool = False,
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
        logger.debug(
            f"Replacing documents in collection '{collection_name}': {len(documents)} documents provided"
        )

        if not documents:
            logger.warning("No documents provided for replacement")
            raise ValueError("No documents provided for replacement.")

        collection = await self._get_collection(collection_name)

        # Extract unique source filenames from the documents
        document_source_filenames = set()
        for doc in documents:
            source = doc.metadata.get("source")
            if source:
                source_filename = os.path.basename(source) if not is_web else source
                document_source_filenames.add(source_filename)

        logger.debug(
            f"Found {len(document_source_filenames)} unique source files in documents to replace"
        )

        # Count metrics
        sources_updated = len(document_source_filenames)
        docs_replaced = 0

        # For each source filename, find and remove existing chunks
        for source_filename in document_source_filenames:
            logger.debug(
                f"Processing replacements for source file: '{source_filename}'"
            )

            # Get all documents and their metadata
            results = await asyncio.to_thread(collection.get, include=["metadatas"])

            if results and results["ids"] and results["metadatas"]:
                # Filter documents with matching source filename
                docs_to_delete = []
                for i, metadata in enumerate(results["metadatas"]):
                    if metadata and "source" in metadata:
                        doc_source = metadata["source"]
                        doc_source_basename = (
                            os.path.basename(doc_source) if not is_web else doc_source
                        )
                        if doc_source_basename == source_filename:
                            docs_to_delete.append(results["ids"][i])

                if docs_to_delete:
                    # Count how many docs we're replacing
                    docs_replaced += len(docs_to_delete)
                    logger.debug(
                        f"Deleting {len(docs_to_delete)} existing documents for source file: '{source_filename}'"
                    )

                    # Delete chunks with this source filename
                    await asyncio.to_thread(collection.delete, ids=docs_to_delete)
                    logger.debug(
                        f"Successfully deleted documents for source file: '{source_filename}'"
                    )
                else:
                    logger.debug(
                        f"No existing documents found for source file: '{source_filename}'"
                    )
            else:
                logger.debug("No documents found in collection")

        # Add the new chunks - force add because old chunks are deleted
        logger.debug(f"Adding {len(documents)} new document versions")
        added_count = await self.add_documents(
            documents=documents,
            ids=ids,
            collection_name=collection_name,
            batch_size=batch_size,
            skip_existing=False,  # Force add new chunks
        )

        logger.debug(
            f"Document replacement complete: {added_count[0]} added, {docs_replaced} replaced, {sources_updated} sources updated"
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
    retriever = await store.get_retriever(collection_name, k)
    logger.debug("Chroma retriever created successfully")
    return retriever
