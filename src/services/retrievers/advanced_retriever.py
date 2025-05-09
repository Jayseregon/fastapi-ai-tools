from typing import List

from chromadb.api import ClientAPI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.configs.env_config import config
from src.services.db import chroma_service
from src.services.vectorstore import chroma_retriever


class MultiQRerankedRetriever:
    """Service to handle advanced document retrieval operations using multiple queries and reranking."""

    def __init__(self) -> None:
        """
        Initialize the retriever with client and language model.
        """
        self.client: ClientAPI = chroma_service()
        self.llm: BaseChatModel = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=config.OPENAI_API_KEY,
        )

    async def __call__(
        self, query: str, collection_name: str = "default_collection", top_n: int = 3
    ) -> List[Document]:
        retriever = await self._retriever(collection_name=collection_name, top_n=top_n)

        results = await retriever.ainvoke(query)

        return results

    async def _retriever(
        self, collection_name: str = "default_collection", top_n: int = 3
    ) -> ContextualCompressionRetriever:
        """
        Create an advanced retriever with multi-query generation and reranking.

        Args:
            collection_name: Name of the collection to retrieve documents from.
            top_n: Number of top documents to return after reranking.

        Returns:
            A contextual compression retriever with multi-query and reranking capabilities.
        """
        # Get the base retriever for the specified collection
        base_retriever = await chroma_retriever(collection_name=collection_name)

        # Create a compressor for reranking results
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=top_n)
        # default model: ms-marco-TinyBERT-L-2-v2
        # best cross-encoder model: ms-marco-MiniLM-L-12-v2

        # Use MultiQueryRetriever to generate multiple search queries from the original query
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
            include_original=True,
        )

        # Add reranking for better results
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=multi_query_retriever
        )

        return retriever
