"""Core Corrective RAG implementation."""

from pathlib import Path
from typing import Dict, Iterable, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from .ingestion import IngestionPipeline, IngestionResult
from .nodes import GraphState
from .retrievers import RetrievalResult, VectorRetriever

# Load environment variables
load_dotenv()


class CorrectiveRAG:
    """
    Main class for the Corrective RAG system.

    This class orchestrates the corrective RAG pipeline using LangGraph
    for workflow management and various LangChain components for RAG functionality.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Corrective RAG system.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.ingestion_pipeline: Optional[IngestionPipeline] = None
        self.retriever: Optional[VectorRetriever] = None
        self._last_state: Optional[GraphState] = None
        self._setup_components()

    def _setup_components(self):
        """Set up the various components of the RAG system."""
        self.ingestion_pipeline = IngestionPipeline(
            config=self.config.get("ingestion")
        )
        self.retriever = self._build_retriever()
        # TODO: Initialize LangGraph workflow
        # TODO: Set up language model
        # TODO: Set up web search tool
        pass

    def _build_retriever(self) -> Optional[VectorRetriever]:
        retrieval_cfg = self.config.get("retrieval", {})
        chroma_dir = Path(retrieval_cfg.get("chroma_dir", "vectorstores/chroma"))
        faiss_dir = Path(retrieval_cfg.get("faiss_dir", "vectorstores/faiss"))
        top_k = int(retrieval_cfg.get("top_k", 5))
        embedding = retrieval_cfg.get("embedding")
        if embedding is not None and not isinstance(embedding, Embeddings):
            raise TypeError(
                "retrieval.embedding must be an instance of LangChain Embeddings"
            )

        try:
            return VectorRetriever(
                chroma_dir=chroma_dir,
                faiss_dir=faiss_dir,
                embedding=embedding,
                top_k=top_k,
            )
        except RuntimeError:
            return None

    def query(self, question: str) -> str:
        """
        Process a query through the corrective RAG pipeline.

        Args:
            question: The input question to answer

        Returns:
            The generated answer
        """
        state = GraphState(question=question)

        retrieval = self._run_retrieval(question)
        if retrieval is not None:
            state.add_documents(retrieval.documents)
            state.metadata["sources"] = retrieval.sources

        state.final_answer = self._draft_answer(state)
        self._last_state = state
        return state.final_answer

    def _run_retrieval(self, question: str) -> Optional[RetrievalResult]:
        if self.retriever is None:
            return None

        return self.retriever.retrieve(question)

    def _draft_answer(self, state: GraphState) -> str:
        if not state.retrieved_documents:
            return (
                "No relevant documents were retrieved. Try ingesting more data or "
                "adjusting your question."
            )

        top_doc = state.retrieved_documents[0]
        source = top_doc.metadata.get("source", "unknown source")
        snippet = top_doc.page_content.strip().replace("\n", " ")[:280]

        return (
            f"Top match from {source}:\n"
            f"{snippet}...\n\n"
            "(Full answer generation not yet implemented.)"
        )

    @property
    def last_state(self) -> Optional[GraphState]:
        """Return the most recent graph state produced by ``query``."""

        return self._last_state

    def add_documents(self, documents: Iterable[str]) -> IngestionResult:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of document texts to add
        """
        if self.ingestion_pipeline is None:
            raise RuntimeError("Ingestion pipeline is not configured.")

        return self.ingestion_pipeline.ingest_documents(documents)

    def ingest_knowledge_base(self) -> IngestionResult:
        """Ingest all resources from the configured knowledge base."""

        if self.ingestion_pipeline is None:
            raise RuntimeError("Ingestion pipeline is not configured.")

        return self.ingestion_pipeline.ingest_knowledge_base()
