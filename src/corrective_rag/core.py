"""Core Corrective RAG implementation."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from .graph import build_corrective_rag_graph
from .ingestion import IngestionPipeline, IngestionResult
from .nodes import GraphState, RelevanceScore
from .nodes.base import BaseNode
from .nodes.document_grader import DocumentGraderNode
from .retrievers import RetrievalResult, VectorRetriever

# Load environment variables
load_dotenv()


DEFAULT_USER_AGENT = "corrective-rag-langgraph/0.1.0"

if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = DEFAULT_USER_AGENT


logger = logging.getLogger(__name__)


class HeuristicDocumentGraderChain:
    """Lightweight scorer that keeps documents containing question keywords."""

    def __init__(self, min_keyword_length: int = 3) -> None:
        self.min_keyword_length = min_keyword_length

    def invoke(self, inputs) -> RelevanceScore:
        question = inputs.get("question", "")
        document = inputs.get("document", "")

        keywords = [
            token
            for token in re.findall(r"\b\w+\b", question.lower())
            if len(token) >= self.min_keyword_length
        ]

        if not keywords:
            return RelevanceScore(binary_score="yes")

        doc_text = document.lower()
        relevant = any(keyword in doc_text for keyword in keywords)
        return RelevanceScore(binary_score="yes" if relevant else "no")


class OfflineWebSearchNode(BaseNode):
    """No-op web search node used when an external API is unavailable."""

    name = "web_search"

    def __init__(self, *, config: Optional[Dict] = None) -> None:
        super().__init__(config=config)
        self.always_search = bool(self.config.get("always_search", False))

    def run(self, state: GraphState) -> GraphState:  # type: ignore[override]
        should_search = self.always_search or state.metadata.get("web_search_required", False)

        state.metadata.setdefault("web_search", {})
        state.metadata["web_search"].update(
            {
                "performed": bool(should_search),
                "query": None,
                "results": 0,
                "documents_added": 0,
            }
        )

        state.metadata["web_search_required"] = False
        return state


class SimpleGenerationNode(BaseNode):
    """Heuristic answer generator mirroring the legacy fallback response."""

    name = "generation"

    def __init__(self, *, config: Optional[Dict] = None) -> None:
        super().__init__(config=config)

    def run(self, state: GraphState) -> GraphState:  # type: ignore[override]
        used_documents = len(state.retrieved_documents)

        if used_documents == 0:
            answer = (
                "No relevant documents were retrieved. Try ingesting more data or "
                "adjusting your question."
            )
            context_chars = 0
        else:
            top_doc = state.retrieved_documents[0]
            snippet = top_doc.page_content.strip().replace("\n", " ")[:280]
            source = top_doc.metadata.get("source", "unknown source")
            answer = (
                f"Top match from {source}:\n"
                f"{snippet}...\n\n"
                "(Full answer generation not yet implemented.)"
            )
            context_chars = sum(len(doc.page_content) for doc in state.retrieved_documents)

        state.final_answer = answer
        state.metadata.setdefault("generation", {})
        state.metadata["generation"].update(
            {
                "used_documents": used_documents,
                "context_chars": context_chars,
                "strategy": "simple_fallback",
            }
        )

        return state


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
        self._graph_app = None
        self._using_llm_nodes = False
        self._setup_components()

    def _set_optional_env_values(self, graph_config: Dict[str, Any]) -> None:
        openai_key = graph_config.get("openai_api_key")
        if openai_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_key

        tavily_key = graph_config.get("tavily_api_key")
        if tavily_key and not os.getenv("TAVILY_API_KEY"):
            os.environ["TAVILY_API_KEY"] = tavily_key

        user_agent = graph_config.get("user_agent") or DEFAULT_USER_AGENT
        if user_agent and not os.getenv("USER_AGENT"):
            os.environ["USER_AGENT"] = user_agent

    def _has_openai_credentials(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def _has_tavily_credentials(self) -> bool:
        return bool(os.getenv("TAVILY_API_KEY"))

    def _missing_credentials(self, overrides: Dict[str, Any]) -> List[str]:
        missing: List[str] = []

        openai_needed = (
            "document_grader" not in overrides or overrides.get("document_grader") is None
        ) or (
            "generation" not in overrides or overrides.get("generation") is None
        )

        if openai_needed and not self._has_openai_credentials():
            missing.append("OPENAI_API_KEY")

        tavily_needed = "web_search" not in overrides or overrides.get("web_search") is None
        if tavily_needed and not self._has_tavily_credentials():
            missing.append("TAVILY_API_KEY")

        return missing

    def _determine_llm_usage(
        self,
        graph_config: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        missing = self._missing_credentials(overrides)
        explicit = graph_config.get("use_llm_nodes")

        if explicit is None:
            return len(missing) == 0, missing

        use_llm = bool(explicit)
        if use_llm and missing:
            creds = ", ".join(missing)
            raise EnvironmentError(
                "use_llm_nodes=True requires configured credentials. Missing: " f"{creds}"
            )

        return use_llm, missing

    def _setup_components(self):
        """Set up the various components of the RAG system."""
        self.ingestion_pipeline = IngestionPipeline(
            config=self.config.get("ingestion")
        )
        self.retriever = self._build_retriever()

        self._graph_app = None
        self._using_llm_nodes = False

        if self.retriever is None:
            return

        graph_section = self.config.get("graph", {})
        user_overrides = graph_section.get("overrides") or {}

        self._set_optional_env_values(graph_section)

        use_llm_nodes, missing_credentials = self._determine_llm_usage(
            graph_section, user_overrides
        )

        if not use_llm_nodes and graph_section.get("use_llm_nodes") is None:
            if missing_credentials:
                logger.info(
                    "Falling back to heuristic nodes; missing credentials: %s",
                    ", ".join(missing_credentials),
                )

        default_overrides = {}
        if not use_llm_nodes:
            heuristic_chain = HeuristicDocumentGraderChain()
            default_overrides = {
                "document_grader": DocumentGraderNode(
                    chain=heuristic_chain,
                    config=graph_section.get("document_grader"),
                ),
                "web_search": OfflineWebSearchNode(
                    config=graph_section.get("web_search"),
                ),
                "generation": SimpleGenerationNode(
                    config=graph_section.get("generation"),
                ),
            }

        # User-provided overrides take precedence over defaults.
        node_overrides = {**default_overrides, **user_overrides}

        try:
            graph = build_corrective_rag_graph(
                retriever=self.retriever,
                config=graph_section,
                overrides=node_overrides if node_overrides else None,
            )
            self._graph_app = graph.compile()
            self._using_llm_nodes = use_llm_nodes and self._graph_app is not None
        except Exception:
            # Fall back to legacy behaviour if graph setup fails.
            self._graph_app = None
            self._using_llm_nodes = False

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
        except Exception as exc:
            logger.info("Vector retriever unavailable: %s", exc)
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

        if self._graph_app is not None:
            try:
                result = self._graph_app.invoke(state)
                if isinstance(result, GraphState):
                    final_state = result
                else:
                    final_state = GraphState(**result)
            except Exception:
                return self._fallback_flow(state).final_answer
            else:
                if not final_state.final_answer:
                    final_state.final_answer = self._draft_answer(final_state)
                self._last_state = final_state
                return final_state.final_answer

        final_state = self._fallback_flow(state)
        return final_state.final_answer

    def _run_retrieval(self, question: str) -> Optional[RetrievalResult]:
        if self.retriever is None:
            return None

        return self.retriever.retrieve(question)

    def _fallback_flow(self, state: GraphState) -> GraphState:
        retrieval = self._run_retrieval(state.question)
        if retrieval is not None:
            state.add_documents(retrieval.documents)
            state.metadata["sources"] = retrieval.sources

        state.final_answer = self._draft_answer(state)
        self._last_state = state
        return state

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
