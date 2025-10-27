"""Retrieve node for the corrective RAG LangGraph workflow."""

from __future__ import annotations

from typing import Optional

from .base import BaseNode
from .state import GraphState
from ..retrievers import RetrievalResult, VectorRetriever


class RetrieveNode(BaseNode):
    """Fetches relevant documents for the current graph state."""

    name = "retrieve"

    def __init__(
        self,
        retriever: VectorRetriever,
        *,
        top_k: Optional[int] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(config=config)
        self.retriever = retriever
        self.top_k = top_k or self.config.get("top_k")
        self._last_result: Optional[RetrievalResult] = None

    def run(self, state: GraphState) -> GraphState:  # type: ignore[override]
        if not state.question or not state.question.strip():
            raise ValueError("GraphState.question must be a non-empty string.")

        result = self.retriever.retrieve(state.question, top_k=self.top_k)
        self._last_result = result

        if result.chunks:
            state.add_documents(result.documents)

        state.metadata.setdefault("retrieval", {})
        state.metadata["retrieval"].update(
            {
                "query": result.query,
                "sources": result.sources,
                "chunk_count": len(result.chunks),
            }
        )

        return state

    @property
    def last_result(self) -> Optional[RetrievalResult]:
        """Expose the most recent retrieval output."""

        return self._last_result


__all__ = ["RetrieveNode"]
