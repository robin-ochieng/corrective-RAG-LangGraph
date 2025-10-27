"""LangGraph assembly for the corrective RAG workflow."""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from .nodes import DocumentGraderNode, GenerationNode, GraphState, RetrieveNode, WebSearchNode
from .retrievers import VectorRetriever


def build_corrective_rag_graph(
    *,
    retriever: VectorRetriever,
    config: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> StateGraph[GraphState]:
    """Construct the LangGraph state machine for the corrective RAG pipeline."""

    config = config or {}
    overrides = overrides or {}

    def _resolve_node(name: str, default_factory):
        node = overrides.get(name)
        if node is None:
            node = default_factory()
        return node

    retrieve_node = _resolve_node(
        "retrieve",
        lambda: RetrieveNode(
            retriever=retriever,
            config=config.get("retrieve"),
        ),
    )

    grader_node = _resolve_node(
        "document_grader",
        lambda: DocumentGraderNode(
            config=config.get("document_grader"),
        ),
    )

    web_search_node = _resolve_node(
        "web_search",
        lambda: WebSearchNode(
            config=config.get("web_search"),
        ),
    )

    generation_node = _resolve_node(
        "generation",
        lambda: GenerationNode(
            config=config.get("generation"),
        ),
    )

    graph = StateGraph(GraphState)

    def _node_runner(node):
        if hasattr(node, "run"):
            return node.run
        if callable(node):
            return node
        raise TypeError(f"Node override for {node} must define a 'run' method or be callable.")

    graph.add_node("retrieve", _node_runner(retrieve_node))
    graph.add_node("document_grader", _node_runner(grader_node))
    graph.add_node("web_search", _node_runner(web_search_node))
    graph.add_node("generation", _node_runner(generation_node))

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "document_grader")

    web_search_always = bool(getattr(web_search_node, "always_search", False))

    def _document_grader_route(state: GraphState | Dict[str, Any]) -> str:
        """Determine whether to branch into web search before generation."""

        metadata: Dict[str, Any]
        if isinstance(state, dict):
            metadata = state.get("metadata", {}) or {}
        else:
            metadata = getattr(state, "metadata", {}) or {}

        should_search = bool(metadata.get("web_search_required", False))

        if should_search or web_search_always:
            return "web_search"

        return "generation"

    graph.add_conditional_edges(
        "document_grader",
        _document_grader_route,
        {
            "web_search": "web_search",
            "generation": "generation",
        },
    )

    graph.add_edge("web_search", "generation")
    graph.add_edge("generation", END)

    return graph


__all__ = ["build_corrective_rag_graph"]
