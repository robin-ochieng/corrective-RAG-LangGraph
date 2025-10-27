"""Web search node leveraging Tavily to augment the graph state."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from tavily import TavilyClient

from .base import BaseNode
from .state import GraphState


class WebSearchNode(BaseNode):
    """Augments retrieved evidence by calling the Tavily search API."""

    name = "web_search"

    def __init__(
        self,
        *,
        client: Optional[TavilyClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)
        self.max_results = int(self.config.get("max_results", 5))
        self.search_depth = self.config.get("search_depth", "basic")
        self.include_domains = self.config.get("include_domains")
        self.exclude_domains = self.config.get("exclude_domains")
        self.always_search = bool(self.config.get("always_search", False))
        self.query_template = self.config.get("query_template", "{question}")

        self.client = client or self._build_client()
        self._last_response: Optional[Dict[str, Any]] = None

    def _build_client(self) -> TavilyClient:
        api_key = self.config.get("api_key") or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is required for web search. Set it in the environment or config."
            )
        return TavilyClient(api_key=api_key)

    def run(self, state: GraphState) -> GraphState:  # type: ignore[override]
        if not state.question or not state.question.strip():
            raise ValueError("GraphState.question must be a non-empty string for web search.")

        should_search = self.always_search or state.metadata.get("web_search_required", False)
        if not should_search:
            state.metadata.setdefault("web_search", {"performed": False})
            return state

        query = self.query_template.format(question=state.question)

        request_kwargs: Dict[str, Any] = {
            "query": query,
            "max_results": self.max_results,
            "search_depth": self.search_depth,
        }
        if self.include_domains:
            request_kwargs["include_domains"] = self.include_domains
        if self.exclude_domains:
            request_kwargs["exclude_domains"] = self.exclude_domains

        response = self.client.search(**request_kwargs)
        self._last_response = response

        results = response.get("results", [])
        documents: List[Document] = []

        for item in results:
            content = item.get("content") or item.get("snippet") or ""
            if not content:
                continue

            metadata = {
                "source": item.get("url") or item.get("href"),
                "title": item.get("title"),
                "provider": "tavily",
                "score": item.get("score"),
            }
            documents.append(Document(page_content=content.strip(), metadata=metadata))

        if documents:
            state.add_documents(documents)

        state.metadata.setdefault("web_search", {})
        state.metadata["web_search"].update(
            {
                "performed": True,
                "query": query,
                "results": len(results),
                "documents_added": len(documents),
            }
        )

        state.metadata["web_search_required"] = len(documents) == 0

        return state

    @property
    def last_response(self) -> Optional[Dict[str, Any]]:
        """Return the raw Tavily response from the most recent invocation."""

        return self._last_response


__all__ = ["WebSearchNode"]
