"""Graph state definitions for the corrective RAG LangGraph workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document


@dataclass
class GraphState:
    """Captures the evolving state as data flows through the LangGraph pipeline."""

    question: str
    intent: Optional[str] = None
    retrieved_documents: List[Document] = field(default_factory=list)
    preliminary_answer: Optional[str] = None
    critiques: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def add_documents(self, documents: List[Document]) -> None:
        self.retrieved_documents.extend(documents)

    def add_critique(self, critique: str) -> None:
        self.critiques.append(critique)


__all__ = ["GraphState"]
