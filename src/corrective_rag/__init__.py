"""Corrective RAG with LangGraph."""

import os

__version__ = "0.1.0"
__author__ = "Robin Ochieng"

_DEFAULT_USER_AGENT = "corrective-rag-langgraph/0.1.0"
if not os.getenv("USER_AGENT"):
	os.environ["USER_AGENT"] = _DEFAULT_USER_AGENT

from .core import CorrectiveRAG
from .graph import build_corrective_rag_graph
from .ingestion import IngestionPipeline, IngestionResult
from .nodes import (
	DocumentGraderNode,
	GenerationNode,
	GraphState,
	RelevanceScore,
	WebSearchNode,
)
from .retrievers import RetrievalResult, VectorRetriever

__all__ = [
	"CorrectiveRAG",
	"DocumentGraderNode",
	"GenerationNode",
	"GraphState",
	"IngestionPipeline",
	"IngestionResult",
	"build_corrective_rag_graph",
	"RetrievalResult",
	"RelevanceScore",
	"WebSearchNode",
	"VectorRetriever",
]
