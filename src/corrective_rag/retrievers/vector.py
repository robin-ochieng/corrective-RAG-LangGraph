"""Vector-based retrieval utilities for corrective RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS


@dataclass
class RetrievedChunk:
    """Represents a single retrieved chunk with score metadata."""

    document: Document
    score: float
    source: str


@dataclass
class RetrievalResult:
    """Aggregated retrieval output."""

    query: str
    chunks: List[RetrievedChunk]

    @property
    def documents(self) -> List[Document]:
        return [chunk.document for chunk in self.chunks]

    @property
    def sources(self) -> List[str]:
        return [chunk.source for chunk in self.chunks]


class VectorRetriever:
    """Retrieves relevant chunks from multiple vector stores."""

    def __init__(
        self,
        chroma_dir: Path,
        faiss_dir: Path,
        embedding: Optional[Embeddings] = None,
        top_k: int = 5,
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.faiss_dir = Path(faiss_dir)
        self.embedding = embedding or self._default_embeddings()
        self.top_k = top_k

        self._chroma = self._load_chroma_store()
        self._faiss = self._load_faiss_store()

        if self._chroma is None and self._faiss is None:
            raise RuntimeError(
                "No vector stores could be loaded. Ensure ingestion has been run."
            )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve the most relevant chunks for the given query."""

        limit = top_k or self.top_k
        results: List[Tuple[Document, float]] = []

        if self._chroma is not None:
            try:
                results.extend(
                    self._chroma.similarity_search_with_score(query, k=limit)
                )
            except Exception:
                # If embeddings are incompatible (e.g., different dimension), skip store.
                pass

        if self._faiss is not None:
            try:
                results.extend(
                    self._faiss.similarity_search_with_score(query, k=limit)
                )
            except Exception:
                pass

        combined = self._rank_and_deduplicate(results, limit)

        return RetrievalResult(
            query=query,
            chunks=[
                RetrievedChunk(
                    document=doc,
                    score=score,
                    source=str(doc.metadata.get("source", "unknown")),
                )
                for doc, score in combined
            ],
        )

    def _rank_and_deduplicate(
        self, pairs: Sequence[Tuple[Document, float]], limit: int
    ) -> List[Tuple[Document, float]]:
        seen: set[str] = set()
        ranked: List[Tuple[Document, float]] = []

        for doc, score in sorted(pairs, key=lambda item: item[1]):
            key = f"{doc.metadata.get('source', '')}::{hash(doc.page_content)}"
            if key in seen:
                continue
            seen.add(key)
            ranked.append((doc, score))
            if len(ranked) >= limit:
                break

        return ranked

    def _default_embeddings(self) -> Embeddings:
        model = "text-embedding-3-small"
        return OpenAIEmbeddings(model=model)

    def _load_chroma_store(self):
        if not self.chroma_dir.exists():
            return None

        try:
            return Chroma(
                persist_directory=str(self.chroma_dir),
                embedding_function=self.embedding,
            )
        except Exception:
            return None

    def _load_faiss_store(self):
        if not self.faiss_dir.exists():
            return None

        try:
            return FAISS.load_local(
                str(self.faiss_dir),
                embeddings=self.embedding,
                allow_dangerous_deserialization=True,
            )
        except ValueError:
            return None


__all__ = ["RetrievedChunk", "RetrievalResult", "VectorRetriever"]
