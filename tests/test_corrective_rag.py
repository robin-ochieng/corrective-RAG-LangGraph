"""Tests for the corrective RAG system."""

from pathlib import Path

import pytest

from langchain_core.embeddings import Embeddings

from corrective_rag import CorrectiveRAG, IngestionPipeline


@pytest.fixture
def dummy_embedding() -> "DummyEmbeddings":
    return DummyEmbeddings()


def test_corrective_rag_initialization():
    """Test that CorrectiveRAG can be initialized."""
    rag = CorrectiveRAG()
    assert rag is not None


def test_corrective_rag_query():
    """Test basic query functionality."""
    rag = CorrectiveRAG()
    response = rag.query("What is machine learning?")
    assert isinstance(response, str)
    assert "No relevant documents" in response


def test_corrective_rag_with_config():
    """Test initialization with configuration."""
    config = {"debug": True}
    rag = CorrectiveRAG(config=config)
    assert rag.config == config


class DummyEmbeddings(Embeddings):
    """Simple embeddings implementation for tests."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def embed_documents(self, texts):  # type: ignore[override]
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):  # type: ignore[override]
        value = float(len(text) % (self.dimension or 1))
        return [value + float(index) for index in range(self.dimension)]


def test_ingestion_pipeline_round_trip(tmp_path: Path, dummy_embedding: "DummyEmbeddings"):
    """Ensure the ingestion pipeline persists to both vector stores."""

    config = {
        "chroma_dir": str(tmp_path / "chroma"),
        "faiss_dir": str(tmp_path / "faiss"),
    }
    pipeline = IngestionPipeline(config=config, embedding=dummy_embedding)
    result = pipeline.ingest_documents(["doc 1", "doc 2"])

    assert result.processed == 2
    assert result.failed == 0
    assert (tmp_path / "chroma").exists()
    assert (tmp_path / "faiss").exists()


def test_corrective_rag_retrieval(tmp_path: Path, dummy_embedding: "DummyEmbeddings"):
    """Query should surface content from persisted vector stores."""

    ingestion_config = {
        "chroma_dir": str(tmp_path / "chroma"),
        "faiss_dir": str(tmp_path / "faiss"),
    }

    pipeline = IngestionPipeline(config=ingestion_config, embedding=dummy_embedding)
    pipeline.ingest_documents(
        [
            "Machine learning enables systems to learn from data without explicit programming.",
            "Deep learning is a subset of machine learning that uses neural networks.",
        ]
    )

    rag = CorrectiveRAG(
        config={
            "ingestion": ingestion_config,
            "retrieval": {
                "chroma_dir": ingestion_config["chroma_dir"],
                "faiss_dir": ingestion_config["faiss_dir"],
                "embedding": dummy_embedding,
                "top_k": 3,
            },
        }
    )

    answer = rag.query("What is machine learning?")

    assert "Top match" in answer
    assert rag.last_state is not None
    assert rag.last_state.retrieved_documents
    assert "machine learning" in rag.last_state.retrieved_documents[0].page_content.lower()
