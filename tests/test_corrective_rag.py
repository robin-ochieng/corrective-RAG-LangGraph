"""Tests for the corrective RAG system."""

from pathlib import Path

import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from corrective_rag import CorrectiveRAG, IngestionPipeline, build_corrective_rag_graph
from corrective_rag.nodes import (
    DocumentGraderNode,
    GenerationNode,
    GraphState,
    RelevanceScore,
    RetrieveNode,
    WebSearchNode,
)
from corrective_rag.retrievers import RetrievalResult, VectorRetriever


@pytest.fixture
def dummy_embedding() -> "DummyEmbeddings":
    return DummyEmbeddings()


@pytest.fixture(autouse=True)
def clear_llm_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


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


def test_retrieve_node_updates_state(
    tmp_path: Path, dummy_embedding: "DummyEmbeddings"
):
    """Retrieve node should populate state documents and metadata."""

    ingestion_config = {
        "chroma_dir": str(tmp_path / "chroma"),
        "faiss_dir": str(tmp_path / "faiss"),
    }

    pipeline = IngestionPipeline(config=ingestion_config, embedding=dummy_embedding)
    pipeline.ingest_documents(
        [
            "Retrieval augmented generation pairs models with retrievers.",
            "LangGraph orchestrates multi-node LLM workflows.",
        ]
    )

    retriever = VectorRetriever(
        chroma_dir=Path(ingestion_config["chroma_dir"]),
        faiss_dir=Path(ingestion_config["faiss_dir"]),
        embedding=dummy_embedding,
        top_k=2,
    )
    node = RetrieveNode(retriever=retriever, top_k=2)

    state = GraphState(question="What is retrieval augmented generation?")
    updated = node.run(state)

    assert updated.retrieved_documents
    assert updated.metadata["retrieval"]["chunk_count"] > 0
    assert node.last_result is not None
    assert node.last_result.query == state.question


class KeywordGrader:
    """Simple grader used for testing."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword.lower()

    def invoke(self, inputs):
        content = inputs["document"].lower()
        score = "yes" if self.keyword in content else "no"
        return RelevanceScore(binary_score=score)


def test_document_grader_filters_documents():
    """Document grader should keep only relevant documents and flag search."""

    docs = [
        Document(
            page_content="Retrieval augmented generation improves LLM outputs.",
            metadata={"source": "doc1"},
        ),
        Document(
            page_content="The history of unrelated topics is discussed here.",
            metadata={"source": "doc2"},
        ),
    ]

    state = GraphState(question="What is retrieval augmented generation?", retrieved_documents=docs)

    grader = DocumentGraderNode(chain=KeywordGrader("retrieval augmented generation"))
    updated = grader.run(state)

    assert len(updated.retrieved_documents) == 1
    assert updated.retrieved_documents[0].metadata["source"] == "doc1"
    assert updated.metadata["retrieval"]["kept_documents"] == 1
    assert updated.metadata["retrieval"]["dropped_documents"] == 1
    assert updated.metadata["web_search_required"] is True

class FakeTavilyClient:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": self.results}


def test_web_search_node_adds_documents():
    client = FakeTavilyClient(
        [
            {
                "title": "Intro to RAG",
                "url": "https://example.com/rag",
                "content": "Retrieval augmented generation combines search and generation.",
                "score": 0.1,
            },
            {
                "title": "Advanced RAG",
                "url": "https://example.com/advanced",
                "content": "Corrective loops refine answers using external knowledge.",
                "score": 0.2,
            },
        ]
    )

    node = WebSearchNode(client=client)
    state = GraphState(
        question="What is corrective RAG?",
        retrieved_documents=[],
        metadata={"web_search_required": True},
    )

    updated = node.run(state)

    assert len(updated.retrieved_documents) == 2
    assert updated.metadata["web_search"]["performed"] is True
    assert updated.metadata["web_search"]["documents_added"] == 2
    assert updated.metadata["web_search_required"] is False
    assert client.calls  # search was executed


def test_web_search_node_handles_empty_results():
    client = FakeTavilyClient([])
    node = WebSearchNode(client=client)
    state = GraphState(
        question="Need new sources",
        retrieved_documents=[],
        metadata={"web_search_required": True},
    )

    updated = node.run(state)

    assert not updated.retrieved_documents
    assert updated.metadata["web_search"]["documents_added"] == 0
    assert updated.metadata["web_search_required"] is True


def test_web_search_node_skips_when_not_required():
    client = FakeTavilyClient([])
    node = WebSearchNode(client=client)
    state = GraphState(question="No search", retrieved_documents=[], metadata={})

    updated = node.run(state)

    assert updated.metadata["web_search"]["performed"] is False
    assert not client.calls


class EchoChain:
    def __init__(self):
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs)
        return f"Answering: {inputs['question']}"


def test_generation_node_produces_answer():
    docs = [
        Document(page_content="RAG blends retrieval with generation.", metadata={"source": "kb1"}),
        Document(page_content="Corrective steps improve reliability.", metadata={"title": "whitepaper"}),
    ]

    state = GraphState(question="What is corrective RAG?", retrieved_documents=docs)
    chain = EchoChain()
    node = GenerationNode(chain=chain, config={"include_sources": True})

    updated = node.run(state)

    assert updated.final_answer.startswith("Answering: What is corrective RAG?")
    assert updated.metadata["generation"]["used_documents"] == 2
    assert node.last_prompt is not None
    assert "[SOURCE: kb1]" in node.last_prompt["context"]
    assert "[SOURCE: whitepaper]" in node.last_prompt["context"]
    assert chain.calls


def test_generation_node_handles_missing_context():
    state = GraphState(question="Explain?", retrieved_documents=[])
    chain = EchoChain()
    node = GenerationNode(chain=chain, config={"include_sources": False})

    updated = node.run(state)

    assert "Explain?" in updated.final_answer
    assert updated.metadata["generation"]["used_documents"] == 0
    assert "No supporting context" in node.last_prompt["context"]


def test_graph_execution(tmp_path: Path, dummy_embedding: "DummyEmbeddings"):
    ingestion_config = {
        "chroma_dir": str(tmp_path / "chroma"),
        "faiss_dir": str(tmp_path / "faiss"),
    }

    pipeline = IngestionPipeline(config=ingestion_config, embedding=dummy_embedding)
    pipeline.ingest_documents(
        [
            "RAG blends retrieval and generation for grounded answers.",
            "Grading and web search improve coverage when local context is insufficient.",
        ]
    )

    retriever = VectorRetriever(
        chroma_dir=Path(ingestion_config["chroma_dir"]),
        faiss_dir=Path(ingestion_config["faiss_dir"]),
        embedding=dummy_embedding,
        top_k=2,
    )

    overrides = {
        "document_grader": DocumentGraderNode(chain=KeywordGrader("rag")),
        "web_search": WebSearchNode(client=FakeTavilyClient([]), config={"always_search": False}),
        "generation": GenerationNode(chain=EchoChain()),
    }

    graph = build_corrective_rag_graph(retriever=retriever, overrides=overrides)
    app = graph.compile()

    initial_state = GraphState(question="How does corrective RAG work?")
    raw_state = app.invoke(initial_state)
    if isinstance(raw_state, GraphState):
        final_state = raw_state
    else:
        final_state = GraphState(**raw_state)

    assert final_state.final_answer is not None
    assert final_state.retrieved_documents
    assert final_state.metadata["generation"]["used_documents"] >= 1


def test_graph_routes_through_web_search_when_required(monkeypatch):
    call_order = []

    class DummyRetriever:
        def retrieve(self, query: str, top_k=None):
            return RetrievalResult(query=query, chunks=[])  # type: ignore[arg-type]

    def fake_retrieve_run(self, state):
        call_order.append("retrieve")
        state.metadata.setdefault("web_search_required", False)
        return state

    def fake_document_grader_run(self, state):
        call_order.append("document_grader")
        state.metadata["web_search_required"] = True
        return state

    def fake_web_search_run(self, state):
        call_order.append("web_search")
        state.metadata["web_search_required"] = False
        return state

    def fake_generation_run(self, state):
        call_order.append("generation")
        state.final_answer = "done"
        return state

    monkeypatch.setattr(DocumentGraderNode, "_build_chain", lambda self, llm: None, raising=False)
    monkeypatch.setattr(WebSearchNode, "_build_client", lambda self: object(), raising=False)
    monkeypatch.setattr(GenerationNode, "_build_chain", lambda self, llm: None, raising=False)

    monkeypatch.setattr(RetrieveNode, "run", fake_retrieve_run, raising=False)
    monkeypatch.setattr(DocumentGraderNode, "run", fake_document_grader_run, raising=False)
    monkeypatch.setattr(WebSearchNode, "run", fake_web_search_run, raising=False)
    monkeypatch.setattr(GenerationNode, "run", fake_generation_run, raising=False)

    graph = build_corrective_rag_graph(retriever=DummyRetriever())
    app = graph.compile()

    state = GraphState(question="Need search")
    app.invoke(state)

    assert call_order == ["retrieve", "document_grader", "web_search", "generation"]


def test_graph_skips_web_search_when_not_required(monkeypatch):
    call_order = []

    class DummyRetriever:
        def retrieve(self, query: str, top_k=None):
            return RetrievalResult(query=query, chunks=[])  # type: ignore[arg-type]

    def fake_retrieve_run(self, state):
        call_order.append("retrieve")
        return state

    def fake_document_grader_run(self, state):
        call_order.append("document_grader")
        state.metadata["web_search_required"] = False
        return state

    def fake_web_search_run(self, state):
        raise AssertionError("Web search should not run when not required")

    def fake_generation_run(self, state):
        call_order.append("generation")
        state.final_answer = "done"
        return state

    monkeypatch.setattr(DocumentGraderNode, "_build_chain", lambda self, llm: None, raising=False)
    monkeypatch.setattr(WebSearchNode, "_build_client", lambda self: object(), raising=False)
    monkeypatch.setattr(GenerationNode, "_build_chain", lambda self, llm: None, raising=False)

    monkeypatch.setattr(RetrieveNode, "run", fake_retrieve_run, raising=False)
    monkeypatch.setattr(DocumentGraderNode, "run", fake_document_grader_run, raising=False)
    monkeypatch.setattr(WebSearchNode, "run", fake_web_search_run, raising=False)
    monkeypatch.setattr(GenerationNode, "run", fake_generation_run, raising=False)

    graph = build_corrective_rag_graph(retriever=DummyRetriever())
    app = graph.compile()

    state = GraphState(question="Skip search")
    app.invoke(state)

    assert call_order == ["retrieve", "document_grader", "generation"]


def test_corrective_rag_requires_credentials_for_llm_nodes(
    tmp_path: Path, dummy_embedding: "DummyEmbeddings", monkeypatch
):
    ingestion_config = {
        "chroma_dir": str(tmp_path / "chroma"),
        "faiss_dir": str(tmp_path / "faiss"),
    }

    pipeline = IngestionPipeline(config=ingestion_config, embedding=dummy_embedding)
    pipeline.ingest_documents(["RAG improves grounded answers."])

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    config = {
        "retrieval": {
            "chroma_dir": ingestion_config["chroma_dir"],
            "faiss_dir": ingestion_config["faiss_dir"],
            "embedding": dummy_embedding,
        },
        "graph": {"use_llm_nodes": True},
    }

    with pytest.raises(EnvironmentError):
        CorrectiveRAG(config=config)


def test_corrective_rag_query_with_llm_nodes(
    tmp_path: Path, dummy_embedding: "DummyEmbeddings", monkeypatch
):
    ingestion_config = {
        "chroma_dir": str(tmp_path / "chroma"),
        "faiss_dir": str(tmp_path / "faiss"),
    }

    pipeline = IngestionPipeline(config=ingestion_config, embedding=dummy_embedding)
    pipeline.ingest_documents(
        [
            "Retrieval augmented generation (RAG) combines search and generation.",
            "This paragraph covers unrelated history.",
        ]
    )

    fake_client = FakeTavilyClient(
        [
            {
                "title": "Supplementary RAG context",
                "url": "https://example.com/context",
                "content": "Corrective RAG expands on RAG with grading and web search.",
                "score": 0.3,
            }
        ]
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.setattr(
        DocumentGraderNode,
        "_build_chain",
        lambda self, llm: KeywordGrader("rag"),
        raising=False,
    )
    monkeypatch.setattr(
        WebSearchNode,
        "_build_client",
        lambda self: fake_client,
        raising=False,
    )
    monkeypatch.setattr(
        GenerationNode,
        "_build_chain",
        lambda self, llm: EchoChain(),
        raising=False,
    )

    config = {
        "retrieval": {
            "chroma_dir": ingestion_config["chroma_dir"],
            "faiss_dir": ingestion_config["faiss_dir"],
            "embedding": dummy_embedding,
            "top_k": 2,
        },
        "graph": {
            "use_llm_nodes": True,
            "web_search": {"always_search": True},
        },
    }

    rag = CorrectiveRAG(config=config)
    answer = rag.query("What is RAG?")

    assert answer.startswith("Answering: What is RAG?")
    assert rag.last_state is not None
    assert "strategy" not in rag.last_state.metadata["generation"]
    assert rag._using_llm_nodes is True
    assert fake_client.calls
