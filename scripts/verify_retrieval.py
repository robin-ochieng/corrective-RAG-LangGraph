"""Manual verification that CorrectiveRAG retrieves from persisted vector stores."""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.embeddings import Embeddings

from corrective_rag import CorrectiveRAG, IngestionPipeline


class DemoEmbeddings(Embeddings):
    """Tiny deterministic embedding implementation suitable for tests."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def embed_documents(self, texts):  # type: ignore[override]
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):  # type: ignore[override]
        base = float(len(text) % self.dimension)
        return [base + float(idx) for idx in range(self.dimension)]


if __name__ == "__main__":
    embedding = DemoEmbeddings()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        chroma_dir = tmp_path / "chroma"
        faiss_dir = tmp_path / "faiss"

        pipeline = IngestionPipeline(
            config={
                "chroma_dir": str(chroma_dir),
                "faiss_dir": str(faiss_dir),
            },
            embedding=embedding,
        )

        pipeline.ingest_documents(
            [
                "Generative AI models create content like text, images, or audio.",
                "Retrieval augmented generation grounds responses using a knowledge base.",
            ]
        )

        rag = CorrectiveRAG(
            config={
                "retrieval": {
                    "chroma_dir": str(chroma_dir),
                    "faiss_dir": str(faiss_dir),
                    "embedding": embedding,
                    "top_k": 2,
                },
                "graph": {
                    "use_llm_nodes": False,
                },
            }
        )

        question = "What does retrieval augmented generation do?"
        answer = rag.query(question)

        print("Question:", question)
        print("Answer:", answer)
        print("\nRetrieved documents:")
        if rag.last_state and rag.last_state.retrieved_documents:
            for doc in rag.last_state.retrieved_documents:
                print("-", doc.page_content[:80].replace("\n", " "), "| source:", doc.metadata.get("source"))
        else:
            print("(No documents retrieved)")

        print("\nMetadata:")
        if rag.last_state:
            print(rag.last_state.metadata)
        else:
            print("(Graph state unavailable)")
