"""Convenience CLI for querying the corrective RAG pipeline with LLM synthesis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from corrective_rag import CorrectiveRAG
from corrective_rag.nodes import GraphState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to ask the corrective RAG pipeline")
    parser.add_argument(
        "--chroma-dir",
        default="vectorstores/chroma",
        type=Path,
        help="Directory containing the persisted Chroma index.",
    )
    parser.add_argument(
        "--faiss-dir",
        default="vectorstores/faiss",
        type=Path,
        help="Directory containing the persisted FAISS index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of documents to retrieve for each query.",
    )
    parser.add_argument(
        "--always-search",
        action="store_true",
        help="Force the web-search node to run even when local documents are available.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict:
    graph_config: Dict = {
        "web_search": {"always_search": args.always_search},
        "use_llm_nodes": True,
    }

    return {
        "retrieval": {
            "chroma_dir": str(args.chroma_dir),
            "faiss_dir": str(args.faiss_dir),
            "top_k": args.top_k,
        },
        "graph": graph_config,
    }


def main() -> None:
    args = parse_args()
    config = build_config(args)
    rag = CorrectiveRAG(config=config)

    answer = rag.query(args.question)

    print("Question:", args.question)
    print("\nAnswer:\n" + answer)

    state = rag.last_state
    if state is None:
        print("\n(No graph state captured.)")
        return

    print("\nMetadata:")
    for key, value in state.metadata.items():
        print(f"- {key}: {value}")

    if state.retrieved_documents:
        print("\nRetrieved documents:")
        for idx, doc in enumerate(state.retrieved_documents, start=1):
            source = doc.metadata.get("source") or doc.metadata.get("title") or "unknown"
            preview = doc.page_content.strip().replace("\n", " ")[:160]
            print(f"  {idx}. {source} :: {preview}")
    else:
        print("\n(No documents retrieved.)")


if __name__ == "__main__":
    main()
