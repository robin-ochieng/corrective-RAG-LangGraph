"""Command-line entry point for the corrective RAG project."""

from __future__ import annotations

import argparse
import sys

from corrective_rag import CorrectiveRAG


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        description="Interact with the corrective RAG pipeline from the command line."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to route through the corrective RAG workflow.",
    )
    parser.add_argument(
        "--document",
        dest="documents",
        action="append",
        help="Document text to ingest before answering the question. Can be repeated.",
    )
    parser.add_argument(
        "--config-debug",
        action="store_true",
        help="Enable debug mode within the CorrectiveRAG configuration.",
    )
    parser.add_argument(
        "--ingest-knowledge-base",
        action="store_true",
        help="Ingest all files found in the configured knowledge base before answering.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    config = {"debug": True} if args.config_debug else None
    rag = CorrectiveRAG(config=config)

    if args.ingest_knowledge_base:
        result = rag.ingest_knowledge_base()
        stores = ", ".join(result.vectorstores) or "no"
        print(
            f"Knowledge base ingestion complete. Processed {result.processed} chunks "
            f"across {stores} stores."
        )

    if args.documents:
        result = rag.add_documents(args.documents)
        stores = ", ".join(result.vectorstores) or "no"
        print(
            f"Ingested {result.processed} inline chunks across {stores} stores."
        )

    if args.question:
        answer = rag.query(args.question)
        print(answer)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
