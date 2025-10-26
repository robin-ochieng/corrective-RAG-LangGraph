"""Entry points for running the ingestion pipeline standalone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from corrective_rag.ingestion import IngestionPipeline, IngestionResult


def build_pipeline(config: dict | None = None) -> IngestionPipeline:
    """Convenience constructor used by CLI or notebooks."""

    return IngestionPipeline(config=config)


def load_config(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None:
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the local knowledge base into Chroma and FAISS stores.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a JSON config file overriding ingestion settings.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = build_pipeline(config=config)
    result = pipeline.ingest_knowledge_base()

    print(
        json.dumps(
            {
                "processed": result.processed,
                "failed": result.failed,
                "details": result.details,
                "vectorstores": result.vectorstores,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["IngestionPipeline", "IngestionResult", "build_pipeline", "main"]
