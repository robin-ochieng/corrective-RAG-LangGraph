"""Document ingestion pipeline for corrective RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class IngestionResult:
    """Represents the outcome of an ingestion operation."""

    processed: int
    failed: int = 0
    details: List[str] = field(default_factory=list)
    vectorstores: List[str] = field(default_factory=list)


class IngestionPipeline:
    """Coordinates document ingestion into the corrective RAG knowledge base."""

    def __init__(
        self,
        config: Optional[dict] = None,
        embedding: Optional[Embeddings] = None,
    ) -> None:
        self.config = config or {}
        self.embedding = embedding

        self.knowledge_base_dir = Path(
            self.config.get("knowledge_base_dir", "knowledge_base")
        )
        self.urls_file = Path(
            self.config.get("urls_file", self.knowledge_base_dir / "urls")
        )
        self.chroma_dir = Path(
            self.config.get("chroma_dir", "vectorstores/chroma")
        )
        self.faiss_dir = Path(self.config.get("faiss_dir", "vectorstores/faiss"))
        self.chunk_size = int(self.config.get("chunk_size", 1000))
        self.chunk_overlap = int(self.config.get("chunk_overlap", 150))

    def ingest_documents(self, documents: Iterable[str]) -> IngestionResult:
        """Ingest raw text snippets into all configured vector stores."""

        docs = [
            Document(
                page_content=text,
                metadata={"source": f"inline::{idx}"},
            )
            for idx, text in enumerate(documents)
            if text.strip()
        ]

        if not docs:
            return IngestionResult(processed=0, details=["No documents provided."])

        return self._upsert_documents(docs)

    def ingest_knowledge_base(self) -> IngestionResult:
        """Load the local knowledge base (PDFs + URLs) and ingest it."""

        documents: List[Document] = []
        failures: List[str] = []

        if self.urls_file.exists():
            url_list = self._read_urls(self.urls_file)
            if url_list:
                try:
                    loader = WebBaseLoader(url_list)
                    url_docs = loader.load()
                    documents.extend(url_docs)
                except Exception as exc:  # pragma: no cover - network failure
                    failures.append(f"Failed to load URLs: {exc}")
            else:
                failures.append("No valid URLs found in urls file.")
        else:
            failures.append(f"URLs file not found: {self.urls_file}")

        for pdf_path in sorted(self.knowledge_base_dir.glob("*.pdf")):
            try:
                loader = PyPDFLoader(str(pdf_path))
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    doc.metadata.setdefault("source", str(pdf_path))
                documents.extend(pdf_docs)
            except Exception as exc:  # pragma: no cover - PDF parsing failure
                failures.append(f"Failed to parse {pdf_path.name}: {exc}")

        if not documents:
            return IngestionResult(processed=0, failed=len(failures), details=failures)

        result = self._upsert_documents(documents)
        result.failed += len(failures)
        result.details.extend(failures)
        return result

    def _upsert_documents(self, documents: Sequence[Document]) -> IngestionResult:
        """Split, embed, and store the supplied documents."""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_docs = text_splitter.split_documents(list(documents))

        if not split_docs:
            return IngestionResult(processed=0, details=["No content after splitting."])

        embeddings = self._get_embeddings()

        # Persist to Chroma
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        chroma_store = Chroma.from_documents(
            split_docs,
            embeddings,
            persist_directory=str(self.chroma_dir),
        )
        if self.config.get("force_persist") and hasattr(chroma_store, "persist"):
            chroma_store.persist()

        # Persist to FAISS
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        faiss_store = FAISS.from_documents(split_docs, embeddings)
        faiss_store.save_local(str(self.faiss_dir))

        sources = sorted({doc.metadata.get("source", "unknown") for doc in documents})
        return IngestionResult(
            processed=len(split_docs),
            details=sources,
            vectorstores=["chroma", "faiss"],
        )

    def _get_embeddings(self) -> Embeddings:
        if self.embedding is not None:
            return self.embedding

        embedding_config = self.config.get("embedding", {})
        model = embedding_config.get("model", "text-embedding-3-small")

        try:
            return OpenAIEmbeddings(model=model)
        except Exception as exc:  # pragma: no cover - requires external service
            raise RuntimeError(
                "Failed to initialise OpenAI embeddings. Ensure your OPENAI_API_KEY "
                "is set or provide a custom Embeddings implementation via the "
                "`embedding` parameter."
            ) from exc

    @staticmethod
    def _read_urls(url_file: Path) -> List[str]:
        urls: List[str] = []
        for raw_line in url_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
        return urls
