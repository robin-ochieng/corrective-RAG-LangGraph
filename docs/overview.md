# Corrective RAG with LangGraph – High-Level Overview

This document provides a big-picture view of the project so you can quickly understand how the pieces fit together and what still lies ahead as the corrective RAG system evolves.

## Vision

The goal is to build a Retrieval-Augmented Generation (RAG) system that doesn’t just answer questions, but actively corrects itself. LangGraph orchestrates the workflow, enabling the pipeline to grade intermediate results, branch when signals indicate uncertainty, and potentially call auxiliary tools like web search for fresh evidence.

## Key Components

### 1. Knowledge Ingestion
- **Location:** `src/corrective_rag/ingestion/`
- **Purpose:** Consolidate documents from local PDFs (`knowledge_base/`) and curated URLs into vector stores.
- **Workflow:**
  1. Load raw sources (using `PyPDFLoader`, `WebBaseLoader`).
  2. Chunk and embed content via `RecursiveCharacterTextSplitter` and embeddings (`OpenAIEmbeddings` by default, pluggable).
  3. Persist vectors to **Chroma** and **FAISS** under `vectorstores/`.
- **CLI:** `poetry run python ingestion/pipeline.py` triggers full ingestion.

### 2. Retrieval Layer
- **Location:** `src/corrective_rag/retrievers/`
- **Core Class:** `VectorRetriever`
- **Highlights:**
  - Loads both Chroma and FAISS vector stores.
  - Performs similarity search across stores, deduplicates results, and provides scored chunks with source metadata.
  - Gracefully handles store/embedding mismatches.

### 3. LangGraph Nodes (Current Progress)
- **RetrieveNode (`nodes/retrieve.py`):**
  - Reads the question from `GraphState`, fetches relevant documents via the retriever, and stores them on the state.
- **DocumentGraderNode (`nodes/document_grader.py`):**
  - Runs an LLM-powered relevance grader with structured Pydantic output (or a heuristic grader in offline mode).
  - Filters out non-relevant documents and flags the state for web search when the kept context is too thin.
- **WebSearchNode (`nodes/web_search.py`):**
  - Calls the Tavily API when more evidence is required and appends fresh documents to the state.
  - Records query metadata and toggles the `web_search_required` flag depending on whether useful context was found.
- **GenerationNode (`nodes/generation.py`):**
  - Crafts the final answer using a RAG prompt, combining retrieved and web-sourced context with the user’s question.
  - Supports configurable LLMs, source annotations, and exposes the last prompt for observability. Falls back to a heuristic answer generator when LLM credentials are absent.
- **GraphState (`nodes/state.py`):**
  - The shared data structure carrying question, retrieved docs, critiques, flags, and metadata through the graph.

### 4. Core Orchestrator
- **File:** `src/corrective_rag/core.py`
- **Responsibilities:**
  - Initialize ingestion pipeline and retriever.
  - Build and compile the LangGraph via `build_corrective_rag_graph`, wiring in heuristic node overrides when API keys are unavailable.
  - Auto-detect whether OpenAI/Tavily credentials are available: real LLM/web-search nodes are enabled when keys exist (or when `graph.use_llm_nodes=True`), otherwise heuristic stand-ins keep the pipeline offline-friendly.
  - Provide the high-level `query` interface that now executes the compiled graph (with conditional branching into web search) and falls back to the legacy retrieval flow on error.
  - Store the latest `GraphState` for inspection, including metadata about retrieval, grading, web search, and generation.

### 5. Command-Line Entry Point
- **File:** `main.py`
- **Capabilities:**
  - Ingest the full knowledge base (`--ingest-knowledge-base`).
  - Ingest ad-hoc documents (`--document`).
  - Ask a question via `CorrectiveRAG` and print the response summary.

### 6. Tests
- **Location:** `tests/`
- **Coverage:**
  - Ingestion pipeline persistence to Chroma/FAISS.
  - Retriever integration and deduplication.
  - Retrieve node state updates.
  - Document grader filtering behaviour.

## Data & Configuration Layout
```
corrective-rag-langgraph/
├── knowledge_base/        # PDFs and URL list
├── vectorstores/          # Persisted embeddings (Chroma + FAISS)
├── src/corrective_rag/
│   ├── ingestion/         # Loaders, chunking, persistence
│   ├── retrievers/        # Vector retrieval utilities
│   ├── nodes/             # LangGraph nodes & state definitions
│   └── core.py            # Main orchestrator
└── docs/                  # Project documentation (this file)
```

## Current Behaviour Flow
1. Ingest knowledge base (optional, but required for retrieval).
2. Build the LangGraph via `build_corrective_rag_graph` (or rely on `CorrectiveRAG` to do so lazily).
3. Execute the graph, which currently flows through:
  - `RetrieveNode`
  - `DocumentGraderNode` → if documents are dropped, branch to `WebSearchNode`
  - `WebSearchNode` (optional, conditional on `web_search_required`)
  - `GenerationNode`
- `DocumentGraderNode` → if documents are dropped, branch to `WebSearchNode`
- `WebSearchNode` (optional, conditional on `web_search_required`)
- `GenerationNode`
4. Inspect the final `GraphState` for diagnostics, retrieved docs, and metadata flags.

### Credential Handling & Overrides
- Set `OPENAI_API_KEY` and `TAVILY_API_KEY` (or provide `graph.openai_api_key` / `graph.tavily_api_key`) to activate the full corrective RAG path automatically.
- Optionally set `graph.user_agent` (or `USER_AGENT`) so outbound tool requests identify the application; defaults to `corrective-rag-langgraph/0.1.0` when not provided.
- Without credentials, the system swaps in lightweight heuristics (`HeuristicDocumentGraderChain`, `OfflineWebSearchNode`, `SimpleGenerationNode`) so local workflows continue to function.
- Force a specific behaviour via `graph.use_llm_nodes=True/False`. When set to `True`, missing credentials raise an error unless custom overrides are supplied.
- Provide bespoke components through `graph.overrides` (e.g., inject a custom retriever grader, web-search client, or generator) which also sidesteps the default credential checks for those nodes.

## Pending & Upcoming Work
- **LLM-backed execution path:** Allow end users to opt into the full LLM nodes via `config['graph']['use_llm_nodes']=True` (currently defaults to heuristic substitutes).
- **Reasoning / correction loops:** Add nodes for iterative drafts, critique, self-correction, and retries beyond the linear path.
- **Robust tool integrations:** Expand web search/tooling coverage (Tavily is plugged in but optional), add evaluators and telemetry hooks.
- **Configuration ergonomics:** Surface richer config (env files, CLI) for swapping embeddings, LLMs, and node overrides without code edits.
- **Quality evaluation:** Add automated metrics/tests around answer quality and corrective behaviour.

## Getting Started Checklist
1. `poetry install`
2. Add secrets to `.env` (`OPENAI_API_KEY`, `TAVILY_API_KEY`, etc.).
3. `poetry run python ingestion/pipeline.py`
4. `poetry run python main.py --ingest-knowledge-base "What is RAG?"`
5. Inspect `last_state` to review retrieved documents, grading signals, and flags.

---
As new nodes and branching logic are introduced, update this overview to reflect the expanded flow. The aim is to keep the architecture transparent so collaborators can reason about the pipeline and contribute confidently.
