# 🏗️ Robinhood RAG - Project Architecture

This document provides a detailed technical overview of the Robinhood RAG system architecture, components, and design decisions.

## Table of Contents
- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Tracing & Observability](#tracing--observability)
- [Directory Structure](#directory-structure)
- [Key Components](#key-components)

---

## System Overview

Robinhood RAG is an **Agentic Retrieval-Augmented Generation** system designed to answer questions about Robinhood's products and services by:

1. **Scraping** Robinhood support documentation
2. **Indexing** content using hybrid search (vector + keyword)
3. **Retrieving** relevant context using Reciprocal Rank Fusion
4. **Generating** natural language responses with citations

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            STREAMLIT UI                                  │
│                     (app.py - Chat Interface)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG AGENT LAYER                                 │
│  ┌────────────────┐    ┌─────────────────┐    ┌────────────────────┐   │
│  │ Query Rewriter │──▶ │   Agent Loop    │──▶ │   Response Gen     │   │
│  │   (Chain)      │    │  (ReAct Flow)   │    │   with Citations   │   │
│  └────────────────┘    └─────────────────┘    └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────┐     ┌─────────────────────────────────────┐
│       HYBRID SEARCH         │     │          LLM CLIENT                  │
│  ┌──────────┬──────────┐    │     │  ┌────────────────────────────────┐ │
│  │ ChromaDB │   BM25   │    │     │  │    OpenAI-Compatible API       │ │
│  │ (Vector) │(Keyword) │    │     │  │  (DeepSeek / OpenAI / Local)   │ │
│  └────┬─────┴────┬─────┘    │     │  └────────────────────────────────┘ │
│       └────┬─────┘          │     └─────────────────────────────────────┘
│            ▼                │
│  ┌─────────────────────┐    │
│  │  RRF Fusion (k=60)  │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       ARIZE PHOENIX                                      │
│               (Distributed Tracing & Observability)                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  rag_request (Parent)                                              │  │
│  │    ├── query_rewrite (Chain)                                       │  │
│  │    ├── hybrid_search (Tool)                                        │  │
│  │    ├── ChatCompletion (LLM) [auto-instrumented]                    │  │
│  │    └── ...                                                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Scraper Layer (`src/scraper/`)

| Component | File | Description |
|-----------|------|-------------|
| Web Crawler | `crawler.py` | Uses crawl4ai to scrape Robinhood support pages |

**Features:**
- Async crawling with rate limiting
- Markdown extraction from HTML
- Metadata preservation (title, URL, timestamps)

### 2. Processing Layer (`src/processing/`)

| Component | File | Description |
|-----------|------|-------------|
| Text Chunker | `chunker.py` | LangChain RecursiveTextSplitter |

**Configuration:**
- Chunk size: 1000 characters
- Overlap: 200 characters
- Output: `data/processed/chunks.json`

### 3. Indexing Layer (`src/indexing/`)

| Component | File | Description |
|-----------|------|-------------|
| Vector Store | `vector_store.py` | ChromaDB with sentence-transformers embeddings |
| Keyword Store | `keyword_store.py` | BM25 index with rank-bm25 library |
| Hybrid Search | `hybrid_search.py` | Reciprocal Rank Fusion of both stores |

**Hybrid Search Formula:**
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
where k = 60 (fusion constant)
```

### 4. LLM Layer (`src/llm/`)

| Component | File | Description |
|-----------|------|-------------|
| LLM Client | `client.py` | OpenAI-compatible API wrapper |
| RAG Agent | `agent.py` | ReAct-style agentic flow with tracing |

**Agent Capabilities:**
- Query rewriting (context-aware)
- Multi-turn reasoning with `<thinking>` tags
- Tool calling (`<search>` tag)
- Streaming responses
- Hierarchical tracing

### 5. Telemetry Layer (`src/telemetry/`)

| Component | File | Description |
|-----------|------|-------------|
| Phoenix Tracer | `phoenix_tracer.py` | OpenTelemetry + Phoenix setup |

---

## Data Flow

### Indexing Pipeline
```
Robinhood Website → Crawler → Raw Markdown → Chunker → chunks.json
                                                          │
                    ┌─────────────────────────────────────┴────────────────────┐
                    ▼                                                           ▼
              ChromaDB                                                        BM25
        (Embeddings: all-MiniLM-L6-v2)                               (Tokenized Terms)
```

### Query Pipeline
```
User Question → Query Rewriter → Standalone Query → Hybrid Search → Context
                                                                        │
                                                                        ▼
                                                           Agent Loop (max 5 iterations)
                                                                        │
                                                                        ▼
                                                                 Final Answer + Sources
```

---

## Tracing & Observability

The application uses **Arize Phoenix** for full observability with **hierarchical tracing**.

### Trace Hierarchy
```
rag_request (AGENT) ─────────────────────────────┐
│                                                 │
├── query_rewrite (CHAIN)                        │
│     └── ChatCompletion (LLM)                   │
│                                                 │
├── hybrid_search (TOOL)                         │
│     ├── ChromaDB query                         │
│     └── BM25 query                             │
│                                                 │
└── ChatCompletion (LLM) [reasoning/answer]      │
──────────────────────────────────────────────────┘
```

### Span Attributes

| Span | Kind | Attributes |
|------|------|------------|
| `rag_request` | AGENT | input.value, output.value |
| `query_rewrite` | CHAIN | input.value (original), output.value (rewritten) |
| `hybrid_search` | TOOL | tool.name, tool.description, result count |
| `ChatCompletion` | LLM | Auto-instrumented by OpenAIInstrumentor |

---

## Directory Structure

```
Robinhood RAG/
├── app.py                    # Streamlit UI entry point
├── config.py                 # Pydantic settings management
├── build_index.py            # Standalone indexing script
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
│
├── src/
│   ├── __init__.py
│   ├── scraper/
│   │   ├── __init__.py
│   │   └── crawler.py        # crawl4ai web scraper
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   └── chunker.py        # LangChain text chunker
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── vector_store.py   # ChromaDB vector store
│   │   ├── keyword_store.py  # BM25 keyword index
│   │   └── hybrid_search.py  # RRF hybrid search
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py         # OpenAI-compatible LLM client
│   │   └── agent.py          # Agentic RAG with tracing
│   │
│   └── telemetry/
│       ├── __init__.py
│       └── phoenix_tracer.py # Phoenix/OTEL setup
│
├── data/
│   ├── raw/                  # Scraped markdown files
│   └── processed/            # chunks.json
│
└── indexes/
    ├── chroma/               # ChromaDB persistent storage
    └── bm25/                 # BM25 index files
```

---

## Key Components

### RAG Agent (`src/llm/agent.py`)

The core agent implements a **ReAct-style** reasoning loop:

1. **Query Rewriting**: Converts context-dependent queries to standalone questions
2. **Reasoning Loop**: Uses `<thinking>` tags for step-by-step reasoning
3. **Tool Calling**: Uses `<search>` tags to invoke hybrid search
4. **Answer Generation**: Uses `<answer>` tags for final response

**Tracing Integration:**
- Parent span `rag_request` wraps entire request
- Child spans for each component (`query_rewrite`, `hybrid_search`)
- LLM calls auto-instrumented by OpenAIInstrumentor

### Hybrid Search (`src/indexing/hybrid_search.py`)

Combines semantic and lexical search for better retrieval:

- **Vector Search**: ChromaDB with `all-MiniLM-L6-v2` embeddings
- **Keyword Search**: BM25 with custom tokenization
- **Fusion**: Reciprocal Rank Fusion (RRF) with k=60

### Streamlit App (`app.py`)

Features:
- Real-time streaming responses
- Conversation history
- Source citations with links
- Phoenix integration toggle
- Agent reinitialization

---

## Technology Stack

| Category | Technology |
|----------|------------|
| **UI** | Streamlit |
| **LLM** | OpenAI-compatible (DeepSeek, OpenAI, Local) |
| **Vector DB** | ChromaDB |
| **Keyword Search** | rank-bm25 |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Scraping** | crawl4ai, Playwright |
| **Tracing** | OpenTelemetry, Arize Phoenix |
| **Config** | pydantic-settings |
