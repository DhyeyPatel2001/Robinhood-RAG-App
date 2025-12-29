# 🏹 Robinhood RAG - Support Documentation Assistant

An **Agentic RAG** (Retrieval-Augmented Generation) system for Robinhood support documentation with hybrid search, conversation history, and full observability powered by Arize Phoenix.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)

## ✨ Features

- 🤖 **Agentic RAG** - ReAct-style reasoning with query rewriting
- 🔍 **Hybrid Search** - ChromaDB (vector) + BM25 (keyword) with RRF fusion
- 💬 **Streaming Chat** - Real-time responses with thinking process
- 📊 **Full Observability** - Hierarchical tracing with Arize Phoenix
- 🔗 **Source Citations** - Every answer includes source links
- 🔧 **Flexible LLM** - Works with DeepSeek, OpenAI, or any compatible API

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** installed
- **Git** installed
- An **LLM API key** (DeepSeek, OpenAI, or compatible)

---

## 🚀 Quick Start

### Step 1: Clone & Navigate

```bash
cd /path/to/your/projects
git clone <repository-url>
cd "Robinhood RAG"
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate    # macOS/Linux
# OR
.\venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Install Playwright browsers (for web scraping)
playwright install
```

### Step 4: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
nano .env   # or use any text editor
```

**Required `.env` configuration:**
```env
# LLM Provider (DeepSeek is the default)
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek-chat

# Optional: Adjust these based on your needs
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SEARCH_TOP_K=5
MAX_AGENT_ITERATIONS=5
```

### Step 5: Build the Index (First Time Only)

If you have raw data already scraped, just run:

```bash
python build_index.py
```

**OR** to scrape fresh data from Robinhood:

```bash
# Scrape documentation (may take 10-30 minutes)
python -m src.scraper.crawler --max-pages 100

# Process and chunk the data
python -m src.processing.chunker
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

🎉 **Open http://localhost:8501 in your browser!**

---

## 📊 Viewing Telemetry (Arize Phoenix)

The app includes built-in observability via Arize Phoenix.

### Access Phoenix Dashboard

When you run the app, Phoenix automatically starts. Access it at:

**http://localhost:6006**

### What You'll See

1. **Projects** → Click on `robinhood-rag`
2. **Traces Tab** → View all requests
3. **Click any trace** → See hierarchical spans:

```
rag_request (Parent)
├── query_rewrite
├── hybrid_search
├── ChatCompletion (LLM call)
├── hybrid_search
└── ChatCompletion (final answer)
```

### Toggle Telemetry

In the Streamlit sidebar, you can enable/disable Phoenix telemetry.

---

## 🏗️ Architecture

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

### High-Level Flow

```
User Question
     ↓
Query Rewriter (context → standalone question)
     ↓
Agent Loop (ReAct-style reasoning)
     ↓
Hybrid Search (ChromaDB + BM25 → RRF fusion)
     ↓
LLM Generation (with citations)
     ↓
Streaming Response
```

---

## 📁 Project Structure

```
Robinhood RAG/
├── app.py                # Streamlit chat interface
├── config.py             # Configuration (pydantic-settings)
├── build_index.py        # Index building script
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── ARCHITECTURE.md       # Detailed architecture docs
│
├── src/
│   ├── scraper/          # Web scraping (crawl4ai)
│   ├── processing/       # Text chunking (LangChain)
│   ├── indexing/         # Search indexes (ChromaDB + BM25)
│   ├── llm/              # LLM client + RAG agent
│   └── telemetry/        # Phoenix tracing setup
│
├── data/
│   ├── raw/              # Scraped markdown files
│   └── processed/        # chunks.json
│
└── indexes/
    ├── chroma/           # ChromaDB persistent storage
    └── bm25/             # BM25 index files
```

---

## ⚙️ Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | API base URL | `https://api.deepseek.com` |
| `LLM_API_KEY` | API key for LLM service | **(required)** |
| `LLM_MODEL` | Model name | `deepseek-chat` |
| `CHUNK_SIZE` | Text chunk size in characters | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `SEARCH_TOP_K` | Number of search results to retrieve | `5` |
| `MAX_AGENT_ITERATIONS` | Max agent reasoning iterations | `5` |

---

## 🔧 Supported LLM Providers

The system works with any **OpenAI-compatible** API:

| Provider | Base URL | Notes |
|----------|----------|-------|
| **DeepSeek** | `https://api.deepseek.com` | Default, cost-effective |
| **OpenAI** | `https://api.openai.com/v1` | GPT-4, GPT-3.5 |
| **Azure OpenAI** | `https://your-resource.openai.azure.com` | Enterprise |
| **Ollama** | `http://localhost:11434/v1` | Local models |
| **Groq** | `https://api.groq.com/openai/v1` | Fast inference |

---

## 🧪 Testing Components

Test individual components:

```bash
# Test the RAG agent
python -m src.llm.agent

# Test hybrid search
python -m src.indexing.hybrid_search

# Test vector store
python -m src.indexing.vector_store

# Test keyword store  
python -m src.indexing.keyword_store

# Test Phoenix tracing
python test_phoenix.py
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. "API key not set"**
```bash
# Make sure .env exists and has your key
cat .env | grep LLM_API_KEY
```

**2. "No documents in index"**
```bash
# Rebuild the index
python build_index.py
```

**3. "Phoenix not showing traces"**
- Check that telemetry is enabled in the sidebar
- Refresh Phoenix at http://localhost:6006
- Look for the `robinhood-rag` project

**4. Port already in use**
```bash
# Kill existing processes
lsof -ti:8501 | xargs kill -9   # Streamlit
lsof -ti:6006 | xargs kill -9   # Phoenix
```

---

## 📚 Additional Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture
- [.env.example](.env.example) - Environment variable template

---

## 📝 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Arize Phoenix** - Observability platform
- **ChromaDB** - Vector database
- **LangChain** - Text processing utilities
- **Streamlit** - UI framework
- **crawl4ai** - Web scraping
