# 🔍 LangChain RAG Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0.3-green?style=for-the-badge&logo=chainlink&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.41-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-0.6-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Ollama-Llama%203.2-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  A production-ready <strong>Retrieval-Augmented Generation (RAG) agent</strong> built with LangChain and ChromaDB.<br/>
  Ingest your own documents and chat with them — powered by <strong>OpenAI GPT</strong> or a fully <strong>local Llama model via Ollama</strong>.
</p>

---

## ✨ Features

| | Feature | Details |
|---|---|---|
| 📄 | **Multi-format ingestion** | PDF, TXT, Markdown, directories, and web URLs |
| 🧠 | **Two LLM providers** | Switch between OpenAI GPT and local Llama (Ollama) from the UI |
| 🔎 | **MMR retrieval** | Maximal Marginal Relevance for diverse, relevant results |
| 🤖 | **Tool-calling agent** | LLM decides when and how to query the knowledge base |
| 💬 | **Conversational memory** | Remembers the last 10 turns of dialogue |
| 📚 | **Source citations** | Every answer shows which documents it came from |
| 🖥️ | **Two interfaces** | Streamlit web app + interactive CLI |
| 🔄 | **In-app ingestion** | Upload files directly from the Streamlit sidebar |

---

## 🖼️ Screenshots

### Streamlit Web App
![Streamlit UI](screenshots/streamlit_ui.png)

### Model Selector — OpenAI vs Llama
![Model Selector](screenshots/model_selector.png)

### Source Citations
![Source Citations](screenshots/sources.png)

> 📸 _Run `streamlit run app.py` to see it live._

---

## 🏗️ Architecture

```
  Documents (PDF / TXT / MD / URL)
            │
            ▼
       [ ingest.py ]
  RecursiveCharacterTextSplitter
            │
            ▼
  ┌─────────────────────────┐
  │      Embeddings         │
  │  OpenAI  │  Ollama      │  ← chosen at runtime
  │  (cloud) │  (local)     │
  └─────────────────────────┘
            │
            ▼
     ChromaDB (local)
            │
            ▼
     MMR Retriever Tool
            │
            ▼
  ┌─────────────────────────┐
  │         LLM             │
  │  GPT-4o-mini │ Llama3.2 │  ← chosen from UI
  └─────────────────────────┘
            │
            ▼
   LangChain Tool-Calling Agent
   + ConversationBufferWindowMemory
            │
       ┌────┴────┐
       ▼         ▼
  Streamlit    CLI
    (app.py)  (cli.py)
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/sarthak-here/Langchain_based_rag_agent.git
cd Langchain_based_rag_agent

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your keys (see [Configuration](#-configuration)).

### 3. Ingest documents

```bash
# Single PDF
python ingest.py path/to/document.pdf

# Entire folder (picks up .pdf, .txt, .md)
python ingest.py path/to/docs/

# Web page
python ingest.py https://example.com/article

# Reset the vector store and re-ingest from scratch
python ingest.py path/to/docs/ --reset
```

### 4. Chat

**Streamlit web app** _(recommended)_:
```bash
streamlit run app.py
```

**Interactive CLI**:
```bash
python cli.py
```

---

## 🦙 Using Llama (Local / Free)

No OpenAI key? No problem. Run everything locally with [Ollama](https://ollama.com).

**1. Install Ollama** → [ollama.com/download](https://ollama.com/download)

**2. Pull the required models:**
```bash
ollama pull llama3.2          # the chat model
ollama pull nomic-embed-text  # the embedding model
```

**3. Start the app and select _Llama (Ollama)_ in the sidebar:**
```bash
streamlit run app.py
```

That's it — no API key, no cloud calls, fully private.

---

## 🖥️ CLI Commands

```
Commands:
  reset        — Clear conversation memory
  sources on   — Show source documents with each answer
  sources off  — Hide source documents
  help         — Show this message
  exit / quit  — Exit
```

---

## 📁 Project Structure

```
langchain_based_rag_agent/
├── app.py            # Streamlit web UI (model selector, file uploader, source citations)
├── cli.py            # Interactive terminal interface
├── ingest.py         # Document loading, chunking & vector store builder
├── rag_agent.py      # Core RAG agent (supports OpenAI + Ollama)
├── config.py         # Centralised config loaded from .env
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and set the variables you need:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key _(required for GPT)_ |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local path for ChromaDB storage |
| `COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core orchestration framework |
| `langchain-openai` | OpenAI LLM & embeddings |
| `langchain-ollama` | Ollama (local Llama) LLM & embeddings |
| `langchain-chroma` | ChromaDB vector store integration |
| `chromadb` | Local persistent vector database |
| `pypdf` | PDF document loading |
| `streamlit` | Web UI |
| `python-dotenv` | `.env` file loading |

---

## 📄 License

MIT © [sarthak-here](https://github.com/sarthak-here)



