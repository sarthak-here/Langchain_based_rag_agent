# LangChain RAG Agent

A production-ready **Retrieval-Augmented Generation (RAG) agent** built with LangChain, ChromaDB, and OpenAI. It lets you ingest your own documents (PDFs, text files, web pages) and chat with them through a tool-calling agent that has conversational memory.

## Architecture

```
Documents (PDF / TXT / URL)
        │
        ▼
   [ ingest.py ]
   RecursiveCharacterTextSplitter
        │
        ▼
   OpenAI Embeddings
        │
        ▼
   ChromaDB (persisted locally)
        │
        ▼
   [ rag_agent.py ]
   Retriever Tool (MMR search)
        │
        ▼
   LangChain Tool-Calling Agent
   (GPT-4o-mini + memory)
        │
        ▼
   CLI / Streamlit UI
```

## Features

- **Document ingestion** — PDF, plain text, directories, and web URLs
- **Persistent vector store** — ChromaDB stored locally, survives restarts
- **MMR retrieval** — Maximal Marginal Relevance for diverse, relevant results
- **Tool-calling agent** — LLM decides when to query the knowledge base
- **Conversational memory** — remembers the last 10 turns of dialogue
- **Two interfaces** — interactive CLI and Streamlit web app

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/sarthak-here/langchain_based_rag_agent.git
cd langchain_based_rag_agent

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
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest documents

```bash
# Ingest a PDF
python ingest.py path/to/document.pdf

# Ingest a directory of PDFs and text files
python ingest.py path/to/docs/

# Ingest a web page
python ingest.py https://example.com/article
```

### 4. Chat

**CLI:**
```bash
python cli.py
```

**Streamlit web app:**
```bash
streamlit run app.py
```

## Project Structure

```
langchain_based_rag_agent/
├── app.py            # Streamlit web interface
├── cli.py            # Interactive terminal interface
├── ingest.py         # Document loading & vector store builder
├── rag_agent.py      # Core RAG agent (LangChain AgentExecutor)
├── config.py         # Centralised config loaded from .env
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model name |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local path for ChromaDB storage |
| `COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |

## How It Works

1. **Ingestion** (`ingest.py`) — Documents are loaded, split into overlapping chunks, embedded with OpenAI embeddings, and persisted in a local ChromaDB collection.

2. **Retrieval** (`rag_agent.py`) — At query time, the agent's retriever tool performs MMR search over the vector store to fetch the `k` most relevant and diverse chunks.

3. **Generation** — A `tool_calling` LangChain agent (backed by GPT-4o-mini) decides whether to invoke the retriever, synthesises the retrieved context, and streams the answer back to the user.

4. **Memory** — `ConversationBufferWindowMemory` keeps the last 10 exchanges so follow-up questions work naturally.

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core framework |
| `langchain-openai` | OpenAI LLM & embeddings |
| `langchain-chroma` | ChromaDB integration |
| `chromadb` | Local vector database |
| `sentence-transformers` | Optional local embeddings |
| `pypdf` | PDF loading |
| `streamlit` | Web UI |
| `python-dotenv` | Environment variable loading |

## License

MIT
