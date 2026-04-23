# LangChain RAG Agent - System Design

## What It Does
A production-ready RAG agent that lets you chat with your own documents. Ingest any
folder of files (PDF, TXT, MD, DOCX) into ChromaDB, then query in natural language via
Streamlit UI or CLI -- using OpenAI GPT or a local Llama model via Ollama.

---

## Architecture

```
INGEST PATH (one-time setup)
==============================
  docs/ folder (PDF, TXT, MD, DOCX)
        |
        v
  ingest.py
  - DocumentLoader (per file type)
  - RecursiveCharacterTextSplitter
    (chunk_size=1000, overlap=200)
  - Embed: OpenAIEmbeddings OR OllamaEmbeddings
  - Store: ChromaDB (./chroma_store, persistent)


QUERY PATH (every chat turn)
==============================
  User question
        |
        v
  rag_agent.py
  - Embed question (same model as ingest)
  - ChromaDB.similarity_search(k=5)
  - LangChain prompt:
      [system] + [retrieved chunks] + [history] + [question]
  - LLM: OpenAI ChatGPT or Ollama (one config line)
  - Stream answer + source citations
        |
        v
  app.py (Streamlit UI)  OR  cli.py (terminal)
```

---

## Input

| Input          | Detail                                              |
|----------------|-----------------------------------------------------|
| Documents      | Any folder: PDF, TXT, MD, DOCX                      |
| config.py      | LLM provider, model name, chunk size, top-K         |
| User query     | Natural language via CLI or Streamlit               |
| .env file      | OPENAI_API_KEY, OLLAMA_BASE_URL                     |

---

## Data Flow

```
INGEST:
  ingest.py --source ./docs
  -> DocumentLoader -> RecursiveCharacterTextSplitter
  -> Embedding model -> ChromaDB.add_documents()

QUERY:
  "What does the refund policy say?"
        |
  Embed question -> query vector
        |
  ChromaDB.similarity_search(query, k=5)
  -> Top 5 relevant chunks
        |
  LangChain RetrievalQA chain:
    Prompt = "Use ONLY the context to answer.
              Context: [chunks]
              Question: [query]"
        |
  LLM -> streamed answer + source chunk citations
```

---

## Key Design Decisions

| Decision                           | Reason                                         |
|------------------------------------|------------------------------------------------|
| ChromaDB (local, persistent)       | No external vector DB infra needed             |
| RecursiveCharacterTextSplitter     | Respects paragraph/sentence boundaries         |
| 200-token chunk overlap            | Prevents answers being cut across chunk edges  |
| LangChain abstraction              | Swap OpenAI for Ollama with one config change  |
| Source citations in response       | Users verify which document chunk drove answer |

---

## Interview Conclusion

This implements the canonical RAG architecture: offline ingest into a vector store,
followed by retrieval-augmented generation at query time. The two-phase design is
intentional -- ingest once, query many times -- keeping query latency fast regardless
of document count. Chunk overlap (200 tokens) is a critical parameter: too small and
answers split across chunks; too large and irrelevant context floods the prompt. The
LangChain abstraction makes it provider-agnostic: same code works with GPT-4o for
quality or local Llama for privacy. Scaling: add cross-encoder re-ranking, HyDE query
expansion, and a feedback loop to improve retrieval on user ratings.
