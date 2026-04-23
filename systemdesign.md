# LangChain RAG Agent — System Design

## What It Does
A production-ready Retrieval-Augmented Generation (RAG) agent that lets you chat with your own documents. Ingest any folder of files, chunk and embed them into ChromaDB, then query in natural language via CLI or a Streamlit web app — using OpenAI GPT or a local Llama model via Ollama.

---

## Architecture

```
           INGEST PATH (one-time)
           ========================
  docs/ folder
       |
       v
  ingest.py
  - Load files (PDF, TXT, MD, DOCX)
  - Split: RecursiveCharacterTextSplitter
    (chunk_size=1000, overlap=200)
  - Embed: OpenAIEmbeddings OR OllamaEmbeddings
  - Store: ChromaDB (./chroma_store, persistent)


           QUERY PATH (every chat turn)
           ==============================
  User question
       |
       v
  rag_agent.py
  - Embed question with same embedding model
  - ChromaDB similarity search -> top-K chunks
  - Build LangChain prompt:
      [system] + [retrieved context] + [chat history] + [question]
  - Call LLM (OpenAI ChatGPT or local Ollama)
  - Stream answer tokens
       |
       v
  app.py (Streamlit UI)  OR  cli.py (terminal)
```

---

## Input

| Input | Detail |
|---|---|
| Documents | Any folder: PDF, TXT, MD, DOCX |
| Config | config.py: LLM provider, model, chunk size, top-K |
| User query | Natural language question via CLI or Streamlit |
| .env file | OPENAI_API_KEY, OLLAMA_BASE_URL |

---

## Data Flow

```
INGEST:
  ingest.py --source ./docs
        |
  DocumentLoader (PyPDF, TextLoader, etc.)
        |
  RecursiveCharacterTextSplitter
  chunk_size=1000, overlap=200
        |
  Embedding model
  (OpenAIEmbeddings or nomic-embed-text via Ollama)
        |
  ChromaDB.add_documents()
  Persisted to ./chroma_store/

QUERY:
  "What does section 3 say about refunds?"
        |
  Same embedding model -> query vector
        |
  ChromaDB.similarity_search(query, k=5)
        |
  Top 5 relevant chunks
        |
  LangChain RetrievalQA chain:
    Prompt = "Use ONLY the context below to answer.
              Context: [chunks]
              Question: [query]"
        |
  LLM (gpt-4o / llama3) -> streamed answer
        |
  Displayed in Streamlit chat UI with source citations
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| ChromaDB (local) | No external vector DB infra needed; persistent across sessions |
| RecursiveCharacterTextSplitter | Respects paragraph/sentence boundaries better than fixed splits |
| 200-token chunk overlap | Prevents answer context from being cut at chunk boundaries |
| LangChain chain abstraction | Swapping OpenAI for Ollama is one config line in config.py |
| Source citation display | Users can verify which document chunk generated the answer |

---

## Interview Conclusion

This project implements the canonical RAG architecture: offline document ingestion into a vector store, followed by retrieval-augmented generation at query time. The two-phase design is intentional — ingest once, query many times — which keeps the query path fast regardless of document count. The chunk overlap (200 tokens) is a critical parameter: too small and answers get split across chunks; too large and irrelevant context floods the LLM prompt. The LangChain abstraction layer makes the system provider-agnostic: the same codebase works with GPT-4o for maximum quality or with a local Llama model for full data privacy. If I were scaling this to production, I would add a re-ranking step (cross-encoder) after the initial retrieval, implement query expansion via HyDE (Hypothetical Document Embeddings), and add a feedback loop to improve retrieval quality based on user ratings.
