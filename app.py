"""
Streamlit web UI for the LangChain RAG Agent.
Run: streamlit run app.py
"""

import tempfile
import os

import streamlit as st
from rag_agent import RAGAgent
from ingest import ingest
import config

st.set_page_config(
    page_title="LangChain RAG Agent",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 LangChain RAG Agent")
st.caption("Ask questions about your ingested documents.")


@st.cache_resource(show_spinner="Loading RAG agent...")
def get_agent(provider: str, model: str, ollama_base_url: str):
    return RAGAgent(provider=provider, model=model, ollama_base_url=ollama_base_url)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Model provider ────────────────────────────────────────────────────────
    st.subheader("Model Provider")
    provider = st.radio(
        "Choose LLM",
        options=["OpenAI GPT", "Llama (Ollama)"],
        index=0,
        horizontal=True,
    )

    if provider == "OpenAI GPT":
        model_name = st.text_input("Model name", value=config.OPENAI_MODEL)
        ollama_url = config.OLLAMA_BASE_URL
        selected_provider = "openai"
        st.caption("Requires OPENAI_API_KEY in your .env file.")
    else:
        model_name = st.text_input("Model name", value=config.OLLAMA_MODEL)
        ollama_url = st.text_input("Ollama server URL", value=config.OLLAMA_BASE_URL)
        selected_provider = "ollama"
        st.caption(
            f"Ollama must be running locally. "
            f"Embeddings use `{config.OLLAMA_EMBED_MODEL}` — make sure it is pulled."
        )

    st.markdown("---")

    # ── Conversation controls ─────────────────────────────────────────────────
    st.subheader("Controls")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        # reset memory on current agent if already loaded
        key = (selected_provider, model_name, ollama_url)
        if st.session_state.get("agent_key") == key:
            get_agent(*key).reset_memory()
        st.rerun()

    st.markdown("---")

    # ── Document ingestion ────────────────────────────────────────────────────
    st.subheader("Ingest Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    reset_on_ingest = st.checkbox("Reset vector store before ingesting", value=False)

    if st.button("Ingest uploaded files") and uploaded_files:
        with st.spinner("Ingesting..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for f in uploaded_files:
                    dest = os.path.join(tmp_dir, f.name)
                    with open(dest, "wb") as out:
                        out.write(f.read())
                ingest(tmp_dir, reset=reset_on_ingest)
        st.success(f"Ingested {len(uploaded_files)} file(s).")

    st.markdown("---")
    st.markdown("**CLI ingestion:**")
    st.code("python ingest.py <path_or_url> [--reset]", language="bash")


# ── Load agent based on current provider/model selection ─────────────────────
agent_key = (selected_provider, model_name, ollama_url)
if st.session_state.get("agent_key") != agent_key:
    st.session_state.messages = []
    st.session_state.agent_key = agent_key

agent = get_agent(selected_provider, model_name, ollama_url)

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    page_info = f" — page {src['page']}" if src.get("page") is not None else ""
                    st.markdown(f"**{src['source']}{page_info}**")
                    st.caption(src["snippet"])

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, sources = agent.chat_with_sources(prompt)
        st.write(response)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    page_info = f" — page {src['page']}" if src.get("page") is not None else ""
                    st.markdown(f"**{src['source']}{page_info}**")
                    st.caption(src["snippet"])

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
