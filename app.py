"""
Streamlit web UI for the LangChain RAG Agent.
Run: streamlit run app.py
"""

import tempfile
import os

import streamlit as st
from rag_agent import RAGAgent
from ingest import ingest

st.set_page_config(
    page_title="LangChain RAG Agent",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 LangChain RAG Agent")
st.caption("Ask questions about your ingested documents.")


@st.cache_resource(show_spinner="Loading RAG agent...")
def get_agent():
    return RAGAgent()


agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        agent.reset_memory()
        st.rerun()

    st.markdown("---")
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

# ── Chat history ─────────────────────────────────────────────────────────────
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
