"""
Streamlit web UI for the LangChain RAG Agent.
Run: streamlit run app.py
"""

import streamlit as st
from rag_agent import RAGAgent

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

with st.sidebar:
    st.header("Controls")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        agent.reset_memory()
        st.rerun()

    st.markdown("---")
    st.markdown("**How to ingest documents:**")
    st.code("python ingest.py <path_or_url>", language="bash")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(prompt)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
