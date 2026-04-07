"""
LangChain RAG Agent with tool-use and conversational memory.

The agent uses:
- ChromaDB as the vector store (retriever tool)
- OpenAI GPT or Llama (via Ollama) as the LLM
- LangChain AgentExecutor with tool-calling
- ConversationBufferWindowMemory for multi-turn context
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

import config


SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base.
Use the retriever tool to look up relevant information before answering questions.
If the answer is not found in the knowledge base, say so honestly rather than making something up.
Always cite the source document when possible.
"""


def build_agent(
    provider: str = "openai",
    model: str | None = None,
    ollama_base_url: str | None = None,
) -> AgentExecutor:
    """
    Build and return the RAG agent executor.

    Args:
        provider: "openai" or "ollama"
        model: model name override (defaults to config values)
        ollama_base_url: Ollama server URL (defaults to config.OLLAMA_BASE_URL)
    """
    if provider == "ollama":
        llm_model = model or config.OLLAMA_MODEL
        base_url = ollama_base_url or config.OLLAMA_BASE_URL
        embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=base_url,
        )
        llm = ChatOllama(
            model=llm_model,
            base_url=base_url,
            temperature=0,
        )
    else:
        llm_model = model or config.OPENAI_MODEL
        embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
        llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            openai_api_key=config.OPENAI_API_KEY,
        )

    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.RETRIEVER_K, "fetch_k": config.RETRIEVER_K * 2},
    )

    retriever_tool = create_retriever_tool(
        retriever,
        name="knowledge_base_search",
        description=(
            "Search the knowledge base for relevant information. "
            "Use this tool whenever you need to answer a question based on the ingested documents."
        ),
    )

    tools = [retriever_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )

    return agent_executor


def _extract_sources(intermediate_steps: list) -> list[dict]:
    """Extract unique source documents from agent intermediate steps."""
    seen = set()
    sources = []
    for _, observation in intermediate_steps:
        if not isinstance(observation, list):
            continue
        for doc in observation:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            key = meta.get("source", "") + str(meta.get("page", ""))
            if key and key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "source": meta.get("source", "Unknown"),
                        "page": meta.get("page"),
                        "snippet": doc.page_content[:200].strip(),
                    }
                )
    return sources


class RAGAgent:
    """Wrapper around AgentExecutor for easy use."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        ollama_base_url: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.executor = build_agent(provider, model, ollama_base_url)

    def chat(self, query: str) -> str:
        result = self.executor.invoke({"input": query})
        return result["output"]

    def chat_with_sources(self, query: str) -> tuple[str, list[dict]]:
        """Return (answer, sources) where sources is a list of dicts with source/page/snippet."""
        result = self.executor.invoke({"input": query})
        sources = _extract_sources(result.get("intermediate_steps", []))
        return result["output"], sources

    def reset_memory(self):
        self.executor.memory.clear()
        print("Conversation memory cleared.")
