"""
LangChain RAG Agent with tool-use and conversational memory.

The agent uses:
- ChromaDB as the vector store (retriever tool)
- OpenAI GPT as the LLM
- LangChain AgentExecutor with tool-calling
- ConversationBufferWindowMemory for multi-turn context
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage

import config


SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base.
Use the retriever tool to look up relevant information before answering questions.
If the answer is not found in the knowledge base, say so honestly rather than making something up.
Always cite the source document when possible.
"""


def build_agent() -> AgentExecutor:
    """Build and return the RAG agent executor."""
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)

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

    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )

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
    )

    return agent_executor


class RAGAgent:
    """Wrapper around AgentExecutor for easy use."""

    def __init__(self):
        self.executor = build_agent()

    def chat(self, query: str) -> str:
        result = self.executor.invoke({"input": query})
        return result["output"]

    def reset_memory(self):
        self.executor.memory.clear()
        print("Conversation memory cleared.")
