"""
Document ingestion pipeline: loads documents, splits them into chunks,
embeds them, and stores in ChromaDB vector store.
"""

import os
import argparse
import shutil
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config


def load_documents(source: str) -> list:
    """Load documents from a file, directory, or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"Loading from URL: {source}")
        loader = WebBaseLoader(source)
        return loader.load()

    path = Path(source)

    if path.is_dir():
        print(f"Loading all .txt, .pdf, and .md files from directory: {source}")
        loaders = [
            DirectoryLoader(source, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(source, glob="**/*.txt", loader_cls=TextLoader),
            DirectoryLoader(source, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        ]
        docs = []
        for loader in loaders:
            try:
                loaded = loader.load()
                docs.extend(loaded)
            except Exception as e:
                print(f"Warning: {e}")
        return docs

    if path.suffix == ".pdf":
        print(f"Loading PDF: {source}")
        return PyPDFLoader(source).load()

    if path.suffix == ".md":
        print(f"Loading Markdown: {source}")
        return UnstructuredMarkdownLoader(source).load()

    print(f"Loading text file: {source}")
    return TextLoader(source).load()


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} document(s) into {len(chunks)} chunks.")
    return chunks


def reset_vector_store():
    """Delete the existing ChromaDB persist directory."""
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
        print(f"Cleared existing vector store at: {config.CHROMA_PERSIST_DIR}")


def build_vector_store(chunks: list) -> Chroma:
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    print(f"Vector store saved to: {config.CHROMA_PERSIST_DIR}")
    return vector_store


def ingest(source: str, reset: bool = False):
    if reset:
        reset_vector_store()
    docs = load_documents(source)
    if not docs:
        print("No documents found. Exiting.")
        return
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "source",
        help="Path to a file, directory, or a URL to ingest.",
    )
    parser.add_argument(
        "--reset", "-r",
        action="store_true",
        help="Clear the existing vector store before ingesting.",
    )
    args = parser.parse_args()
    ingest(args.source, reset=args.reset)
