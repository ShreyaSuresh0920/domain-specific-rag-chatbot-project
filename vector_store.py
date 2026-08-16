"""Chunk documents, create embeddings, and manage the FAISS vector store."""

import os
from functools import lru_cache
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
INDEX_DIR = str(Path(__file__).resolve().parent / "vector_store" / "saved_index")


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def chunk_documents(documents):
    """Split extracted page texts into overlapping chunks, keeping metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for piece in pieces:
            chunks.append(Document(
                page_content=piece,
                metadata={"source": doc["source"], "page": doc["page"]},
            ))
    return chunks


def build_vector_store(chunks):
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    return store


def save_vector_store(store, path=INDEX_DIR):
    os.makedirs(path, exist_ok=True)
    store.save_local(path)


def index_exists(path=INDEX_DIR):
    return os.path.exists(os.path.join(path, "index.faiss"))


def load_vector_store(path=INDEX_DIR):
    """Load a previously saved index. Returns None if none exists or it fails to load."""
    if not index_exists(path):
        return None
    try:
        embeddings = get_embeddings()
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None
