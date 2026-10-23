"""ChromaDB Vector Store Module for Designs Law Assistant.

Manages persistent embeddings and similarity retrieval over the
Manual of Designs Practice & Procedure.
"""

import os
import shutil
from typing import List, Tuple, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions

from src.pdf_processor import load_and_parse_pdf, chunk_manual_pages

load_dotenv()


class LocalChromaDefaultEmbeddings(Embeddings):
    """LangChain-compatible wrapper around ChromaDB's fast, offline ONNX embedding function.
    
    Uses all-MiniLM-L6-v2 (384 dimensions) running locally with zero rate limits,
    zero network latency, and zero API quota consumption.
    """

    def __init__(self):
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]


def get_embedding_function(provider: Optional[str] = None) -> Embeddings:
    """Instantiate the requested or best available embedding model."""
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "chroma_default").lower()

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model_name = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=api_key)

    else:
        # Default to local ONNX embeddings (fast, robust, offline)
        return LocalChromaDefaultEmbeddings()


def get_or_create_vector_store(
    pdf_path: str = "data/manual_of_designs_practice_and_procedure.pdf",
    persist_directory: str = "./chroma_db",
    collection_name: str = "designs_law_manual",
    force_rebuild: bool = False,
    embedding_provider: Optional[str] = None
) -> Chroma:
    """Build or load a persistent ChromaDB vector store.
    
    If persistent database exists and force_rebuild is False, it loads instantly
    from disk without re-extracting or re-embedding the 87 pages.
    """
    embeddings = get_embedding_function(embedding_provider)

    if force_rebuild and os.path.exists(persist_directory):
        print(f"Force rebuild requested. Removing existing database at: {persist_directory}")
        shutil.rmtree(persist_directory, ignore_errors=True)

    # Check if existing persistent directory has data
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        try:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            count = vector_store._collection.count()
            if count > 0:
                print(f"Loaded existing ChromaDB collection '{collection_name}' with {count} chunks.")
                return vector_store
        except Exception as e:
            print(f"Could not load existing Chroma database ({e}). Rebuilding...")

    # Build fresh index from the PDF
    print(f"Parsing PDF document from: {pdf_path}...")
    pages_data = load_and_parse_pdf(pdf_path)
    print(f"Successfully extracted {len(pages_data)} pages. Generating chunks...")
    chunks = chunk_manual_pages(pages_data)
    print(f"Generated {len(chunks)} chunks with chapter/section metadata. Indexing into ChromaDB...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    print(f"ChromaDB index successfully built and persisted at: {persist_directory} ({len(chunks)} chunks).")
    return vector_store


def retrieve_relevant_chunks(
    vector_store: Chroma,
    query: str,
    k: int = 5
) -> List[Tuple[Document, float]]:
    """Retrieve top-k relevant chunks with similarity distance scores.
    
    Chroma uses L2 distance by default (lower distance = higher similarity).
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results
