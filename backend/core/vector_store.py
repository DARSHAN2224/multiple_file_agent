import os
import shutil
from typing import Dict, List, Optional
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from backend.models.schemas import SourceChunk


class SessionVectorStore:
    """Manages isolated FAISS indexes and BM25 Corpus per session, with disk persistence."""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.active_sessions: Dict[str, FAISS] = {}
        self.active_bm25: Dict[str, BM25Okapi] = {}

    def _session_index_path(self, session_id: str) -> str:
        return os.path.join(self.index_dir, session_id)

    def get_bm25(self, session_id: str) -> Optional[BM25Okapi]:
        return self.active_bm25.get(session_id)

    def get_store(self, session_id: str, embeddings_model) -> FAISS:
        """Retrieves or loads-from-disk a FAISS store for the session.
        
        Priority:
          1. Already in memory  -> return it directly.
          2. Exists on disk     -> load it into memory and return.
          3. Neither            -> create a fresh empty store.
        """
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        disk_path = self._session_index_path(session_id)
        if os.path.exists(disk_path):
            try:
                store = FAISS.load_local(
                    disk_path,
                    embeddings_model,
                    allow_dangerous_deserialization=True,
                )
                self.active_sessions[session_id] = store
                return store
            except Exception as e:
                # If the saved index is corrupt, fall through and create fresh
                import logging
                logging.getLogger(__name__).error(
                    f"Failed to load FAISS index from disk for session {session_id}: {e}"
                )

        # Create a fresh empty store
        dummy_embedding = embeddings_model.embed_query("test")
        dim = len(dummy_embedding)
        index = faiss.IndexFlatL2(dim)
        store = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.active_sessions[session_id] = store
        return store

    def add_chunks(self, session_id: str, chunks: List[SourceChunk], embeddings_model):
        """Adds text chunks to the session FAISS store and persists to disk."""
        store = self.get_store(session_id, embeddings_model)
        if not chunks:
            return

        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "token_count": chunk.token_count,
                },
            )
            for chunk in chunks
        ]

        store.add_documents(documents)

        # Persist the updated FAISS index to disk immediately
        disk_path = self._session_index_path(session_id)
        os.makedirs(disk_path, exist_ok=True)
        store.save_local(disk_path)

    def rebuild_bm25(self, session_id: str, all_chunks: List[SourceChunk]):
        """Rebuilds BM25 over the FULL corpus passed in."""
        corpus = [chunk.text.lower().split() for chunk in all_chunks]
        self.active_bm25[session_id] = BM25Okapi(corpus) if corpus else None

    def cleanup_session(self, session_id: str):
        """Removes session FAISS store from memory and from disk."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.active_bm25:
            del self.active_bm25[session_id]

        disk_path = self._session_index_path(session_id)
        if os.path.exists(disk_path):
            shutil.rmtree(disk_path)
