import os
import pickle
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from backend.models.schemas import SourceChunk
from backend.config import settings
from backend.core.document_processor import DocumentProcessor
from backend.utils import app_logger
from langchain_community.embeddings import OllamaEmbeddings
from backend.core.vector_store import SessionVectorStore

router = APIRouter(prefix="/api/documents", tags=["documents"])

SESSIONS_DIR = "session_data"
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Initialize Models
try:
    embeddings = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
except Exception as e:
    app_logger.error(f"Failed to initialize embeddings: {e}")
    embeddings = None

# Global vector store manager used across routers
vector_store_manager = SessionVectorStore(settings.faiss_index_dir)

# In-memory cache (populated lazily from disk on first access)
session_chunks = {}
session_timestamps = {}
session_files = {}  # tracks file paths on disk per session


def _session_file(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.pkl")


def _load_session(session_id: str):
    """Load session chunks from disk into memory if not already loaded."""
    if session_id not in session_chunks:
        path = _session_file(session_id)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                session_chunks[session_id] = data.get("chunks", [])
                session_timestamps[session_id] = data.get("timestamp", time.time())
                session_files[session_id] = data.get("files", [])
            except Exception as e:
                app_logger.error(f"Failed to load session from disk: {e}")
                session_chunks[session_id] = []
                session_files[session_id] = []


def _save_session(session_id: str):
    """Persist session chunks and file paths to disk."""
    try:
        with open(_session_file(session_id), "wb") as f:
            pickle.dump({
                "chunks": session_chunks[session_id],
                "timestamp": session_timestamps.get(session_id, time.time()),
                "files": session_files.get(session_id, []),
            }, f)
    except Exception as e:
        app_logger.error(f"Failed to persist session to disk: {e}")


@router.post("/upload")
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    app_logger.info("Received documents upload request", extra={"extra_data": {"session_id": session_id, "num_files": len(files)}})

    _load_session(session_id)
    if session_id not in session_chunks:
        session_chunks[session_id] = []
    if session_id not in session_files:
        session_files[session_id] = []

    session_timestamps[session_id] = time.time()

    new_chunks = []
    for file in files:
        if file.size > settings.max_file_size_mb * 1024 * 1024:
            app_logger.warning("File size limit exceeded", extra={"extra_data": {"filename": file.filename}})
            raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds {settings.max_file_size_mb} MB limit")

        file_path = os.path.join(settings.upload_dir, f"{session_id}_{file.filename}")

        # Skip files already indexed in this session to prevent duplicate chunks
        if file_path in session_files[session_id]:
            app_logger.info("Skipping already-indexed file", extra={"extra_data": {"filename": file.filename}})
            continue

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        try:
            chunks = DocumentProcessor.process_file(file_path, file.filename)
            session_chunks[session_id].extend(chunks)
            new_chunks.extend(chunks)
            
            # Only track the path as 'successful' after processing finishes
            session_files[session_id].append(file_path)
            
            app_logger.info("Successfully processed document", extra={"extra_data": {"filename": file.filename, "num_chunks": len(chunks)}})
        except Exception as e:
            app_logger.error("Failed to process document", extra={"extra_data": {"filename": file.filename, "error": str(e)}})
            # Remove failed partial file if exists so it's not orphaned
            if os.path.exists(file_path):
                try: os.remove(file_path)
                except: pass
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {str(e)}")

    # Immediately index all NEW chunks in FAISS and refresh BM25 once for the entire batch
    if new_chunks:
        try:
            if embeddings is None:
                raise ValueError("Embeddings model not initialized. Check Ollama connection.")
            vector_store_manager.add_chunks(session_id, new_chunks, embeddings)
            vector_store_manager.rebuild_bm25(session_id, session_chunks[session_id])
            app_logger.info("Batch indexing complete", extra={"extra_data": {"num_chunks": len(new_chunks)}})
        except Exception as e:
            app_logger.error(f"Failed to index batch: {e}")
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    # Persist to disk so restarts don't lose the data
    _save_session(session_id)

    return {"message": "Files uploaded successfully", "total_chunks": len(session_chunks[session_id])}


@router.get("/list/{session_id}")
async def list_documents(session_id: str):
    _load_session(session_id)
    if session_id not in session_chunks or not session_chunks[session_id]:
        return {"documents": []}

    docs = list(set([c.source_file for c in session_chunks[session_id]]))
    return {"documents": docs}


def get_session_chunks(session_id: str) -> List[SourceChunk]:
    _load_session(session_id)
    return session_chunks.get(session_id, [])


@router.get("/sessions")
async def list_sessions():
    """Return all saved sessions from disk so the frontend can offer session restore."""
    saved = []
    try:
        for fname in os.listdir(SESSIONS_DIR):
            if not fname.endswith(".pkl"):
                continue
            sid = fname[:-4]  # strip .pkl
            path = _session_file(sid)
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                chunks = data.get("chunks", [])
                docs = list(set(c.source_file for c in chunks))
                saved.append({
                    "session_id": sid,
                    "documents": docs,
                    "chunk_count": len(chunks),
                    "timestamp": data.get("timestamp", 0),
                })
            except Exception:
                continue  # skip corrupt files
    except Exception as e:
        app_logger.error(f"Failed to list sessions: {e}")
    # Sort newest first
    saved.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"sessions": saved}


def cleanup_session_data(session_id: str):
    # Remove persisted temp upload files
    for file_path in session_files.get(session_id, []):
        if os.path.exists(file_path):
            os.remove(file_path)

    if session_id in session_chunks:
        del session_chunks[session_id]
    if session_id in session_timestamps:
        del session_timestamps[session_id]
    if session_id in session_files:
        del session_files[session_id]
    path = _session_file(session_id)
    if os.path.exists(path):
        os.remove(path)
