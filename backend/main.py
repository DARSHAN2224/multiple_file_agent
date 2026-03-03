from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import documents, query
from backend.utils import app_logger
import asyncio

app = FastAPI(title="Multi-Document Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(query.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

# Note: In a true production app, we would use Celery/APScheduler for session cleanup.
# Here we will use an asyncio background task bounded to the app lifecycle to purge old sessions.
class SessionCleanupTask:
    def __init__(self):
        self.task = None

    async def run(self):
        while True:
            await asyncio.sleep(600)  # Run every 10 minutes
            app_logger.info("Running background session cleanup...")
            try:
                # We import routers here lazily to avoid circular imports if any
                from backend.routers.documents import session_chunks, session_timestamps, vector_store_manager
                import time
                
                current_time = time.time()
                expired = []
                for sid, timestamp in list(session_timestamps.items()):
                    if current_time - timestamp > 3600 * 2: # 2 hours TTL
                        expired.append(sid)
                        
                for sid in expired:
                    if sid in session_timestamps: del session_timestamps[sid]
                    if sid in session_chunks: del session_chunks[sid]
                    vector_store_manager.cleanup_session(sid)
                    app_logger.info("Cleaned up expired session", extra={"extra_data": {"session_id": sid}})
            except Exception as e:
                app_logger.error(f"Error during cleanup: {e}")

cleanup_worker = SessionCleanupTask()

@app.on_event("startup")
async def startup_event():
    app_logger.info("Starting up FastAPI Backend")
    cleanup_worker.task = asyncio.create_task(cleanup_worker.run())

@app.on_event("shutdown")
async def shutdown_event():
    app_logger.info("Shutting down FastAPI Backend")
    if cleanup_worker.task:
        cleanup_worker.task.cancel()
