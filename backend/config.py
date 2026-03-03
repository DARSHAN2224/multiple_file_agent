import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Ollama Settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # System Thresholds & Guardrails
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
    distance_threshold: float = float(os.getenv("DISTANCE_THRESHOLD", "0.3"))
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
    
    # Performance & Storage
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
    
    # Paths
    upload_dir: str = "temp_uploads"
    faiss_index_dir: str = "faiss_indices"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Global singleton configuration object
settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.faiss_index_dir, exist_ok=True)
