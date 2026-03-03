import logging
import json
import uuid
import datetime
from backend.models.schemas import RetrieverResult, SourceChunk
from typing import List

# Configure structured JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
        }
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "extra_data"):
            log_record["extra_data"] = record.extra_data
            
        return json.dumps(log_record)

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

app_logger = get_logger("multi_doc_agent")

def deduplicate_chunks(chunks: List[RetrieverResult], similarity_threshold: float = 0.8) -> List[RetrieverResult]:
    """Drops near-duplicate chunks using Jaccard text similarity to preserve LLM context window."""
    unique_chunks = []
    
    for item in chunks:
        is_duplicate = False
        item_words = set(item.chunk.text.lower().split())
        
        for u in unique_chunks:
            u_words = set(u.chunk.text.lower().split())
            if not item_words or not u_words:
                continue
                
            intersection = len(item_words.intersection(u_words))
            union = len(item_words.union(u_words))
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard >= similarity_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_chunks.append(item)
            
    return unique_chunks

def generate_session_id() -> str:
    return str(uuid.uuid4())
