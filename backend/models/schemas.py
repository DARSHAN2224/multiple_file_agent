from pydantic import BaseModel, Field
from typing import List, Optional

class SourceChunk(BaseModel):
    source_file: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    text: str
    token_count: int

class RetrieverResult(BaseModel):
    chunk: SourceChunk
    semantic_score: float
    sparse_score: float
    final_score: float

class QueryRequest(BaseModel):
    session_id: str
    query: str
    filter_files: Optional[List[str]] = None  # If set, only search within these files

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    confidence: float
    sources: List[SourceChunk]

class ComparisonCluster(BaseModel):
    topic: str
    chunks: List[SourceChunk]
    summary: str

class ComparisonResponse(BaseModel):
    session_id: str
    clusters: List[ComparisonCluster]
    differences: str
    commonalities: str
