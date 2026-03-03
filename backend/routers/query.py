import time
import re
import asyncio
from fastapi import APIRouter, HTTPException
from backend.models.schemas import QueryRequest, QueryResponse, ComparisonResponse
from backend.core.cache import query_cache
from backend.routers.documents import get_session_chunks, vector_store_manager
from backend.core.retriever import HybridRetriever
from backend.core.synthesis import SynthesisEngine
from backend.core.cluster import SemanticClusterer
from backend.utils import app_logger
from backend.config import settings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
router = APIRouter(prefix="/api", tags=["query"])

# Initialize models
try:
    llm = Ollama(model=settings.llm_model, base_url=settings.ollama_base_url)
    embeddings = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
except Exception as e:
    app_logger.error(f"Failed to initialize Ollama models: {e}")
    llm = None
    embeddings = None

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    start_time = time.time()
    app_logger.info("Received query request", extra={"extra_data": {"session_id": request.session_id, "query": request.query}})
    
    # Check cache (include filter_files in key so different filters don't collide)
    filter_key = "|".join(sorted(request.filter_files)) if request.filter_files else "all"
    cache_key = f"{request.session_id}_{hash(request.query)}_{filter_key}"
    cached_response = query_cache.get(cache_key)
    if cached_response:
        app_logger.info("Cache hit for query")
        return cached_response

    all_chunks = get_session_chunks(request.session_id)
    if not all_chunks:
        raise HTTPException(status_code=400, detail="No documents uploaded for this session.")

    # Apply file filter for retrieval scope only; keep all_chunks for BM25 building
    chunks = all_chunks
    if request.filter_files:
        chunks = [c for c in all_chunks if c.source_file in request.filter_files]
        if not chunks:
            raise HTTPException(status_code=400, detail=f"No chunks found for the selected file(s): {request.filter_files}")

    # Ensure FAISS store is loaded and BM25 is built
    store = vector_store_manager.get_store(request.session_id, embeddings)
    if vector_store_manager.get_bm25(request.session_id) is None:
        app_logger.info("Rebuilding BM25 for session", extra={"extra_data": {"session_id": request.session_id}})
        vector_store_manager.rebuild_bm25(request.session_id, all_chunks)

    # Retrieve — always pass the full session corpus so BM25 indices align.
    # File-scope filtering is applied AFTER retrieval so semantic search still
    # considers the full index before narrowing results to the requested files.
    bm25_model = vector_store_manager.get_bm25(request.session_id)
    retriever = HybridRetriever(store, bm25_model, all_chunks)
    results = await asyncio.to_thread(retriever.retrieve, request.query, top_k=10)

    # Apply file filter to the ranked results (not to the retrieval corpus)
    if request.filter_files:
        filter_set = set(request.filter_files)
        results = [r for r in results if r.chunk.source_file in filter_set]
        if not results:
            raise HTTPException(status_code=400, detail=f"No matching chunks found for the selected file(s): {request.filter_files}")

    # Synthesize
    synthesis_engine = SynthesisEngine(llm)
    answer, context_chunks, confidence = synthesis_engine.generate_answer(request.query, results)
    
    response = QueryResponse(
        session_id=request.session_id,
        query=request.query,
        answer=answer,
        confidence=confidence,
        sources=context_chunks
    )
    
    # Cache result
    query_cache.put(cache_key, response)
    
    latency = time.time() - start_time
    app_logger.info("Query processed", extra={"extra_data": {"latency": latency, "confidence": confidence}})
    
    return response

@router.post("/common-sections", response_model=ComparisonResponse)
async def get_common_sections(request: QueryRequest):
    start_time = time.time()

    chunks = get_session_chunks(request.session_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="No documents uploaded for this session.")

    # Apply file filter if specified
    if request.filter_files:
        chunks = [c for c in chunks if c.source_file in request.filter_files]
        if not chunks:
            raise HTTPException(status_code=400, detail=f"No chunks found for the selected file(s): {request.filter_files}")

    clusterer = SemanticClusterer(embeddings)

    # Run blocking clustering + per-cluster LLM calls off the event loop
    clusters = await asyncio.to_thread(clusterer.cluster_chunks, chunks, llm)

    # Build overall differences/commonalities summary (also blocking LLM call)
    def _overall_summary():
        prompt = """
        You are an expert document analyst. Below is a list of topics and their summaries extracted from multiple documents.
        Your task is to provide a meta-analysis specifically highlighting:
        1. DIFFERENCES: Where do the documents disagree or offer unique information?
        2. COMMONALITIES: What do most or all documents agree on?
        
        Be concise and professional.
        
        Format:
        DIFFERENCES: <text>
        COMMONALITIES: <text>
        
        Topics:
        """
        for c in clusters:
            prompt += f"- Topic: {c.topic}\n  Summary: {c.summary}\n\n"
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, 'content') else str(resp)

    overall_text = await asyncio.to_thread(_overall_summary)

    diff_match = re.search(r"DIFFERENCES:\s*(.*?)(?=COMMONALITIES:|$)", overall_text, re.IGNORECASE | re.DOTALL)
    comm_match = re.search(r"COMMONALITIES:\s*(.*)", overall_text, re.IGNORECASE | re.DOTALL)
    
    differences = diff_match.group(1).strip() if diff_match else "See detailed clusters."
    commonalities = comm_match.group(1).strip() if comm_match else "See detailed clusters."

    if not differences or differences.lower() == "none":
        differences = "No significant differences found across documents."
    if not commonalities or commonalities.lower() == "none":
        commonalities = "No significant commonalities found across documents."

    response = ComparisonResponse(
        session_id=request.session_id,
        clusters=clusters,
        differences=differences,
        commonalities=commonalities
    )

    latency = time.time() - start_time
    app_logger.info("Common sections processed", extra={"extra_data": {"latency": latency, "num_clusters": len(clusters)}})

    return response
