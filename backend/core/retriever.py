from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np
from backend.models.schemas import SourceChunk, RetrieverResult
from backend.utils import deduplicate_chunks
from backend.config import settings

class HybridRetriever:
    """Implements Semantic (FAISS) + Keyword (BM25) deterministic re-ranking."""
    
    def __init__(self, faiss_store: FAISS, bm25_model: BM25Okapi, all_chunks: List[SourceChunk]):
        self.faiss_store = faiss_store
        self.all_chunks = all_chunks
        self.bm25_model = bm25_model
        
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-Max normalization to bring scores between 0 and 1.
        
        If all scores are identical (including all-zero), returns equal positive
        weights (0.5) so that chunks are not silently discarded.
        """
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s - min_s == 0:
            # All scores equal — assign a neutral weight so chunks are not zeroed out
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
        
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        """Retrieve chunks using hybrid deterministic scoring formula."""
        if not self.all_chunks:
            return []

        # Step 1: Semantic Retrieval
        # Request more than top_k from FAISS to allow re-ranking to shift items.
        # `similarity_search_with_relevance_scores` converts L2 distance to a score in
        # (-inf, 1]. Clamp to [0, 1] so negative distances don't collapse normalization.
        semantic_docs_with_scores = self.faiss_store.similarity_search_with_relevance_scores(query, k=top_k*2)
        
        # Build mapping for semantic scores — clamp negatives to 0.0
        semantic_map = {}
        for doc, score in semantic_docs_with_scores:
            semantic_map[hash(doc.page_content)] = max(0.0, score)

        # Step 2: Keyword Retrieval (Sparse)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_model.get_scores(tokenized_query) if self.bm25_model else [0.0] * len(self.all_chunks)
        
        # Build mapping for sparse scores mapping back to chunks
        sparse_map = {hash(chunk.text): score for chunk, score in zip(self.all_chunks, bm25_scores)}
        
        # Extract all candidate hashes that appeared in semantic search OR top subset of BM25
        top_sparse_indices = np.argsort(bm25_scores)[-top_k*2:] if len(bm25_scores) > top_k*2 else range(len(bm25_scores))
        top_sparse_hashes = [hash(self.all_chunks[i].text) for i in top_sparse_indices]
        
        candidate_hashes = set(semantic_map.keys()).union(set(top_sparse_hashes))
        
        candidates: List[RetrieverResult] = []
        raw_semantic = []
        raw_sparse = []
        
        # Find corresponding chunk for each hash
        hash_to_chunk = {hash(c.text): c for c in self.all_chunks}
        
        for chash in candidate_hashes:
            if chash not in hash_to_chunk:
                continue
            sem_val = semantic_map.get(chash, 0.0)
            spar_val = sparse_map.get(chash, 0.0)
            
            raw_semantic.append(sem_val)
            raw_sparse.append(spar_val)
            
            candidates.append(RetrieverResult(
                chunk=hash_to_chunk[chash],
                semantic_score=sem_val,
                sparse_score=spar_val,
                final_score=0.0 # Will calculate next
            ))
            
        # Step 3: Lightweight Normalized Weighted Re-ranking
        norm_sem = self._normalize_scores(raw_semantic)
        norm_spar = self._normalize_scores(raw_sparse)
        
        for i, candidate in enumerate(candidates):
            candidate.semantic_score = norm_sem[i]
            candidate.sparse_score = norm_spar[i]
            # Deterministic Formula: 70% Dense + 30% Sparse
            candidate.final_score = (0.7 * norm_sem[i]) + (0.3 * norm_spar[i])
            
        # Sort by final score descending
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Deduplicate to preserve context window breadth
        unique_candidates = deduplicate_chunks(candidates)
        
        return unique_candidates[:top_k]
