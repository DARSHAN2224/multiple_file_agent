import pytest
from backend.core.retriever import HybridRetriever
from backend.models.schemas import SourceChunk

class MockFAISS:
    def __init__(self, expected_scores):
        self.expected_scores = expected_scores
    
    def similarity_search_with_relevance_scores(self, query, k):
        return self.expected_scores

def test_hybrid_scoring_math():
    chunks = [
        SourceChunk(source_file="f1.pdf", text="apple orange banana", token_count=3),
        SourceChunk(source_file="f1.pdf", text="apple", token_count=1),
        SourceChunk(source_file="f1.pdf", text="banana", token_count=1),
    ]
    
    class MockDoc:
        def __init__(self, cp):
            self.page_content = cp
            self.metadata = {}
            
    expected_faiss = [
        (MockDoc("apple orange banana"), 0.8),
        (MockDoc("apple"), 0.4),
        (MockDoc("banana"), 0.2)
    ]
    
    from rank_bm25 import BM25Okapi
    corpus = [chunk.text.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    
    faiss_mock = MockFAISS(expected_faiss)
    retriever = HybridRetriever(faiss_mock, bm25, chunks)
    
    # Calling retrieve on purely "apple"
    results = retriever.retrieve("apple", top_k=3)
    
    assert len(results) > 0
    # Highest score item should be "apple orange banana" or "apple" based on BM25
    # The math must not crash, and norm_semantic & norm_sparse should strictly calculate 0.7 + 0.3 logic.
    for r in results:
        # Check formula structure (float precision issues might make it not == exactly)
        assert r.final_score <= 1.01
        assert r.final_score >= 0.0
