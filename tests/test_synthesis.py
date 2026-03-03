import pytest
from backend.core.synthesis import SynthesisEngine
from backend.models.schemas import RetrieverResult, SourceChunk
from backend.config import settings

class MockLLM:
    def invoke(self, prompt):
        class Resp:
            content = "Mocked Answer"
        return Resp()

def test_context_window_trimming():
    engine = SynthesisEngine(MockLLM())
    
    # Create chunks that will exceed MAX_CONTEXT_TOKENS
    # Suppose MAX_CONTEXT_TOKENS is 4000
    
    results = [
        RetrieverResult(
            chunk=SourceChunk(source_file="f1.pdf", text="chunk 1", token_count=2000),
            semantic_score=0.9, sparse_score=0.9, final_score=0.9
        ),
        RetrieverResult(
            chunk=SourceChunk(source_file="f1.pdf", text="chunk 2", token_count=1500),
            semantic_score=0.8, sparse_score=0.8, final_score=0.8
        ),
        RetrieverResult(
            chunk=SourceChunk(source_file="f1.pdf", text="chunk 3", token_count=1000), # This one pushes us over 4000 (2000+1500+1000 = 4500)
            semantic_score=0.7, sparse_score=0.7, final_score=0.7
        )
    ]
    
    context_chunks, max_score = engine.enforce_context_window(results)
    
    assert len(context_chunks) == 2
    assert context_chunks[0].text == "chunk 1"
    assert context_chunks[1].text == "chunk 2"
    assert max_score == 0.9

def test_below_threshold_returns_not_found():
    engine = SynthesisEngine(MockLLM())
    
    # Results all below 0.75 strict threshold
    results = [
        RetrieverResult(
            chunk=SourceChunk(source_file="f1.pdf", text="chunk 1", token_count=100),
            semantic_score=0.6, sparse_score=0.5, final_score=0.57
        )
    ]
    
    answer, chunks, score = engine.generate_answer("what is this?", results)
    
    assert "No references to that topic were found" in answer
    assert score == 0.57
