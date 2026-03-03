from typing import List, Tuple
from backend.models.schemas import SourceChunk, RetrieverResult
from backend.config import settings
from backend.utils import app_logger

class SynthesisEngine:
    """Manages Context Window Trimming, Threshold Enforcement, and Strict LLM Generation."""
    
    def __init__(self, llm_engine):
        self.llm = llm_engine
        
    def enforce_context_window(self, retrieved_results: List[RetrieverResult]) -> Tuple[List[SourceChunk], float]:
        """Strictly trims chunks to prevent context window overflow."""
        if not retrieved_results:
            return [], 0.0
            
        max_retrieved_score = max(r.final_score for r in retrieved_results)
        
        context_chunks: List[SourceChunk] = []
        current_tokens = 0
        
        for result in retrieved_results:
            if current_tokens + result.chunk.token_count <= settings.max_context_tokens:
                context_chunks.append(result.chunk)
                current_tokens += result.chunk.token_count
            else:
                app_logger.warning("Context window trimming activated.", extra={"extra_data": {
                    "max_tokens": settings.max_context_tokens,
                    "attempted_tokens": current_tokens + result.chunk.token_count
                }})
                break
                
        return context_chunks, max_retrieved_score
        
    def generate_answer(self, query: str, retrieved_results: List[RetrieverResult]) -> Tuple[str, List[SourceChunk], float]:
        """Synthesizes answer with rigorous hallucination guardrails."""
        
        # 1. Enforce Token Bounds
        context_chunks, max_score = self.enforce_context_window(retrieved_results)
        
        # 2. Guard: if no chunks survived context trimming, there is nothing to answer from
        if not context_chunks:
            return "No references to that topic were found in the uploaded documents.", [], 0.0

        # Note: we intentionally do NOT gate on a numerical score threshold here.
        # The hybrid final_score is a relative ranking metric, not an absolute similarity measure.
        # The LLM prompt instructs the model to respond with the exact fallback phrase if the
        # topic is genuinely absent \u2014 that is the true hallucination guard.
        app_logger.info(
            "Proceeding to LLM synthesis.",
            extra={"extra_data": {"max_score": round(max_score, 4), "num_chunks": len(context_chunks)}},
        )

            
        # Build Context String with Traceability Metadata
        # Use actual filenames as citation labels so the LLM can reference them directly
        context_str_parts = []
        for i, chunk in enumerate(context_chunks):
            label = f"[{chunk.source_file}"
            if chunk.page_number:
                label += f" | Page {chunk.page_number}"
            if chunk.section_title and chunk.section_title != "General":
                label += f" | Section: {chunk.section_title}"
            label += "]"

            context_str_parts.append(f"--- Source {i+1}: {label} ---\n{chunk.text}")
            
        context_str = "\n\n".join(context_str_parts)
        
        # Determine file types present in context for better intent handling
        file_types = set()
        for chunk in context_chunks:
            ext = chunk.source_file.rsplit('.', 1)[-1].lower() if '.' in chunk.source_file else 'unknown'
            file_types.add(ext)

        file_type_hint = ""
        if file_types & {'csv', 'xlsx', 'xls'}:
            file_type_hint = "\n        - The context includes tabular/spreadsheet data. For data questions, present key values clearly (e.g. as a list or table-style text). If the user asks 'what is in this file?' describe the columns and data patterns."
        if file_types & {'pdf', 'docx', 'txt'}:
            file_type_hint += "\n        - The context includes document text. For 'what is this about?' or summary questions, give a structured paragraph-level summary covering the main topics."

        # 3. Enriched Intent-Aware Anti-Hallucination Prompting
        system_prompt = f"""
        You are an intelligent enterprise document analysis assistant. Answer the user's question by reasoning carefully about what they want to know, then answer using ONLY the provided context chunks below.

        Rules:
        - Understand the USER'S INTENT first. A question like "what is this about?" wants a summary. "What are the requirements?" wants a list. "Show me the data" wants key values.
        - You MAY synthesize, summarize, and combine information from different context chunks to form a complete, coherent answer.
        - You MUST ground every claim in the provided context. Do not invent facts, figures, names, or details not present in the context.
        - If the topic is genuinely not mentioned anywhere in the context, respond with exactly: "No references to that topic were found in the uploaded documents."
        - For broad/overview questions, give a concise, well-structured answer covering the main themes.
        - When citing information, use the EXACT source label from the context header, e.g. [filename.pdf | Page 2]. This tells the user which specific file the information came from.
        - Keep your answer concise and professional. Use bullet points or numbered lists where appropriate.{file_type_hint}

        CONTEXT:
        {context_str}

        USER QUERY: {query}
        """
        
        app_logger.info("Invoking LLM for synthesis.", extra={"extra_data": {"query": query}})
        
        try:
            response = self.llm.invoke(system_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            app_logger.error(f"LLM Generation Failed: {str(e)}")
            answer = "Sorry, an internal error occurred while generating the answer."
            
        return answer, context_chunks, max_score
