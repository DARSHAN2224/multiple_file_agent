import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import AgglomerativeClustering
from backend.models.schemas import SourceChunk, ComparisonCluster
from backend.config import settings
from backend.utils import app_logger

class SemanticClusterer:
    """Uses Agglomerative Hierarchical Clustering to group common themes/topics."""
    
    def __init__(self, embeddings_model):
        self.em = embeddings_model
        
    def cluster_chunks(self, chunks: List[SourceChunk], llm) -> List[ComparisonCluster]:
        if not chunks:
            return []
            
        if self.em is None:
            app_logger.warning("Embeddings model not initialized. Skipping semantic clustering.")
            return [ComparisonCluster(
                topic="Raw Document Content",
                chunks=chunks,
                summary="Clustering unavailable due to missing embeddings connection."
            )]
            
        if len(chunks) == 1:
            # Cannot cluster 1 item meaningfully
            topic, summary = self._generate_summary_for_cluster(chunks, llm)
            return [ComparisonCluster(
                topic=topic,
                chunks=chunks,
                summary=summary
            )]
            
        app_logger.info("Starting agglomerative clustering", extra={"extra_data": {"num_chunks": len(chunks)}})
            
        # 1. Embed all chunks (in batches preferably, but for this scale direct is OK)
        texts = [c.text for c in chunks]
        embeddings = self.em.embed_documents(texts)
        X = np.array(embeddings)
        
        # 2. Perform Agglomerative Clustering
        # We use cosine distance. Scikit-learn requires 'euclidean' or precomputed for ward, 
        # so we use 'average' linkage with 'cosine' affinity.
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=settings.distance_threshold
        )
        
        labels = clustering.fit_predict(X)
        
        # 3. Group chunks by label
        clusters_map: Dict[int, List[SourceChunk]] = {}
        for i, label in enumerate(labels):
            if label not in clusters_map:
                clusters_map[label] = []
            clusters_map[label].append(chunks[i])
            
        # 4. Generate summaries / titles for each cluster
        result_clusters = []
        for label, cluster_chunks in clusters_map.items():
            # Quick LLM pass to get a summary and topic name
            topic, summary = self._generate_summary_for_cluster(cluster_chunks, llm)
            result_clusters.append(ComparisonCluster(
                topic=topic,
                chunks=cluster_chunks,
                summary=summary
            ))
            
        app_logger.info("Clustering complete", extra={"extra_data": {"num_clusters": len(result_clusters)}})
        return result_clusters
        
    def _generate_summary_for_cluster(self, chunks: List[SourceChunk], llm) -> tuple[str, str]:
        """Calls LLM to synthesize the cluster's contents into a Topic Name and Summary."""
        combined_text = "\n\n---\n\n".join([f"[{c.source_file}, {c.section_title}] {c.text}" for c in chunks])
        
        # Ensure we don't exceed context window (approx 3000 tokens / 12000 chars safety limit)
        if len(combined_text) > 12000:
            combined_text = combined_text[:12000] + "\n... [TRUNCATED FOR CONTEXT WINDOW] ..."

        prompt = f"""
        You are a document comparison assistant. Below are extracts from multiple documents that share a common theme.
        
        Task 1: Provide a short, 3-5 word Title for this topic.
        Task 2: Provide a concise summary comparing what each document says about this topic. Highlight differences and commonalities.
        
        Format your response exactly like this:
        TITLE: <your title>
        SUMMARY: <your summary>
        
        Extracts:
        {combined_text}
        """
        
        response = llm.invoke(prompt)
        text_resp = response.content if hasattr(response, 'content') else str(response)
        
        title_match = re.search(r"TITLE:\s*(.*?)(?:\n|SUMMARY:|$)", text_resp, re.IGNORECASE)
        summary_match = re.search(r"SUMMARY:\s*(.*)", text_resp, re.IGNORECASE | re.DOTALL)
        
        title = title_match.group(1).strip() if title_match else "Common Theme"
        summary = summary_match.group(1).strip() if summary_match else text_resp
            
        return title, summary
