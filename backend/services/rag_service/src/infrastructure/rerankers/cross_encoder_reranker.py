import asyncio
import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker utilizing a local cross-encoder model.
    Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast and effective for English).
    Batch reranks (query, passage) pairs and returns top_k results.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info("Loading CrossEncoder model %s", self.model_name)
        self.model = CrossEncoder(self.model_name)

    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10, 
        min_score: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Rerank a list of documents against a query.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of document texts to rerank.
            top_k (int): Number of top results to return.
            min_score (float): Minimum score threshold to keep a result.
            
        Returns:
            List[Tuple[int, float]]: List of tuples containing (original_index, score), sorted descending by score.
        """
        if not documents:
            return []

        # CrossEncoder expects a list of pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc] for doc in documents]
        
        # Run inference in a separate thread to prevent event loop blocking
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None, 
            lambda: self.model.predict(pairs, batch_size=self.batch_size)
        )
        
        # Combine indices with scores
        scored_results = [(idx, float(score)) for idx, score in enumerate(scores)]
        
        # Filter and sort
        filtered_results = [res for res in scored_results if res[1] >= min_score]
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_results[:top_k]
