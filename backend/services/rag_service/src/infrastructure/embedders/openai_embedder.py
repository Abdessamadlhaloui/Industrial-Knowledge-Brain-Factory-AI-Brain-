import asyncio
import logging
import os
from typing import List

from openai import AsyncOpenAI
from openai import RateLimitError
from sentence_transformers import SentenceTransformer

from backend.shared.utils.retry import retry

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """
    Embedder using OpenAI's text-embedding-3-large model (3072 dims).
    Supports batching up to 100 texts per API call with exponential backoff on rate limits.
    Falls back to a local SentenceTransformers model (BGE-M3 or fallback) if no API key is provided.
    """

    def __init__(self, model_name: str = "text-embedding-3-large", batch_size: int = 100):
        self.model_name = model_name
        self.batch_size = batch_size
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.local_fallback = None
            logger.info("Initialized OpenAIEmbedder with %s", self.model_name)
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY not set. Falling back to local SentenceTransformers.")
            # Default to a decent local model for dense vectors
            self.local_fallback = SentenceTransformer("BAAI/bge-m3")

    @retry(max_attempts=4, exceptions=(RateLimitError,), backoff_factor=2.0)
    async def _embed_batch_openai(self, batch: List[str]) -> List[List[float]]:
        """Call OpenAI API with retry logic."""
        if not self.client:
            raise ValueError("OpenAI client not initialized.")
            
        response = await self.client.embeddings.create(
            input=batch,
            model=self.model_name
        )
        # response.data is sorted by index
        return [item.embedding for item in response.data]

    async def _embed_batch_local(self, batch: List[str]) -> List[List[float]]:
        """Run local embedding off the main thread to prevent blocking."""
        if not self.local_fallback:
            raise ValueError("Local fallback model not initialized.")
            
        # Run sentence-transformers in a thread pool since it's synchronous CPU bound
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, self.local_fallback.encode, batch)
        return embeddings.tolist()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings, splitting them into batches automatically.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if self.client:
                batch_embeddings = await self._embed_batch_openai(batch)
            else:
                batch_embeddings = await self._embed_batch_local(batch)
                
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
