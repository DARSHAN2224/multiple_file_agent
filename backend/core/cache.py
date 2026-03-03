from collections import OrderedDict
import hashlib
from typing import Any, Optional

class LRUCache:
    """Least Recently Used cache for repeated queries to achieve < 5s latency."""
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

class EmbeddingCache:
    """Hashes text chunks to avoid re-vectorizing identical strings."""
    def __init__(self):
        self._cache = {}

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        return self._cache.get(self._get_hash(text))

    def put(self, text: str, embedding: list[float]):
        self._cache[self._get_hash(text)] = embedding

# Singletons for the application lifecycle
query_cache = LRUCache()
embedding_cache = EmbeddingCache()
