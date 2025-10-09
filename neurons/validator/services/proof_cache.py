import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import bittensor as bt


class LRUProofCache:
    """Thread-safe LRU cache for challenge proof data with fixed capacity

    Keyed by a per-worker cache key (e.g., "{hotkey}:{worker_id}").
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def store_proof(self, cache_key: str, proof_data: Dict[str, Any]) -> List[str]:
        """
        Store proof data for a worker cache key, returning list of evicted challenge_ids

        Args:
            cache_key: The cache key (e.g., "{hotkey}:{worker_id}")
            proof_data: The proof data to store

        Returns:
            List of challenge_ids that were evicted due to capacity limits
        """
        evicted_challenge_ids = []

        with self._lock:
            # If key exists, remove it first to update position (keep latest per worker)
            if cache_key in self._cache:
                del self._cache[cache_key]

            # Check capacity and evict if necessary
            while len(self._cache) >= self.max_size:
                # Evict least recently used (first item)
                evicted_key, evicted_data = self._cache.popitem(last=False)
                evicted_challenge_id = evicted_data.get("challenge_id")
                if evicted_challenge_id:
                    evicted_challenge_ids.append(evicted_challenge_id)
                    bt.logging.debug(
                        f"cache evicted | key={evicted_key[:12]}... challenge_id={evicted_challenge_id}"
                    )

            # Store new data at end (most recently used)
            self._cache[cache_key] = proof_data

        bt.logging.debug(
            f"proof stored | key={cache_key[:12]}... cache_size={len(self._cache)}"
        )

        return evicted_challenge_ids

    def get_proof(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve proof data for a worker cache key and mark as recently used

        Args:
            cache_key: The cache key (e.g., "{hotkey}:{worker_id}")

        Returns:
            Proof data if found, None otherwise
        """
        with self._lock:
            if cache_key not in self._cache:
                return None

            # Move to end (mark as most recently used)
            proof_data = self._cache[cache_key]
            self._cache.move_to_end(cache_key)

            return proof_data

    def remove_proof(self, cache_key: str) -> bool:
        """
        Remove proof data for a worker cache key

        Args:
            cache_key: The cache key (e.g., "{hotkey}:{worker_id}")

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                bt.logging.debug(f"proof removed | key={cache_key[:12]}...")
                return True
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "total_entries": len(self._cache),
                "max_size": self.max_size,
                "utilization": (
                    len(self._cache) / self.max_size if self.max_size > 0 else 0.0
                ),
            }

    def clear_all(self) -> int:
        """Clear all cached entries"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            bt.logging.debug(f"cache cleared | removed={count} entries")
            return count

    def shutdown(self):
        """Shutdown the cache"""
        self.clear_all()
        bt.logging.debug("proof cache shutdown")


# Backward compatibility alias
TTLProofCache = LRUProofCache
