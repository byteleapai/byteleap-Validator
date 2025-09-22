import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import bittensor as bt


class LRUProofCache:
    """Thread-safe LRU cache for challenge proof data with fixed capacity"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def store_proof(self, hotkey: str, proof_data: Dict[str, Any]) -> List[str]:
        """
        Store proof data for a hotkey, returning list of evicted challenge_ids

        Args:
            hotkey: The hotkey to store proof for
            proof_data: The proof data to store

        Returns:
            List of challenge_ids that were evicted due to capacity limits
        """
        evicted_challenge_ids = []

        with self._lock:
            # If key exists, remove it first to update position
            if hotkey in self._cache:
                del self._cache[hotkey]

            # Check capacity and evict if necessary
            while len(self._cache) >= self.max_size:
                # Evict least recently used (first item)
                evicted_hotkey, evicted_data = self._cache.popitem(last=False)
                evicted_challenge_id = evicted_data.get("challenge_id")
                if evicted_challenge_id:
                    evicted_challenge_ids.append(evicted_challenge_id)
                    bt.logging.debug(
                        f"cache evicted | hotkey={evicted_hotkey[:8]}... challenge_id={evicted_challenge_id}"
                    )

            # Store new data (will be added at end - most recently used)
            self._cache[hotkey] = proof_data

        bt.logging.debug(
            f"proof stored | hotkey={hotkey[:8]}... cache_size={len(self._cache)}"
        )

        return evicted_challenge_ids

    def get_proof(self, hotkey: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve proof data for a hotkey and mark as recently used

        Args:
            hotkey: The hotkey to retrieve proof for

        Returns:
            Proof data if found, None otherwise
        """
        with self._lock:
            if hotkey not in self._cache:
                return None

            # Move to end (mark as most recently used)
            proof_data = self._cache[hotkey]
            self._cache.move_to_end(hotkey)

            return proof_data

    def remove_proof(self, hotkey: str) -> bool:
        """
        Remove proof data for a hotkey

        Args:
            hotkey: The hotkey to remove proof for

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if hotkey in self._cache:
                del self._cache[hotkey]
                bt.logging.debug(f"proof removed | hotkey={hotkey[:8]}...")
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
