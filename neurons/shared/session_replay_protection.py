"""
Session Replay Protection
Common constants and sliding window logic for sequence validation
"""

from collections import deque
from typing import Deque, Optional, Set

import bittensor as bt

from neurons.shared.crypto import CryptoManager

# Session sequence constants
MAX_SEQ = 2**64 - 1  # 8-byte sequence number limit for AES-GCM nonce safety


class SlidingWindowValidator:
    """Sliding window replay protection with O(1) lookups"""

    def __init__(self, window_size: Optional[int] = None):
        """
        Initialize sliding window validator

        Args:
            window_size: Size of the replay protection window
        """
        self.window_size = window_size or CryptoManager.REPLAY_WINDOW_SIZE
        self.recv_seq = 0  # Highest sequence received

        if self.window_size > 0:
            self.replay_window: Optional[Deque[int]] = deque(maxlen=self.window_size)
            self.replay_set: Optional[Set[int]] = set()  # O(1) membership checking
        else:
            self.replay_window: Optional[Deque[int]] = None
            self.replay_set: Optional[Set[int]] = None

    def validate_sequence(self, seq: int, context_name: str = "") -> bool:
        """
        Validate sequence number with replay protection

        Args:
            seq: Sequence number to validate
            context_name: Optional context for logging (e.g., "validator", "miner")

        Returns:
            True if sequence is valid (not a replay or too old)
        """
        # If no replay protection, just update sequence
        if self.replay_window is None:
            self.recv_seq = max(self.recv_seq, seq)
            return True

        # At this point, replay protection is enabled, so replay_set is not None
        assert self.replay_set is not None

        # Check for replay using O(1) set lookup
        if seq in self.replay_set:
            bt.logging.warning(
                f"Replay detected{' from ' + context_name if context_name else ''}: seq={seq}"
            )
            return False

        # Check if sequence is too old using fixed window boundary
        window_left_boundary = max(0, self.recv_seq - self.window_size + 1)
        if seq < window_left_boundary:
            bt.logging.warning(
                f"Sequence too old{' from ' + context_name if context_name else ''}: seq={seq}, boundary={window_left_boundary}"
            )
            return False

        # Check for excessive forward jump to prevent DoS
        max_forward_jump = self.window_size * 2  # Allow reasonable forward jumps
        if seq > self.recv_seq + max_forward_jump:
            bt.logging.warning(
                f"Sequence forward jump too large{' from ' + context_name if context_name else ''}: seq={seq}, current={self.recv_seq}, max_jump={max_forward_jump}"
            )
            return False

        # Update sequence tracking
        self.recv_seq = max(self.recv_seq, seq)

        # Manage sliding window with deque+set synchronization
        if len(self.replay_window) == self.window_size:
            # Remove oldest entry from set when deque evicts it
            oldest_seq = self.replay_window[0]
            self.replay_set.discard(oldest_seq)

        self.replay_window.append(seq)
        self.replay_set.add(seq)
        return True


def is_sequence_overflow(seq: int) -> bool:
    """Check if sequence number would overflow"""
    return seq >= MAX_SEQ


def get_max_sequence() -> int:
    """Get maximum allowed sequence number"""
    return MAX_SEQ
