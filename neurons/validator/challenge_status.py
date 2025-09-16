"""
Challenge Status Enumeration

Defines the challenge status state machine and utility methods for challenge lifecycle management.
"""

from typing import List, Optional


class ChallengeStatus:
    """Challenge status enumeration and utility methods"""

    # Status constants
    CREATED = "created"  # Challenge created, not sent to miner
    SENT = "sent"  # Challenge sent to miner, awaiting response
    COMMITTED = "committed"  # Phase 1 commitment received
    VERIFYING = "verifying"  # Phase 2 proof received, async verification queued
    VERIFIED = "verified"  # Verification completed successfully
    FAILED = "failed"  # Verification failed or timeout occurred

    # All valid status values
    ALL_STATUSES = [CREATED, SENT, COMMITTED, VERIFYING, VERIFIED, FAILED]

    @classmethod
    def can_timeout(cls, status: Optional[str]) -> bool:
        """Only sent challenges can timeout from miner non-response"""
        return status == cls.SENT

    @classmethod
    def is_pending_response(cls, status: Optional[str]) -> bool:
        """Challenge is waiting for miner response"""
        return status == cls.SENT

    @classmethod
    def is_pending_send(cls, status: Optional[str]) -> bool:
        """Challenge is waiting to be sent to miner"""
        return status == cls.CREATED

    @classmethod
    def is_processing(cls, status: Optional[str]) -> bool:
        """Challenge is actively being processed (has miner engagement)"""
        return status in [cls.COMMITTED, cls.VERIFYING]

    @classmethod
    def is_completed(cls, status: Optional[str]) -> bool:
        """Challenge processing is finished (success or failure)"""
        return status in [cls.VERIFIED, cls.FAILED]

    @classmethod
    def is_active(cls, status: Optional[str]) -> bool:
        """Challenge is in active processing state (not finished)"""
        return status in [cls.CREATED, cls.SENT, cls.COMMITTED, cls.VERIFYING]

    @classmethod
    def can_expire(cls, status: Optional[str]) -> bool:
        """Challenge can be marked as expired based on expires_at"""
        return status == cls.SENT

    @classmethod
    def requires_timeout_field(cls, status: Optional[str]) -> bool:
        """Status requires expires_at field to be set"""
        return status == cls.SENT

    @classmethod
    def get_next_valid_statuses(cls, current_status: Optional[str]) -> List[str]:
        """Get valid next statuses for state machine validation"""
        transitions = {
            cls.CREATED: [cls.SENT],
            cls.SENT: [cls.COMMITTED, cls.FAILED],  # Can timeout to failed
            cls.COMMITTED: [cls.VERIFYING],
            cls.VERIFYING: [cls.VERIFIED, cls.FAILED],
            cls.VERIFIED: [],  # Terminal state
            cls.FAILED: [],  # Terminal state
        }
        return transitions.get(current_status, [])

    @classmethod
    def validate_transition(cls, from_status: Optional[str], to_status: str) -> bool:
        """Validate if status transition is allowed"""
        valid_next = cls.get_next_valid_statuses(from_status)
        return to_status in valid_next

    @classmethod
    def get_description(cls, status: Optional[str]) -> str:
        """Get human-readable description of status"""
        descriptions = {
            cls.CREATED: "Challenge created, awaiting distribution",
            cls.SENT: "Challenge sent to miner, awaiting response",
            cls.COMMITTED: "Phase 1 commitment received from miner",
            cls.VERIFYING: "Phase 2 proof received, verification in progress",
            cls.VERIFIED: "Verification completed successfully",
            cls.FAILED: "Verification failed or miner timeout",
        }
        return descriptions.get(status, f"Unknown status: {status}")
