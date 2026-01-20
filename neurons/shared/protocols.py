"""
Subnet Communication Protocol Definition
Define communication protocols and data structures between miner and validator
"""

import base64
import binascii
import time
from enum import IntEnum
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type

import bittensor as bt
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Strict base model normalizing strings."""

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_default=True,
        arbitrary_types_allowed=False,
        use_enum_values=True,
    )


class CommunicationResult(StrictModel):
    """Result of communication operation with metadata"""

    success: bool
    data: Optional[Any] = None
    error_code: int = 0
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0


# Protocol constants for stable synapse type identification
class ProtocolTypes:
    """Stable protocol type constants (immune to class name changes)"""

    HEARTBEAT = "HEARTBEAT_V1"
    TASK = "TASK_V1"
    SESSION_INIT = "SESSION_INIT_V1"
    GET_VMGW_ENROLL_TOKEN = "GET_VMGW_ENROLL_TOKEN_V1"


# Error codes
class ErrorCodes:
    """Standard error codes for the subnet protocol"""

    # Success
    SUCCESS = 0

    # General errors (1000-1099)
    INVALID_REQUEST = 1000
    INVALID_RESPONSE = 1001
    TIMEOUT_ERROR = 1002
    NETWORK_ERROR = 1003
    HEARTBEAT_PROCESSING_FAILED = 1004
    TASK_PROCESSING_FAILED = 1005

    # Configuration errors (2000-2099)
    CONFIG_MISSING_KEY = 2000
    CONFIG_INVALID_VALUE = 2001
    CONFIG_VALIDATION_FAILED = 2002

    # Validation errors (3000-3099)
    VALIDATION_FAILED = 3000
    INVALID_SIGNATURE = 3001
    INVALID_PROOF = 3002
    MERKLE_VERIFICATION_FAILED = 3003

    # Session management errors (4000-4099)
    SESSION_REQUIRED = 4010
    SESSION_UNKNOWN = 4011
    SESSION_EXPIRED = 4012
    REHANDSHAKE_REQUIRED = 4013
    REPLAY_DETECTED = 4014
    BAD_AAD = 4015
    BAD_NONCE = 4016
    SEQ_WINDOW_EXCEEDED = 4017
    HANDSHAKE_FAILED = 4018
    SESSION_LIMIT_EXCEEDED = 4019
    SEQUENCE_ERROR = 4020

    # Persistence / DB errors (5000-5099)
    DB_ERROR = 5000
    DB_CONNECTION_ERROR = 5001


class SystemInfo(StrictModel):
    """System hardware and runtime information for resource assessment"""

    cpu_count: int = Field(default=0, description="Number of CPU cores")
    cpu_usage: float = Field(default=0.0, description="CPU usage percentage")
    memory_total: int = Field(default=0, description="Total memory (MB)")
    memory_available: int = Field(default=0, description="Available memory (MB)")
    memory_usage: float = Field(default=0.0, description="Memory usage percentage")
    disk_total: int = Field(default=0, description="Total disk space (GB)")
    disk_free: int = Field(default=0, description="Available disk space (GB)")
    gpu_info: List[Dict[str, Any]] = Field(
        default_factory=list, description="GPU information list"
    )
    gpu_plugin: List[Dict[str, Any]] = Field(
        default_factory=list, description="GPU plugin details with UUIDs"
    )
    public_ip: Optional[str] = Field(default=None, description="Public IP address")

    cpu_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed CPU specifications"
    )
    memory_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Memory module specifications"
    )
    system_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Operating system details"
    )
    motherboard_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Hardware platform information"
    )
    uptime_seconds: Optional[float] = Field(
        default=None, description="System uptime in seconds"
    )


class WorkerInfo(StrictModel):
    """Individual worker information"""

    worker_id: str = Field(description="Unique worker identifier")
    worker_name: Optional[str] = Field(
        default=None, description="User-configured worker name"
    )
    worker_version: Optional[str] = Field(
        default=None, description="Worker software version"
    )
    capabilities: List[str] = Field(
        default_factory=list, description="Worker capabilities"
    )
    status: Literal["online", "busy", "offline"] = Field(
        default="online", description="Worker status: online, busy, offline"
    )
    system_info: SystemInfo = Field(
        default_factory=SystemInfo, description="Worker system information"
    )
    connected_at: float = Field(
        default_factory=time.time, description="Connection timestamp"
    )
    last_heartbeat: float = Field(
        default_factory=time.time, description="Last heartbeat timestamp"
    )


class HeartbeatData(StrictModel):
    """Heartbeat data containing worker status and miner system info"""

    hotkey: str = Field(description="Miner hotkey address")
    timestamp: float = Field(default_factory=time.time, description="Timestamp")
    workers: List[WorkerInfo] = Field(description="Connected worker information")
    miner_info: Optional[SystemInfo] = Field(
        default=None, description="Miner host system information (optional)"
    )
    schema_version: int = Field(
        default=1, ge=1, description="Schema version for compatibility"
    )


class TaskRequest(StrictModel):
    """Task request (miner polls tasks)"""

    hotkey: str = Field(description="Miner hotkey address")
    request_type: Literal["challenge"] = Field(description="Request type: challenge")
    timestamp: float = Field(default_factory=time.time, description="Request timestamp")
    schema_version: int = Field(
        default=1, ge=1, description="Schema version for compatibility"
    )


class TaskResponse(StrictModel):
    """Task response (task returned by validator to miner)"""

    task_type: Literal["compute_challenge_batch", "no_task"] = Field(
        description="Task type"
    )
    task_data: Optional[Dict[str, Any]] = Field(default=None, description="Task data")
    timestamp: float = Field(
        default_factory=time.time, description="Response timestamp"
    )
    schema_version: int = Field(
        default=1, ge=1, description="Schema version for compatibility"
    )


class EncryptedSynapse(bt.Synapse):
    """Base class for encrypted synapse protocols with unified request/response fields"""

    request: Optional[str] = Field(
        default=None, description="Encrypted request data (base64 encoded)"
    )
    response: Optional[str] = Field(
        default=None, description="Encrypted response data (base64 encoded)"
    )
    protocol_version: int = Field(
        default=1, ge=1, description="Protocol version for compatibility"
    )

    def get_request_data(self) -> Optional[str]:
        """Get encrypted request data"""
        return self.request

    def get_response_data(self) -> Optional[str]:
        """Get encrypted response data"""
        return self.response


class HeartbeatSynapse(EncryptedSynapse):
    """Heartbeat data transmission protocol with mandatory encryption"""

    PROTOCOL_TYPE: ClassVar[str] = ProtocolTypes.HEARTBEAT


class TaskSynapse(EncryptedSynapse):
    """Task pulling protocol with mandatory encryption"""

    PROTOCOL_TYPE: ClassVar[str] = ProtocolTypes.TASK


class GetVmgwEnrollTokenRequest(StrictModel):
    """Request payload for VM gateway enroll token retrieval."""

    worker_id: str = Field(description="Worker ID requesting VM gateway enroll token")

    @model_validator(mode="after")
    def validate_worker_id(self) -> "GetVmgwEnrollTokenRequest":
        if not self.worker_id or not self.worker_id.strip():
            raise ValueError("worker_id cannot be empty")
        return self


class GetVmgwEnrollTokenSynapse(EncryptedSynapse):
    """Synapse for retrieving VM gateway enroll tokens."""

    PROTOCOL_TYPE: ClassVar[str] = ProtocolTypes.GET_VMGW_ENROLL_TOKEN


class HeartbeatResponse(StrictModel):
    """Heartbeat processing response"""

    error_code: int = Field(
        default=0, description="Error code: 0=success, non-zero=error"
    )
    message: str = Field(description="Response message")
    workers_processed: int = Field(default=0, description="Number of workers processed")
    timestamp: float = Field(
        default_factory=time.time, description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        default=None, description="Optional request correlation id"
    )
    schema_version: int = Field(
        default=1, ge=1, description="Schema version for compatibility"
    )


# Session-based encryption protocol classes
class SessionInitRequest(StrictModel):
    """Session initialization request data for ECDH key exchange"""

    miner_eph_pub32: str = Field(
        description="Base64 encoded miner ephemeral public key (32 bytes)"
    )
    client_nonce16: str = Field(description="Base64 encoded client nonce (16 bytes)")
    created_at: float = Field(
        default_factory=time.time, description="Request creation timestamp"
    )

    @model_validator(mode="after")
    def validate_session_init(self) -> "SessionInitRequest":
        # 32 bytes public key
        try:
            raw_pub = base64.b64decode(self.miner_eph_pub32, validate=True)
        except binascii.Error as e:
            raise ValueError("miner_eph_pub32 must be valid base64") from e
        if len(raw_pub) != 32:
            raise ValueError("miner_eph_pub32 must decode to 32 bytes")
        try:
            raw_nonce = base64.b64decode(self.client_nonce16, validate=True)
        except binascii.Error as e:
            raise ValueError("client_nonce16 must be valid base64") from e
        if len(raw_nonce) != 16:
            raise ValueError("client_nonce16 must decode to 16 bytes")
        return self


class SessionInitResponse(StrictModel):
    """Session initialization response data"""

    validator_eph_pub32: str = Field(
        description="Base64 encoded validator ephemeral public key (32 bytes)"
    )
    session_id: str = Field(
        description="Session identifier (UUID or 128-bit random string)"
    )
    server_nonce16: str = Field(description="Base64 encoded server nonce (16 bytes)")
    expires_at: float = Field(description="Session expiration timestamp")

    @model_validator(mode="after")
    def validate_session_response(self) -> "SessionInitResponse":
        try:
            raw_pub = base64.b64decode(self.validator_eph_pub32, validate=True)
        except binascii.Error as e:
            raise ValueError("validator_eph_pub32 must be valid base64") from e
        if len(raw_pub) != 32:
            raise ValueError("validator_eph_pub32 must decode to 32 bytes")
        try:
            raw_nonce = base64.b64decode(self.server_nonce16, validate=True)
        except binascii.Error as e:
            raise ValueError("server_nonce16 must be valid base64") from e
        if len(raw_nonce) != 16:
            raise ValueError("server_nonce16 must decode to 16 bytes")
        if not self.session_id:
            raise ValueError("session_id required")
        return self


class SessionInitSynapse(bt.Synapse):
    """Session initialization synapse (plaintext, relies on Bittensor signature)"""

    PROTOCOL_TYPE: ClassVar[str] = ProtocolTypes.SESSION_INIT

    request: Optional[str] = Field(
        default=None, description="Serialized SessionInitRequest"
    )
    response: Optional[str] = Field(
        default=None, description="Serialized SessionInitResponse"
    )


# Public API exports
class ProtocolRegistry:
    """Registry mapping PROTOCOL_TYPE strings to synapse classes."""

    _registry: Dict[str, Type[bt.Synapse]] = {}

    @classmethod
    def register(cls, synapse_cls: Type[bt.Synapse]) -> Type[bt.Synapse]:
        if not hasattr(synapse_cls, "PROTOCOL_TYPE"):
            raise ValueError("Synapse class missing PROTOCOL_TYPE")
        protocol_key = getattr(synapse_cls, "PROTOCOL_TYPE")
        if not isinstance(protocol_key, str) or not protocol_key:
            raise ValueError("PROTOCOL_TYPE must be non-empty string")
        cls._registry[protocol_key] = synapse_cls
        return synapse_cls

    @classmethod
    def get(cls, protocol_type: str) -> Type[bt.Synapse]:
        return cls._registry[protocol_type]

    @classmethod
    def known_protocols(cls) -> List[str]:
        return list(cls._registry.keys())


# Register built-in synapses
ProtocolRegistry.register(HeartbeatSynapse)
ProtocolRegistry.register(TaskSynapse)
ProtocolRegistry.register(GetVmgwEnrollTokenSynapse)
ProtocolRegistry.register(SessionInitSynapse)

__all__ = [
    # constants and codes
    "ProtocolTypes",
    "ErrorCodes",
    # base
    "StrictModel",
    "CommunicationResult",
    # system and worker
    "SystemInfo",
    "WorkerInfo",
    "HeartbeatData",
    # synapses
    "EncryptedSynapse",
    "HeartbeatSynapse",
    "TaskSynapse",
    "GetVmgwEnrollTokenRequest",
    "GetVmgwEnrollTokenSynapse",
    "SessionInitSynapse",
    "ProtocolRegistry",
    # session
    "SessionInitRequest",
    "SessionInitResponse",
    # responses
    "HeartbeatResponse",
]
