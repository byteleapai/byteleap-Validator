"""
Database Model Definition
Define PostgreSQL database models for storing miner information
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import bittensor as bt
from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Index, Integer, String, Text, create_engine, or_, text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

Base = declarative_base()

# Performance tracking constants
TASK_TIME_SLIDING_WINDOW = 100  # Track average time for last N tasks


class WorkerInfo(Base):
    """Individual worker tracking and metrics within miner hotkey scope"""

    __tablename__ = "worker_info"

    # Core identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String(256), index=True, nullable=False)
    hotkey = Column(
        String(256), ForeignKey("miner_info.hotkey"), index=True, nullable=False
    )

    # Worker metadata
    worker_name = Column(String(128), nullable=True)
    worker_version = Column(String(32))
    capabilities = Column(JSON, default=list)

    # Network status
    is_online = Column(Boolean, default=False)
    last_heartbeat = Column(DateTime)
    next_heartbeat_deadline = Column(DateTime, nullable=True)

    # Task performance
    tasks_completed = Column(Integer, default=0)
    tasks_failed = Column(Integer, default=0)
    avg_task_time_ms = Column(Float, default=0.0)

    # Lease management
    lease_score = Column(Float, default=0.0)
    lease_updated_at = Column(DateTime, nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    miner = relationship("MinerInfo", back_populates="workers")

    __table_args__ = (
        Index("idx_worker_hotkey", "hotkey"),
        Index("idx_worker_name", "worker_name"),
        Index("idx_worker_online", "is_online"),
        Index("idx_worker_heartbeat", "last_heartbeat"),
        Index("idx_worker_deleted", "deleted_at"),
        Index("idx_worker_hotkey_worker_id", "hotkey", "worker_id", unique=True),
        Index("idx_worker_lease_score", "lease_score"),
        Index("idx_worker_next_heartbeat", "next_heartbeat_deadline"),
        # Lease ranking per miner (partial)
        Index(
            "idx_worker_hotkey_lease_active",
            "hotkey",
            "lease_score",
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )


class MinerInfo(Base):
    """Miner registration and status tracking"""

    __tablename__ = "miner_info"

    # Core identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    hotkey = Column(String(256), unique=True, index=True, nullable=False)
    miner_version = Column(String(32))

    # Network information
    public_ip = Column(String(45))
    is_online = Column(Boolean, default=False)
    last_heartbeat = Column(DateTime)

    # Challenge tracking
    last_challenge_time = Column(DateTime)

    # Weight management
    current_weight = Column(Float, default=0.0)
    last_weight_update = Column(DateTime)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    hardware_info = relationship("HardwareInfo", back_populates="miner")
    workers = relationship("WorkerInfo", back_populates="miner")
    gpu_inventory = relationship("GPUInventory", back_populates="miner")

    __table_args__ = (
        Index("idx_miner_online", "is_online"),
        Index("idx_heartbeat_time", "last_heartbeat"),
        Index("idx_weight_update", "last_weight_update"),
        Index("idx_miner_deleted", "deleted_at"),
    )


class HardwareInfo(Base):
    """Hardware specifications and performance metrics for workers"""

    __tablename__ = "hardware_info"

    # Core identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    hotkey = Column(
        String(256),
        ForeignKey("miner_info.hotkey"),
        index=True,
        nullable=False,
    )
    worker_id = Column(String(256), index=True, nullable=False)

    # CPU information
    cpu_count = Column(Integer)
    cpu_brand = Column(String(128))
    cpu_model = Column(String(256))
    cpu_architecture = Column(String(64))
    cpu_frequency_mhz = Column(Float)
    cpu_max_frequency_mhz = Column(Float)
    cpu_info = Column(JSON)

    # Memory information
    memory_total_mb = Column(Integer)
    memory_type = Column(String(32))
    memory_frequency_mhz = Column(Float)
    memory_info = Column(JSON)

    # Storage information
    disk_total_gb = Column(Integer)
    disk_type = Column(String(32))
    storage_info = Column(JSON)

    # GPU information
    gpu_count = Column(Integer, default=0)
    gpu_info = Column(JSON)

    # Motherboard information
    motherboard_brand = Column(String(128))
    motherboard_model = Column(String(256))
    motherboard_bios_version = Column(String(128))
    motherboard_info = Column(JSON)

    # System information
    system_os = Column(String(128))
    system_os_version = Column(String(128))
    system_kernel_version = Column(String(128))
    system_info = Column(JSON)
    uptime_seconds = Column(Float)

    # Performance metrics
    avg_cpu_usage = Column(Float, default=0.0)
    avg_memory_usage = Column(Float, default=0.0)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    miner = relationship("MinerInfo", back_populates="hardware_info")

    __table_args__ = (
        Index("idx_hardware_miner", "hotkey"),
        Index("idx_hardware_worker", "worker_id"),
        Index("idx_hardware_hotkey_worker_id", "hotkey", "worker_id", unique=True),
        Index("idx_hardware_updated", "updated_at"),
        Index("idx_hardware_deleted", "deleted_at"),
        Index("idx_hardware_cpu_model", "cpu_model"),
        Index("idx_hardware_gpu_count", "gpu_count"),
    )


class GPUInventory(Base):
    """GPU inventory tracking individual GPU units and their activity"""

    __tablename__ = "gpu_inventory"

    # Core identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    gpu_uuid = Column(String(64), unique=True, index=True, nullable=False)

    # Worker association
    hotkey = Column(
        String(256),
        ForeignKey("miner_info.hotkey"),
        index=True,
        nullable=False,
    )
    worker_id = Column(String(256), index=True, nullable=False)

    # GPU specifications
    gpu_model = Column(String(256), nullable=True)  # e.g., "RTX 4090", "A100"
    gpu_memory_total = Column(Integer, nullable=True)  # VRAM in MB
    gpu_memory_free = Column(Integer, nullable=True)  # Free VRAM in MB
    compute_capability = Column(String(16), nullable=True)  # e.g., "8.9"
    multiprocessor_count = Column(Integer, nullable=True)  # SM count
    clock_rate = Column(Integer, nullable=True)  # Clock rate in kHz
    architecture = Column(String(64), nullable=True)  # e.g., "Ada Lovelace"

    # Performance tracking
    successful_challenges = Column(Integer, default=0)
    failed_challenges = Column(Integer, default=0)
    avg_computation_time_ms = Column(Float, default=0.0)
    last_activity_at = Column(DateTime, nullable=True)  # Last successful verification

    # Status tracking
    is_active = Column(Boolean, default=True)
    last_seen_at = Column(DateTime, default=datetime.utcnow)

    # Additional GPU info
    gpu_info = Column(JSON, nullable=True)  # Extended GPU information

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    # Relationships
    miner = relationship("MinerInfo", back_populates="gpu_inventory")

    __table_args__ = (
        Index("idx_gpu_uuid", "gpu_uuid", unique=True),
        Index("idx_gpu_miner", "hotkey"),
        Index("idx_gpu_worker", "worker_id"),
        Index("idx_gpu_hotkey_worker_id", "hotkey", "worker_id"),
        Index("idx_gpu_model", "gpu_model"),
        Index("idx_gpu_is_active", "is_active"),
        Index("idx_gpu_last_activity", "last_activity_at"),
        Index("idx_gpu_last_seen", "last_seen_at"),
        Index("idx_gpu_deleted", "deleted_at"),
        Index("idx_gpu_compute_capability", "compute_capability"),
        Index("idx_gpu_architecture", "architecture"),
        # Partial index: recent active GPUs for window scans
        Index(
            "idx_gpu_last_seen_active",
            "last_seen_at",
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )


class HeartbeatRecord(Base):
    """Heartbeat record table with worker-level tracking"""

    __tablename__ = "heartbeat_records"

    # Core identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    hotkey = Column(String(256), index=True, nullable=False)
    worker_id = Column(String(256), index=True)

    # System status
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    memory_available_mb = Column(Integer)
    disk_free_gb = Column(Integer)
    gpu_utilization = Column(JSON)

    # Network information
    public_ip = Column(String(45))

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_heartbeat_miner_time", "hotkey", "created_at"),
        Index("idx_heartbeat_worker_time", "worker_id", "created_at"),
        Index("idx_heartbeat_created_at", "created_at"),
        Index("idx_heartbeat_deleted", "deleted_at"),
        # Latest heartbeat per (hotkey, worker)
        Index(
            "idx_hb_hotkey_worker_created_active",
            "hotkey",
            "worker_id",
            "created_at",
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
        # Efficient IP-change scan
        Index(
            "idx_hb_worker_created_ip_active",
            "worker_id",
            "created_at",
            postgresql_where=text("deleted_at IS NULL"),
            postgresql_include=["public_ip"],
        ),
    )


class ComputeChallenge(Base):
    """Compute challenge record table with worker tracking"""

    __tablename__ = "compute_challenges"

    # Core identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    challenge_id = Column(String(128), unique=True, index=True, nullable=False)
    hotkey = Column(String(256), index=True, nullable=False)
    worker_id = Column(String(256), index=True)

    # Challenge configuration
    challenge_type = Column(String(32), nullable=False)
    challenge_data = Column(JSON, nullable=False)
    sent_at = Column(DateTime, nullable=True, index=True)
    expires_at = Column(DateTime, nullable=True)

    # Phase 1 Response
    computation_time_ms = Column(Float)
    computed_at = Column(DateTime)
    merkle_commitments = Column(JSON, nullable=True)

    # Phase 2 Response
    verification_targets = Column(JSON, nullable=True)
    debug_info = Column(JSON, nullable=True)

    # Verification information
    challenge_status = Column(String(20), nullable=False)
    verification_result = Column(Boolean, default=None, nullable=True)
    verification_notes = Column(Text)
    verification_time_ms = Column(Float)
    verified_at = Column(DateTime, nullable=True)
    is_success = Column(Boolean)
    success_count = Column(Integer, default=0)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_challenge_miner_time", "hotkey", "created_at"),
        Index("idx_challenge_worker_time", "worker_id", "created_at"),
        Index("idx_challenge_success", "is_success"),
        Index("idx_challenge_verified", "verified_at"),
        Index("idx_challenge_expires", "expires_at"),
        Index("idx_challenge_deleted", "deleted_at"),
        Index("idx_challenge_status_created", "challenge_status", "created_at"),
        Index("idx_challenge_hotkey_status", "hotkey", "challenge_status"),
        # Verification queue ordering (partial)
        Index(
            "idx_chal_status_comp_created_active",
            "challenge_status",
            "computed_at",
            "created_at",
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
        # Availability: verified flag + created_at (partial)
        Index(
            "idx_chal_verified_created_active",
            "verification_result",
            "created_at",
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )


class MeshHubTask(Base):
    """MeshHub task cache table for validator-side persistence."""

    __tablename__ = "meshhub_tasks"

    # Core identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), unique=True, index=True, nullable=False)
    worker_id = Column(String(256), index=True, nullable=True)
    hotkey = Column(String(64), index=True, nullable=True)

    # Task details
    task_type = Column(String(32), nullable=False)
    task_config = Column(JSON, nullable=False)
    priority = Column(Integer, default=0)

    # Task status
    status = Column(String(20), default="pending", nullable=False)
    sent_at = Column(DateTime, nullable=True, index=True)
    expires_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    failure_reason = Column(Text)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_mesh_task_worker", "worker_id"),
        Index("idx_mesh_task_hotkey", "hotkey"),
        Index("idx_mesh_task_status", "status"),
        Index("idx_mesh_task_deleted", "deleted_at"),
        # Cleanup by created_at (partial)
        Index(
            "idx_mesh_task_created_active",
            "created_at",
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )


class NetworkWeight(Base):
    """Network weight record table"""

    __tablename__ = "network_weights"

    # Core identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    hotkey = Column(String(256), index=True, nullable=False)

    # Weight information
    weight_value = Column(Float, nullable=False)
    calculation_remark = Column(Text)
    apply_remark = Column(String(256))

    # Calculation basis
    challenge_score = Column(Float, default=0.0)
    availability_score = Column(Float, default=0.0)
    lease_score = Column(Float, default=0.0)

    # Status and timing
    is_applied = Column(Boolean, default=False)
    applied_at = Column(DateTime, nullable=True)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_weight_miner_time", "hotkey", "created_at"),
        Index("idx_weight_applied", "is_applied"),
        Index("idx_weight_applied_at", "applied_at"),
        Index("idx_weight_deleted", "deleted_at"),
        # Cleanup by created_at (partial)
        Index(
            "idx_weight_created_active",
            "created_at",
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )


class NetworkLog(Base):
    """Network communication log"""

    __tablename__ = "network_logs"

    # Core identification
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Communication metadata
    direction = Column(String(16), nullable=False)
    endpoint = Column(String(255))
    synapse_type = Column(String(64))

    # Client information
    hotkey = Column(String(255))
    worker_id = Column(String(256))
    client_ip = Column(String(45))
    client_port = Column(Integer)

    # Data storage
    raw_synapse_data = Column(JSON)
    decrypted_data = Column(JSON)

    # Encryption metadata
    encryption_method = Column(String(32), default="session")

    # Processing results
    error_code = Column(Integer, default=0)  # 0 = success, non-zero = error codes
    error_message = Column(Text)
    response_data = Column(JSON)
    processing_time_ms = Column(Float)

    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_network_log_type", "synapse_type"),
        Index("idx_network_log_miner", "hotkey"),
        Index("idx_network_log_worker", "worker_id"),
        Index("idx_network_log_error", "error_code"),
        Index("idx_network_log_direction", "direction"),
        Index("idx_network_log_deleted", "deleted_at"),
        Index(
            "idx_network_log_created", "created_at"
        ),  # Use created_at for time-based queries
    )


class DatabaseManager:
    """Database manager"""

    def __init__(self, database_url: str):
        """
        Initialize database manager

        Args:
            database_url: PostgreSQL connection URL
        """
        # Use different connection parameters for SQLite vs PostgreSQL
        if database_url.startswith("sqlite"):
            self.engine = create_engine(database_url, pool_pre_ping=True)
        else:
            self.engine = create_engine(
                database_url, pool_pre_ping=True, pool_size=10, max_overflow=20
            )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self) -> None:
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def close(self) -> None:
        """Close database connection pool"""
        if hasattr(self, "engine"):
            self.engine.dispose()

    def cleanup_old_data(
        self,
        session: Session,
        retention_days: int = 7,
        min_heartbeat_hours: Optional[int] = None,
    ) -> Dict[str, int]:
        """Hard-delete old rows from event/log tables only.

        Scope (created_at cutoff):
        - network_logs, heartbeat_records, compute_challenges, meshhub_tasks, network_weights

        Heartbeats use a stricter cutoff to preserve at least the availability window
        (min_heartbeat_hours), preventing availability calculation bias when retention_days is small.

        Returns table_name -> deleted_count
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(days=retention_days)
        # Preserve at least the availability window for heartbeats, if provided
        if isinstance(min_heartbeat_hours, int) and min_heartbeat_hours > 0:
            hb_cutoff = now - max(
                timedelta(days=retention_days), timedelta(hours=min_heartbeat_hours)
            )
        else:
            hb_cutoff = cutoff
        deleted: Dict[str, int] = {}

        # Event / history tables (no FK constraints to miner_info)
        try:
            deleted["network_logs"] = (
                session.query(NetworkLog)
                .filter(NetworkLog.created_at < cutoff)
                .delete(synchronize_session=False)
            )
        except Exception:
            deleted["network_logs"] = 0

        try:
            deleted["heartbeat_records"] = (
                session.query(HeartbeatRecord)
                .filter(HeartbeatRecord.created_at < hb_cutoff)
                .delete(synchronize_session=False)
            )
        except Exception:
            deleted["heartbeat_records"] = 0

        try:
            deleted["compute_challenges"] = (
                session.query(ComputeChallenge)
                .filter(ComputeChallenge.created_at < cutoff)
                .delete(synchronize_session=False)
            )
        except Exception:
            deleted["compute_challenges"] = 0

        try:
            deleted["meshhub_tasks"] = (
                session.query(MeshHubTask)
                .filter(MeshHubTask.created_at < cutoff)
                .delete(synchronize_session=False)
            )
        except Exception:
            deleted["meshhub_tasks"] = 0

        try:
            deleted["network_weights"] = (
                session.query(NetworkWeight)
                .filter(NetworkWeight.created_at < cutoff)
                .delete(synchronize_session=False)
            )
        except Exception:
            deleted["network_weights"] = 0

        session.commit()
        return deleted

    def _get_or_create_entity(
        self,
        session: Session,
        model_class,
        filter_conditions: Dict[str, Any],
        **create_kwargs,
    ):
        """Generic get or create method for database entities"""
        query = session.query(model_class)

        # Apply filter conditions
        for field, value in filter_conditions.items():
            if field == "deleted_at":
                query = query.filter(getattr(model_class, field).is_(value))
            else:
                query = query.filter(getattr(model_class, field) == value)

        entity = query.first()

        if not entity:
            # Set default timestamps
            now = datetime.utcnow()
            create_kwargs.setdefault("created_at", now)
            create_kwargs.setdefault("updated_at", now)

            entity = model_class(**create_kwargs)
            session.add(entity)
            session.commit()
            session.refresh(entity)

        return entity

    def get_or_create_miner(self, session: Session, hotkey: str) -> MinerInfo:
        """Get or create miner information"""
        return self._get_or_create_entity(
            session, MinerInfo, {"hotkey": hotkey, "deleted_at": None}, hotkey=hotkey
        )

    def get_or_create_worker(
        self,
        session: Session,
        worker_id: str,
        hotkey: str,
        worker_name: str = None,
        worker_version: str = None,
    ) -> WorkerInfo:
        """Get or create worker information using composite key (hotkey + worker_id)."""
        return self._get_or_create_entity(
            session,
            WorkerInfo,
            {"hotkey": hotkey, "worker_id": worker_id, "deleted_at": None},
            worker_id=worker_id,
            hotkey=hotkey,
            worker_name=worker_name,
            worker_version=worker_version,
        )

    def update_worker_heartbeat(
        self,
        session: Session,
        worker_id: str,
        hotkey: str,
        worker_info: Dict[str, Any],
        heartbeat_interval_minutes: int = 5,
    ) -> None:
        """Update or create worker heartbeat information using composite key"""
        # Ensure the worker exists before updating, creating it if necessary.
        received_worker_name = worker_info.get("worker_name")

        worker = self.get_or_create_worker(
            session,
            worker_id=worker_id,
            hotkey=hotkey,
            worker_name=received_worker_name,
            worker_version=worker_info.get("worker_version"),
        )

        # Update worker status from heartbeat
        now = datetime.utcnow()
        worker.is_online = True  # A heartbeat implies the worker is online
        worker.last_heartbeat = now
        worker.capabilities = worker_info.get("capabilities", [])
        # The version is set at creation, but we can update it on heartbeat as well
        worker.worker_version = worker_info.get("worker_version")
        # Worker names can change dynamically
        if received_worker_name is not None:
            worker.worker_name = received_worker_name
        worker.next_heartbeat_deadline = now + timedelta(
            minutes=heartbeat_interval_minutes * 2
        )  # Allow 2x interval tolerance
        worker.updated_at = now

        # Record worker-specific heartbeat
        system_info = worker_info.get("system_info", {})
        worker_heartbeat = HeartbeatRecord(
            hotkey=hotkey,
            worker_id=worker_id,
            cpu_usage=system_info.get("cpu_usage", 0.0),
            memory_usage=system_info.get("memory_usage", 0.0),
            memory_available_mb=system_info.get("memory_available", 0),
            disk_free_gb=system_info.get("disk_free", 0),
            gpu_utilization=system_info.get("gpu_info", []),
            public_ip=system_info.get("public_ip"),
        )

        session.add(worker_heartbeat)
        session.commit()

    def get_or_create_hardware_info(
        self, session: Session, hotkey: str, worker_id: str
    ) -> HardwareInfo:
        """Get or create hardware information for a specific worker"""
        return self._get_or_create_entity(
            session,
            HardwareInfo,
            {"hotkey": hotkey, "worker_id": worker_id, "deleted_at": None},
            hotkey=hotkey,
            worker_id=worker_id,
        )

    def update_worker_hardware_info(
        self, session: Session, hotkey: str, worker_id: str, system_info: Dict[str, Any]
    ) -> None:
        """Update hardware information for a specific worker"""
        hardware = self.get_or_create_hardware_info(session, hotkey, worker_id)

        # Update hardware information
        hardware.cpu_count = system_info.get("cpu_count", 0)
        hardware.memory_total_mb = system_info.get("memory_total", 0)
        hardware.disk_total_gb = system_info.get("disk_total", 0)
        hardware.gpu_count = len(system_info.get("gpu_info", []))
        hardware.gpu_info = system_info.get("gpu_info", [])

        # CPU information
        cpu_info = system_info.get("cpu_info", {})
        if isinstance(cpu_info, dict):
            hardware.cpu_brand = cpu_info.get("brand")
            hardware.cpu_model = cpu_info.get("model")
            hardware.cpu_architecture = cpu_info.get("architecture")
            freq_info = cpu_info.get("frequency_mhz", {})
            if isinstance(freq_info, dict):
                hardware.cpu_frequency_mhz = freq_info.get("current")
                hardware.cpu_max_frequency_mhz = freq_info.get("max")
            else:
                hardware.cpu_frequency_mhz = freq_info
        hardware.cpu_info = cpu_info

        memory_info = system_info.get("memory_info", {})
        if isinstance(memory_info, dict):
            hardware.memory_type = memory_info.get("type")
            hardware.memory_frequency_mhz = memory_info.get("frequency_mhz")
        hardware.memory_info = memory_info

        motherboard_info = system_info.get("motherboard_info", {})
        if isinstance(motherboard_info, dict):
            hardware.motherboard_brand = motherboard_info.get("brand")
            hardware.motherboard_model = motherboard_info.get(
                "model_identifier"
            ) or motherboard_info.get("model")
            hardware.motherboard_bios_version = motherboard_info.get("bios_version")
        hardware.motherboard_info = motherboard_info

        # Storage information
        storage_info = system_info.get("storage_info", [])
        hardware.storage_info = storage_info
        if isinstance(storage_info, list) and storage_info:
            # Use first storage device info for primary disk type
            primary_storage = storage_info[0]
            if isinstance(primary_storage, dict):
                hardware.disk_type = primary_storage.get("type", "Unknown")

        # System information
        platform_info = system_info.get("platform", {})
        if isinstance(platform_info, dict):
            hardware.system_os = platform_info.get("system")
            hardware.system_os_version = platform_info.get("version")
            hardware.system_kernel_version = platform_info.get("version")
        hardware.system_info = system_info.get("system_info", {})

        hardware.uptime_seconds = system_info.get("uptime_seconds")
        hardware.updated_at = datetime.utcnow()

        # Update performance metrics
        alpha = 0.1
        current_cpu = system_info.get("cpu_usage", 0.0)
        current_memory = system_info.get("memory_usage", 0.0)

        hardware.avg_cpu_usage = (
            current_cpu
            if hardware.avg_cpu_usage is None
            else alpha * current_cpu + (1 - alpha) * hardware.avg_cpu_usage
        )
        hardware.avg_memory_usage = (
            current_memory
            if hardware.avg_memory_usage is None
            else alpha * current_memory + (1 - alpha) * hardware.avg_memory_usage
        )

        session.commit()

    def update_miner_heartbeat(
        self, session: Session, hotkey: str, heartbeat_data: Dict[str, Any]
    ) -> None:
        """Update miner heartbeat information"""
        miner = self.get_or_create_miner(session, hotkey)

        # Use miner_info if available, otherwise use basic heartbeat data
        miner_info = heartbeat_data.get("miner_info", {})
        # Miner software version is provided inside miner_info as miner_version

        # Update miner basic information
        if miner_info:
            miner.public_ip = miner_info.get("public_ip")
            # Update miner software version if provided by miner_info
            if isinstance(miner_info, dict):
                mv = miner_info.get("miner_version")
                if mv:
                    miner.miner_version = mv
        miner.is_online = True
        miner.last_heartbeat = datetime.utcnow()
        miner.updated_at = datetime.utcnow()

        session.commit()

    def get_online_miners(
        self, session: Session, timeout_minutes: int = 5
    ) -> List[MinerInfo]:
        """Get online miners list"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)

        return (
            session.query(MinerInfo)
            .filter(
                MinerInfo.is_online == True,
                MinerInfo.last_heartbeat >= cutoff_time,
                MinerInfo.deleted_at.is_(None),
            )
            .all()
        )

    def get_available_miners(self, session: Session) -> List[MinerInfo]:
        """Get available miners (online with unleased workers)"""
        # Miners need unleased workers for challenges
        from sqlalchemy import exists

        return (
            session.query(MinerInfo)
            .filter(
                MinerInfo.is_online == True,
                MinerInfo.deleted_at.is_(None),
                exists().where(
                    (WorkerInfo.hotkey == MinerInfo.hotkey)
                    & (WorkerInfo.lease_score == 0.0)
                    & (WorkerInfo.deleted_at.is_(None))
                ),
            )
            .all()
        )

    def record_challenge(
        self,
        session: Session,
        challenge_id: str,
        hotkey: str,
        challenge_type: str,
        challenge_data: Dict[str, Any],
        worker_id: str,
        matrix_size: Optional[int] = None,
        challenge_timeout: int = 90,
    ) -> ComputeChallenge:
        """Record compute challenge"""
        from neurons.validator.challenge_status import ChallengeStatus

        now = datetime.utcnow()

        challenge = ComputeChallenge(
            challenge_id=challenge_id,
            hotkey=hotkey,
            worker_id=worker_id,
            challenge_type=challenge_type,
            challenge_data=challenge_data,
            challenge_status=ChallengeStatus.CREATED,
            created_at=now,
            expires_at=None,  # Only set when challenge is sent
        )

        session.add(challenge)
        session.commit()
        session.refresh(challenge)

        return challenge

    def record_weight_update(
        self,
        session: Session,
        hotkey: str,
        weight_value: float,
        scores: Dict[str, float],
        calculation_remark: str = None,
        apply_remark: str = None,
        is_applied: bool = False,
    ) -> NetworkWeight:
        """Record weight update"""
        weight_record = NetworkWeight(
            hotkey=hotkey,
            weight_value=weight_value,
            calculation_remark=calculation_remark,
            apply_remark=apply_remark,
            challenge_score=scores.get("challenge_score", 0.0),
            availability_score=scores.get("availability_score", 0.0),
            lease_score=scores.get("lease_score", 0.0),
            is_applied=is_applied,
        )

        session.add(weight_record)
        session.commit()
        session.refresh(weight_record)

        # Update miner weight information
        miner = (
            session.query(MinerInfo)
            .filter(MinerInfo.hotkey == hotkey, MinerInfo.deleted_at.is_(None))
            .first()
        )
        if miner:
            miner.current_weight = weight_value
            miner.last_weight_update = datetime.utcnow()
            miner.updated_at = datetime.utcnow()
            session.commit()

        return weight_record

    def mark_weight_applied(
        self, session: Session, weight_record_id: int, apply_remark: str = None
    ) -> bool:
        """Mark a weight record as applied to the network"""
        weight_record = (
            session.query(NetworkWeight)
            .filter(
                NetworkWeight.id == weight_record_id, NetworkWeight.deleted_at.is_(None)
            )
            .first()
        )

        if weight_record:
            weight_record.is_applied = True
            weight_record.applied_at = datetime.utcnow()
            if apply_remark:
                weight_record.apply_remark = apply_remark
            weight_record.updated_at = datetime.utcnow()
            session.commit()
            return True

        return False

    def log_network_request(
        self,
        session: Session,
        direction: str,
        endpoint: str = None,
        synapse_type: str = None,
        hotkey: str = None,
        worker_id: str = None,
        client_ip: str = None,
        client_port: int = None,
        raw_synapse_data: dict = None,
        decrypted_data: dict = None,
        error_code: int = 0,
        error_message: str = None,
        response_data: dict = None,
        processing_time_ms: float = None,
    ) -> NetworkLog:
        """Record network communication log"""
        network_log = NetworkLog(
            direction=direction,
            endpoint=endpoint,
            synapse_type=synapse_type,
            hotkey=hotkey,
            worker_id=worker_id,
            client_ip=client_ip,
            client_port=client_port,
            raw_synapse_data=raw_synapse_data,
            decrypted_data=decrypted_data,
            error_code=error_code,
            error_message=error_message,
            response_data=response_data,
            processing_time_ms=processing_time_ms,
            encryption_method="session",
        )

        session.add(network_log)
        session.commit()
        session.refresh(network_log)

        return network_log

    def update_network_log(
        self,
        session: Session,
        log_id: int,
        error_code: int = None,
        error_message: str = None,
        response_data: dict = None,
        processing_time_ms: float = None,
    ) -> NetworkLog:
        """Update network communication log"""
        network_log = (
            session.query(NetworkLog)
            .filter(NetworkLog.id == log_id, NetworkLog.deleted_at.is_(None))
            .first()
        )

        if network_log:
            if error_code is not None:
                network_log.error_code = error_code
            if error_message is not None:
                network_log.error_message = error_message
            if response_data is not None:
                network_log.response_data = response_data
            if processing_time_ms is not None:
                network_log.processing_time_ms = processing_time_ms

            network_log.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(network_log)

        return network_log

    def soft_delete_miner(self, session: Session, hotkey: str) -> bool:
        """Soft delete miner and related hardware info"""
        miner = (
            session.query(MinerInfo)
            .filter(MinerInfo.hotkey == hotkey, MinerInfo.deleted_at.is_(None))
            .first()
        )

        if miner:
            miner.deleted_at = datetime.utcnow()
            miner.updated_at = datetime.utcnow()

            hardware = (
                session.query(HardwareInfo)
                .filter(
                    HardwareInfo.hotkey == hotkey, HardwareInfo.deleted_at.is_(None)
                )
                .first()
            )

            if hardware:
                hardware.deleted_at = datetime.utcnow()
                hardware.updated_at = datetime.utcnow()

            session.commit()
            return True

        return False

    def update_worker_lease_status(
        self, session: Session, hotkey: str, worker_id: str, lease_score: float
    ) -> Optional[WorkerInfo]:
        """
        Update worker lease status (validator-managed)

        Args:
            session: Database session
            hotkey: Miner hotkey
            worker_id: Worker ID
            lease_score: Lease score (0.0 = unleased, >0.0 = leased)
        """
        worker = (
            session.query(WorkerInfo)
            .filter(
                WorkerInfo.hotkey == hotkey,
                WorkerInfo.worker_id == worker_id,
                WorkerInfo.deleted_at.is_(None),
            )
            .first()
        )

        if not worker:
            bt.logging.warning(
                f"Cannot update lease status - worker not found: {hotkey}/{worker_id}"
            )
            return None

        worker.lease_score = lease_score
        worker.lease_updated_at = datetime.utcnow()
        worker.updated_at = datetime.utcnow()

        session.commit()

        bt.logging.info(
            f"Worker lease score set: {hotkey}/{worker_id} -> {lease_score}"
        )

        return worker

    def get_leased_miners(self, session: Session) -> List[MinerInfo]:
        """Get miners with leased workers (lease_score > 0)"""
        from sqlalchemy import exists

        return (
            session.query(MinerInfo)
            .filter(
                MinerInfo.deleted_at.is_(None),
                exists().where(
                    (WorkerInfo.hotkey == MinerInfo.hotkey)
                    & (WorkerInfo.lease_score > 0.0)
                    & (WorkerInfo.deleted_at.is_(None))
                ),
            )
            .all()
        )

    def update_worker_task_statistics(
        self,
        session: Session,
        hotkey: str,
        worker_id: str,
        is_success: bool,
        computation_time_ms: Optional[float] = None,
    ) -> bool:
        """
        Update worker task statistics (completed/failed counts and average time)

        Args:
            session: Database session
            hotkey: Miner hotkey
            worker_id: Worker ID
            is_success: Whether the task was successful
            computation_time_ms: Task computation time in milliseconds

        Returns:
            True if updated successfully, False if worker not found
        """
        worker = (
            session.query(WorkerInfo)
            .filter(
                WorkerInfo.hotkey == hotkey,
                WorkerInfo.worker_id == worker_id,
                WorkerInfo.deleted_at.is_(None),
            )
            .first()
        )

        if not worker:
            bt.logging.warning(
                f"Cannot update task statistics - worker not found: {hotkey}/{worker_id}"
            )
            return False

        # Update task counters atomically
        now = datetime.utcnow()

        if is_success:
            # Atomic increment of tasks_completed
            session.query(WorkerInfo).filter(WorkerInfo.id == worker.id).update(
                {
                    WorkerInfo.tasks_completed: WorkerInfo.tasks_completed + 1,
                    WorkerInfo.updated_at: now,
                },
                synchronize_session=False,
            )
        else:
            # Atomic increment of tasks_failed
            session.query(WorkerInfo).filter(WorkerInfo.id == worker.id).update(
                {
                    WorkerInfo.tasks_failed: WorkerInfo.tasks_failed + 1,
                    WorkerInfo.updated_at: now,
                },
                synchronize_session=False,
            )

        # Refresh worker object to get updated counts
        session.refresh(worker)

        # Update average task time if computation time provided (sliding window of last 100 tasks)
        if computation_time_ms is not None:
            total_tasks = worker.tasks_completed + worker.tasks_failed
            if total_tasks > 0:
                if worker.avg_task_time_ms is None:
                    new_avg = computation_time_ms
                else:
                    # Use sliding window approach for better performance and recent data emphasis
                    # Limit the effective window to configured maximum
                    effective_window = min(total_tasks, TASK_TIME_SLIDING_WINDOW)
                    alpha = 1.0 / effective_window
                    new_avg = (
                        alpha * computation_time_ms
                        + (1 - alpha) * worker.avg_task_time_ms
                    )

                # Update average task time atomically
                session.query(WorkerInfo).filter(WorkerInfo.id == worker.id).update(
                    {WorkerInfo.avg_task_time_ms: new_avg, WorkerInfo.updated_at: now},
                    synchronize_session=False,
                )

        bt.logging.debug(
            f"Updated worker task stats: {hotkey}/{worker_id} "
            f"({'success' if is_success else 'failed'}, "
            f"completed={worker.tasks_completed}, failed={worker.tasks_failed})"
        )

        return True

    def get_worker_by_composite_id(
        self, session: Session, hotkey: str, worker_id: str
    ) -> Optional[WorkerInfo]:
        """Get worker by composite key (hotkey + worker_id)"""
        return (
            session.query(WorkerInfo)
            .filter(
                WorkerInfo.hotkey == hotkey,
                WorkerInfo.worker_id == worker_id,
                WorkerInfo.deleted_at.is_(None),
            )
            .first()
        )

    def get_workers_by_hotkey(self, session: Session, hotkey: str) -> List[WorkerInfo]:
        """Get all workers for a miner hotkey"""
        return (
            session.query(WorkerInfo)
            .filter(WorkerInfo.hotkey == hotkey, WorkerInfo.deleted_at.is_(None))
            .all()
        )

    def get_online_workers(
        self, session: Session, timeout_minutes: int = 5
    ) -> List[WorkerInfo]:
        """Get online workers list"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)

        return (
            session.query(WorkerInfo)
            .filter(
                WorkerInfo.is_online == True,
                WorkerInfo.last_heartbeat >= cutoff_time,
                WorkerInfo.deleted_at.is_(None),
            )
            .all()
        )

    def update_worker_online_status(
        self, session: Session, offline_threshold_minutes: int = 10
    ) -> int:
        """
        Update worker online status based on heartbeat timeout

        Args:
            session: Database session
            offline_threshold_minutes: Minutes after which a worker is considered offline

        Returns:
            Number of workers marked as offline
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=offline_threshold_minutes)

        # Mark workers as offline if they haven't sent heartbeat recently
        result = (
            session.query(WorkerInfo)
            .filter(
                WorkerInfo.is_online == True,
                or_(
                    WorkerInfo.last_heartbeat < cutoff_time,
                    WorkerInfo.last_heartbeat.is_(None),
                ),
                WorkerInfo.deleted_at.is_(None),
            )
            .update(
                {"is_online": False, "updated_at": datetime.utcnow()},
                synchronize_session=False,
            )
        )

        session.commit()
        return result

    def update_miner_online_status(
        self, session: Session, offline_threshold_minutes: int = 10
    ) -> int:
        """
        Update miner online status based on heartbeat timeout

        Args:
            session: Database session
            offline_threshold_minutes: Minutes after which a miner is considered offline

        Returns:
            Number of miners marked as offline
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=offline_threshold_minutes)

        # Mark miners as offline if they haven't sent heartbeat recently
        result = (
            session.query(MinerInfo)
            .filter(
                MinerInfo.is_online == True,
                or_(
                    MinerInfo.last_heartbeat < cutoff_time,
                    MinerInfo.last_heartbeat.is_(None),
                ),
                MinerInfo.deleted_at.is_(None),
            )
            .update(
                {"is_online": False, "updated_at": datetime.utcnow()},
                synchronize_session=False,
            )
        )

        session.commit()
        return result

    def soft_delete_worker(self, session: Session, hotkey: str, worker_id: str) -> bool:
        """Soft delete worker using composite key"""
        worker = (
            session.query(WorkerInfo)
            .filter(
                WorkerInfo.hotkey == hotkey,
                WorkerInfo.worker_id == worker_id,
                WorkerInfo.deleted_at.is_(None),
            )
            .first()
        )

        if worker:
            worker.deleted_at = datetime.utcnow()
            worker.updated_at = datetime.utcnow()
            session.commit()

            return True

        return False

    def get_miner_worker_count(self, session: Session, hotkey: str) -> int:
        """Get current worker count for a miner by counting active workers"""
        return (
            session.query(WorkerInfo)
            .filter(WorkerInfo.hotkey == hotkey, WorkerInfo.deleted_at.is_(None))
            .count()
        )

    def mark_expired_tasks(self, session: Session) -> int:
        """
        Mark expired challenges based on expires_at timestamp

        Returns:
            Number of challenges marked as expired
        """
        now = datetime.utcnow()

        # Find expired challenges that haven't been responded to
        expired_challenges = (
            session.query(ComputeChallenge)
            .filter(
                ComputeChallenge.verification_result.is_(None),
                ComputeChallenge.expires_at < now,
                ComputeChallenge.computed_at.is_(None),
                ComputeChallenge.deleted_at.is_(None),
            )
            .all()
        )

        count = 0
        for challenge in expired_challenges:
            # Mark as failed due to timeout
            challenge.is_success = False
            # computed_at remains NULL since worker never responded
            challenge.verified_at = now  # Validator marks as expired
            challenge.verification_result = False
            challenge.verification_notes = "Challenge expired - timeout"
            challenge.updated_at = now
            count += 1

        if count > 0:
            session.commit()
            bt.logging.info(f"Marked {count} expired challenges as failed")

        return count

    def mark_expired_sent_tasks(self, session: Session) -> int:
        """
        Mark only sent but expired challenges as failed

        Returns:
            Number of sent challenges marked as expired
        """
        from neurons.validator.challenge_status import ChallengeStatus

        now = datetime.utcnow()

        updated_count = (
            session.query(ComputeChallenge)
            .filter(
                ComputeChallenge.challenge_status == ChallengeStatus.SENT,
                ComputeChallenge.expires_at < now,
            )
            .update(
                {
                    "challenge_status": ChallengeStatus.FAILED,
                    "is_success": False,
                    "verification_result": False,
                    "verification_notes": "Challenge timeout after being sent to miner",
                    "verified_at": now,
                    "updated_at": now,
                }
            )
        )

        session.commit()
        bt.logging.debug(
            f"Marked {updated_count} sent but expired challenges as failed"
        )
        return updated_count

    def mark_workers_offline_by_deadline(self, session: Session) -> int:
        """
        Mark workers as offline based on next_heartbeat_deadline
        Also marks GPUs as inactive if not seen for 30 minutes

        Returns:
            Number of workers marked as offline
        """
        now = datetime.utcnow()

        # Mark workers as offline if they've passed their heartbeat deadline
        offline_result = (
            session.query(WorkerInfo)
            .filter(
                WorkerInfo.is_online == True,
                WorkerInfo.next_heartbeat_deadline.isnot(None),
                WorkerInfo.next_heartbeat_deadline < now,
                WorkerInfo.deleted_at.is_(None),
            )
            .update({"is_online": False, "updated_at": now}, synchronize_session=False)
        )

        # Mark GPUs as inactive if not seen for 30 minutes
        gpu_threshold_time = now - timedelta(minutes=30)
        inactive_gpu_result = (
            session.query(GPUInventory)
            .filter(
                GPUInventory.last_seen_at < gpu_threshold_time,
                GPUInventory.is_active == True,
                GPUInventory.deleted_at.is_(None),
            )
            .update({"is_active": False, "updated_at": now}, synchronize_session=False)
        )

        session.commit()

        if offline_result > 0:
            bt.logging.info(
                f"Marked {offline_result} workers as offline due to heartbeat deadline"
            )

        if inactive_gpu_result > 0:
            bt.logging.info(
                f"Marked {inactive_gpu_result} GPUs as inactive (not seen for 30 minutes)"
            )

        return offline_result

    def upsert_gpu_inventory(
        self,
        session: Session,
        gpu_uuid: str,
        hotkey: str,
        worker_id: str,
        gpu_details: Dict[str, Any],
    ) -> GPUInventory:
        """
        Upsert GPU inventory record

        Args:
            session: Database session
            gpu_uuid: Unique GPU identifier
            hotkey: Miner hotkey
            worker_id: Worker identifier
            gpu_details: GPU specifications and info

        Returns:
            GPU inventory record
        """
        # Try to find existing record
        gpu_record = (
            session.query(GPUInventory)
            .filter(
                GPUInventory.gpu_uuid == gpu_uuid,
                GPUInventory.deleted_at.is_(None),
            )
            .first()
        )

        now = datetime.utcnow()

        if gpu_record:
            # Update existing record
            gpu_record.hotkey = hotkey
            gpu_record.worker_id = worker_id
            gpu_record.last_seen_at = now
            gpu_record.updated_at = now

            # Update GPU specifications if provided
            if "name" in gpu_details:
                gpu_record.gpu_model = gpu_details["name"]
            if "total_memory" in gpu_details:
                # Convert bytes to MB for database storage
                gpu_record.gpu_memory_total = int(
                    gpu_details["total_memory"] // (1024 * 1024)
                )
            if "free_memory" in gpu_details:
                # Convert bytes to MB for database storage
                gpu_record.gpu_memory_free = int(
                    gpu_details["free_memory"] // (1024 * 1024)
                )
            if "compute_capability" in gpu_details:
                # Convert [major, minor] array to "major.minor" string
                cc = gpu_details["compute_capability"]
                if isinstance(cc, list) and len(cc) >= 2:
                    gpu_record.compute_capability = f"{cc[0]}.{cc[1]}"
                elif isinstance(cc, str):
                    gpu_record.compute_capability = cc
            if "gpu_cores" in gpu_details:
                gpu_record.multiprocessor_count = gpu_details["gpu_cores"]
            if "gpu_clock_mhz" in gpu_details:
                gpu_record.clock_rate = gpu_details["gpu_clock_mhz"]
            if "architecture" in gpu_details:
                gpu_record.architecture = gpu_details["architecture"]

            # Store extended info
            gpu_record.gpu_info = gpu_details

        else:
            # Create new record
            gpu_record = GPUInventory(
                gpu_uuid=gpu_uuid,
                hotkey=hotkey,
                worker_id=worker_id,
                gpu_model=gpu_details.get("name"),
                gpu_memory_total=(
                    int(gpu_details["total_memory"] // (1024 * 1024))
                    if "total_memory" in gpu_details
                    else None
                ),
                gpu_memory_free=(
                    int(gpu_details["free_memory"] // (1024 * 1024))
                    if "free_memory" in gpu_details
                    else None
                ),
                compute_capability=(
                    f"{gpu_details['compute_capability'][0]}.{gpu_details['compute_capability'][1]}"
                    if "compute_capability" in gpu_details
                    and isinstance(gpu_details["compute_capability"], list)
                    and len(gpu_details["compute_capability"]) >= 2
                    else gpu_details.get("compute_capability")
                ),
                multiprocessor_count=gpu_details.get("multiprocessor_count")
                or gpu_details.get("gpu_cores"),
                clock_rate=gpu_details.get("clock_rate")
                or gpu_details.get("gpu_clock_mhz"),
                architecture=gpu_details.get("architecture"),
                gpu_info=gpu_details,
                is_active=True,
                last_seen_at=now,
                created_at=now,
                updated_at=now,
            )
            session.add(gpu_record)

        session.commit()
        session.refresh(gpu_record)
        return gpu_record

    def update_gpu_activity(
        self,
        session: Session,
        gpu_uuid: str,
        is_successful: bool,
        computation_time_ms: Optional[float] = None,
    ) -> bool:
        """
        Update GPU activity tracking after challenge completion

        Args:
            session: Database session
            gpu_uuid: GPU identifier
            is_successful: Whether the challenge was successful
            computation_time_ms: Challenge computation time

        Returns:
            True if updated successfully
        """
        gpu_record = (
            session.query(GPUInventory)
            .filter(
                GPUInventory.gpu_uuid == gpu_uuid,
                GPUInventory.deleted_at.is_(None),
            )
            .first()
        )

        if not gpu_record:
            bt.logging.warning(f"GPU inventory record not found for UUID: {gpu_uuid}")
            return False

        now = datetime.utcnow()

        # Update basic fields first
        gpu_record.last_activity_at = now
        gpu_record.is_active = (
            True  # GPU is definitely active if it just completed a challenge
        )
        gpu_record.updated_at = now

        if is_successful:
            # Atomic increment of successful_challenges
            session.query(GPUInventory).filter(GPUInventory.id == gpu_record.id).update(
                {
                    GPUInventory.successful_challenges: GPUInventory.successful_challenges
                    + 1,
                    GPUInventory.last_activity_at: now,
                    GPUInventory.is_active: True,
                    GPUInventory.updated_at: now,
                },
                synchronize_session=False,
            )

            # Refresh to get updated count for average calculation
            session.refresh(gpu_record)

            # Update average computation time if provided (sliding window of last 100 successful tasks)
            if computation_time_ms is not None:
                current_avg = gpu_record.avg_computation_time_ms or 0.0
                total_successful = gpu_record.successful_challenges

                if current_avg == 0.0:
                    new_avg = computation_time_ms
                else:
                    # Use sliding window approach for better performance and recent data emphasis
                    # Limit the effective window to configured maximum
                    effective_window = min(total_successful, TASK_TIME_SLIDING_WINDOW)
                    alpha = 1.0 / effective_window
                    new_avg = alpha * computation_time_ms + (1 - alpha) * current_avg

                # Update average atomically
                session.query(GPUInventory).filter(
                    GPUInventory.id == gpu_record.id
                ).update(
                    {
                        GPUInventory.avg_computation_time_ms: new_avg,
                        GPUInventory.updated_at: now,
                    },
                    synchronize_session=False,
                )
        else:
            # Atomic increment of failed_challenges
            session.query(GPUInventory).filter(GPUInventory.id == gpu_record.id).update(
                {
                    GPUInventory.failed_challenges: GPUInventory.failed_challenges + 1,
                    GPUInventory.last_activity_at: now,
                    GPUInventory.is_active: True,
                    GPUInventory.updated_at: now,
                },
                synchronize_session=False,
            )

        session.commit()
        return True

    def get_gpu_inventory_by_worker(
        self, session: Session, hotkey: str, worker_id: str
    ) -> List[GPUInventory]:
        """Get GPU inventory for a specific worker"""
        return (
            session.query(GPUInventory)
            .filter(
                GPUInventory.hotkey == hotkey,
                GPUInventory.worker_id == worker_id,
                GPUInventory.deleted_at.is_(None),
            )
            .all()
        )

    def get_gpu_by_uuid(
        self, session: Session, gpu_uuid: str
    ) -> Optional[GPUInventory]:
        """Get GPU record by UUID"""
        return (
            session.query(GPUInventory)
            .filter(
                GPUInventory.gpu_uuid == gpu_uuid,
                GPUInventory.deleted_at.is_(None),
            )
            .first()
        )

    # --- MeshHub integration helpers ---
    def record_meshhub_task(
        self,
        session: Session,
        task_id: str,
        task_type: str,
        task_config: Dict[str, Any],
        priority: int = 0,
        worker_id: Optional[str] = None,
        hotkey: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        status: str = "pending",
    ) -> MeshHubTask:
        """Persist a MeshHub-dispatched task according to the MeshHub schema."""
        now = datetime.utcnow()
        entity = MeshHubTask(
            task_id=task_id,
            task_type=task_type,
            task_config=task_config or {},
            priority=priority or 0,
            worker_id=worker_id,
            hotkey=hotkey,
            expires_at=expires_at,
            status=status or "pending",
            created_at=now,
            updated_at=now,
        )

        session.add(entity)
        session.commit()
        session.refresh(entity)
        return entity

    def apply_meshhub_lease_scores(
        self, session: Session, worker_scores: "List[Dict[str, Any]]"
    ) -> tuple:
        """Apply MeshHub lease score broadcast to local worker_info table.

        Expects items with keys: workerKey ("<hotkey>:<worker_id>") and score (float-like).
        Returns (updated_count, changes) where changes is a list of {workerKey, from, to} for actual modifications.
        """
        if not worker_scores:
            return 0, []

        updated = 0
        changes: list = []
        now = datetime.utcnow()

        for item in worker_scores:
            try:
                worker_key = item.get("workerKey")
                score_val = item.get("score")
                if worker_key is None or score_val is None:
                    continue
                try:
                    score = float(score_val)
                except Exception:
                    continue

                if ":" not in worker_key:
                    continue
                hotkey, worker_id = worker_key.split(":", 1)

                worker: WorkerInfo = (
                    session.query(WorkerInfo)
                    .filter(
                        WorkerInfo.hotkey == hotkey,
                        WorkerInfo.worker_id == worker_id,
                        WorkerInfo.deleted_at.is_(None),
                    )
                    .first()
                )
                if not worker:
                    # If worker not present yet, skip silently; it may be discovered later.
                    continue

                old = worker.lease_score if worker.lease_score is not None else 0.0
                # Only record change if value differs meaningfully
                if abs((score if score is not None else 0.0) - old) > 1e-12:
                    worker.lease_score = score
                    worker.lease_updated_at = now
                    worker.updated_at = now
                    updated += 1
                    changes.append(
                        {
                            "workerKey": worker_key,
                            "from": float(old),
                            "to": float(score),
                        }
                    )
            except Exception:
                # Keep updating others even if one fails
                continue

        if updated:
            session.commit()

        return updated, changes
