"""add comprehensive indexes and JSONB optimization

Revision ID: e8f2c4d9a1b5
Revises: 3b7f9d12c8ab
Create Date: 2025-12-06 22:20:00+00:00

"""

from alembic import op
from sqlalchemy import text
from sqlalchemy.engine import Connection

# revision identifiers, used by Alembic.
revision = "e8f2c4d9a1b5"
down_revision = "3b7f9d12c8ab"
branch_labels = None
depends_on = None


# JSON columns to convert to JSONB (table_name, column_name)
JSON_COLUMNS = [
    ("worker_info", "capabilities"),
    ("hardware_info", "cpu_info"),
    ("hardware_info", "memory_info"),
    ("hardware_info", "storage_info"),
    ("hardware_info", "gpu_info"),
    ("hardware_info", "motherboard_info"),
    ("hardware_info", "system_info"),
    ("gpu_inventory", "gpu_info"),
    ("heartbeat_records", "gpu_utilization"),
    ("compute_challenges", "challenge_data"),
    ("compute_challenges", "merkle_commitments"),
    ("compute_challenges", "verification_targets"),
    ("compute_challenges", "debug_info"),
    ("meshhub_tasks", "task_config"),
    ("network_logs", "raw_synapse_data"),
    ("network_logs", "decrypted_data"),
    ("network_logs", "response_data"),
]


def is_postgres(conn: Connection) -> bool:
    """Check if the database is PostgreSQL"""
    return conn.dialect.name == "postgresql"


def upgrade() -> None:
    """Add comprehensive performance indexes and convert JSON to JSONB"""
    conn = op.get_bind()

    # ========================================
    # JSON to JSONB conversion (PostgreSQL only)
    # SQLite stores JSON as TEXT, no conversion needed
    # ========================================
    if is_postgres(conn):
        for table_name, column_name in JSON_COLUMNS:
            op.execute(
                text(
                    f"ALTER TABLE {table_name} "
                    f"ALTER COLUMN {column_name} TYPE jsonb "
                    f"USING {column_name}::jsonb"
                )
            )

    # ========================================
    # MinerInfo Table
    # ========================================

    # Unique partial index for get_or_create_miner
    op.create_index(
        "idx_miner_hotkey_active",
        "miner_info",
        ["hotkey"],
        unique=True,
        postgresql_where=text("deleted_at IS NULL"),
        sqlite_where=text("deleted_at IS NULL"),
    )

    # ========================================
    # WorkerInfo Table
    # ========================================

    # Unique partial index for get_or_create_worker (composite key)
    op.create_index(
        "idx_worker_hotkey_worker_active",
        "worker_info",
        ["hotkey", "worker_id"],
        unique=True,
        postgresql_where=text("deleted_at IS NULL"),
        sqlite_where=text("deleted_at IS NULL"),
    )

    # ========================================
    # HardwareInfo Table
    # ========================================

    # Unique partial index for get_or_create_hardware_info
    op.create_index(
        "idx_hardware_hotkey_worker_active",
        "hardware_info",
        ["hotkey", "worker_id"],
        unique=True,
        postgresql_where=text("deleted_at IS NULL"),
        sqlite_where=text("deleted_at IS NULL"),
    )

    # ========================================
    # GPUInventory Table
    # ========================================

    # Unique partial index for upsert_gpu_inventory
    op.create_index(
        "idx_gpu_uuid_active",
        "gpu_inventory",
        ["gpu_uuid"],
        unique=True,
        postgresql_where=text("deleted_at IS NULL"),
        sqlite_where=text("deleted_at IS NULL"),
    )

    # For get_gpu_inventory_by_worker and get_gpu_inventory_for_workers
    op.create_index(
        "idx_gpu_hotkey_worker_active",
        "gpu_inventory",
        ["hotkey", "worker_id"],
        postgresql_where=text("deleted_at IS NULL"),
        sqlite_where=text("deleted_at IS NULL"),
    )

    # ========================================
    # ComputeChallenge Table
    # ========================================

    # For mark_expired_sent_tasks (status + expires_at)
    op.create_index(
        "idx_challenge_status_expires_active",
        "compute_challenges",
        ["challenge_status", "expires_at"],
        postgresql_where=text("deleted_at IS NULL"),
        sqlite_where=text("deleted_at IS NULL"),
    )

    # For get_workers_with_pending_challenges
    op.create_index(
        "idx_challenge_hotkey_worker_status_active",
        "compute_challenges",
        ["hotkey", "worker_id", "challenge_status"],
        postgresql_where=text("deleted_at IS NULL"),
        sqlite_where=text("deleted_at IS NULL"),
    )

    # ========================================
    # NetworkLog Table
    # ========================================

    # For cleanup_old_data
    op.create_index(
        "idx_netlog_created_active",
        "network_logs",
        ["created_at"],
        postgresql_where=text("deleted_at IS NULL"),
        sqlite_where=text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    """Remove comprehensive performance indexes and revert JSONB to JSON"""
    conn = op.get_bind()

    # MinerInfo
    op.drop_index("idx_miner_hotkey_active", "miner_info")

    # WorkerInfo
    op.drop_index("idx_worker_hotkey_worker_active", "worker_info")

    # HardwareInfo
    op.drop_index("idx_hardware_hotkey_worker_active", "hardware_info")

    # GPUInventory
    op.drop_index("idx_gpu_uuid_active", "gpu_inventory")
    op.drop_index("idx_gpu_hotkey_worker_active", "gpu_inventory")

    # ComputeChallenge
    op.drop_index("idx_challenge_status_expires_active", "compute_challenges")
    op.drop_index("idx_challenge_hotkey_worker_status_active", "compute_challenges")

    # NetworkLog
    op.drop_index("idx_netlog_created_active", "network_logs")

    # ========================================
    # JSONB to JSON reversion (PostgreSQL only)
    # ========================================
    if is_postgres(conn):
        for table_name, column_name in JSON_COLUMNS:
            op.execute(
                text(
                    f"ALTER TABLE {table_name} "
                    f"ALTER COLUMN {column_name} TYPE json "
                    f"USING {column_name}::json"
                )
            )
