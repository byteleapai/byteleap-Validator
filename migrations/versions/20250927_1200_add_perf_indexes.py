"""add performance and cleanup indexes

Revision ID: 3b7f9d12c8ab
Revises: 6a8c3a21b2f1
Create Date: 2025-09-27 12:00:00+00:00

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "3b7f9d12c8ab"
down_revision = "6a8c3a21b2f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name if bind is not None else ""

    # --- ComputeChallenge ---
    # 1) Status + time ordering for verification queue (partial on active rows)
    try:
        op.create_index(
            "idx_chal_status_comp_created_active",
            "compute_challenges",
            ["challenge_status", "computed_at", "created_at"],
            unique=False,
            postgresql_where=sa.text("deleted_at IS NULL"),
            sqlite_where=sa.text("deleted_at IS NULL"),
        )
    except Exception:
        pass

    # 2) Verified flag + created_at for availability calculations (partial)
    try:
        op.create_index(
            "idx_chal_verified_created_active",
            "compute_challenges",
            ["verification_result", "created_at"],
            unique=False,
            postgresql_where=sa.text("deleted_at IS NULL"),
            sqlite_where=sa.text("deleted_at IS NULL"),
        )
    except Exception:
        pass

    # --- HeartbeatRecord ---
    # 1) Latest heartbeat per (hotkey, worker) with time ordering (partial)
    try:
        op.create_index(
            "idx_hb_hotkey_worker_created_active",
            "heartbeat_records",
            ["hotkey", "worker_id", "created_at"],
            unique=False,
            postgresql_where=sa.text("deleted_at IS NULL"),
            sqlite_where=sa.text("deleted_at IS NULL"),
        )
    except Exception:
        pass

    # 2) IP change detection scan optimization (partial)
    try:
        if dialect == "postgresql":
            op.create_index(
                "idx_hb_worker_created_ip_active",
                "heartbeat_records",
                ["worker_id", "created_at"],
                unique=False,
                postgresql_include=["public_ip"],
                postgresql_where=sa.text("deleted_at IS NULL"),
            )
        else:
            # Fallback for SQLite and others: include public_ip as keyed column
            op.create_index(
                "idx_hb_worker_created_ip_active",
                "heartbeat_records",
                ["worker_id", "created_at", "public_ip"],
                unique=False,
                sqlite_where=sa.text("deleted_at IS NULL"),
            )
    except Exception:
        pass

    # --- WorkerInfo ---
    # Lease ranking per miner (partial)
    try:
        op.create_index(
            "idx_worker_hotkey_lease_active",
            "worker_info",
            ["hotkey", "lease_score"],
            unique=False,
            postgresql_where=sa.text("deleted_at IS NULL"),
            sqlite_where=sa.text("deleted_at IS NULL"),
        )
    except Exception:
        pass

    # --- GPUInventory ---
    # Recent active GPUs (partial)
    try:
        op.create_index(
            "idx_gpu_last_seen_active",
            "gpu_inventory",
            ["last_seen_at"],
            unique=False,
            postgresql_where=sa.text("deleted_at IS NULL"),
            sqlite_where=sa.text("deleted_at IS NULL"),
        )
    except Exception:
        pass

    # --- NetworkWeight ---
    # Cleanup by created_at (partial)
    try:
        op.create_index(
            "idx_weight_created_active",
            "network_weights",
            ["created_at"],
            unique=False,
            postgresql_where=sa.text("deleted_at IS NULL"),
            sqlite_where=sa.text("deleted_at IS NULL"),
        )
    except Exception:
        pass

    # --- MeshHubTask ---
    # Cleanup by created_at (partial)
    try:
        op.create_index(
            "idx_mesh_task_created_active",
            "meshhub_tasks",
            ["created_at"],
            unique=False,
            postgresql_where=sa.text("deleted_at IS NULL"),
            sqlite_where=sa.text("deleted_at IS NULL"),
        )
    except Exception:
        pass


def downgrade() -> None:
    # Drop in reverse order, ignore errors if missing
    drops = [
        ("idx_mesh_task_created_active", "meshhub_tasks"),
        ("idx_weight_created_active", "network_weights"),
        ("idx_gpu_last_seen_active", "gpu_inventory"),
        ("idx_worker_hotkey_lease_active", "worker_info"),
        ("idx_hb_worker_created_ip_active", "heartbeat_records"),
        ("idx_hb_hotkey_worker_created_active", "heartbeat_records"),
        ("idx_chal_verified_created_active", "compute_challenges"),
        ("idx_chal_status_comp_created_active", "compute_challenges"),
    ]
    for name, table in drops:
        try:
            op.drop_index(name, table_name=table)
        except Exception:
            pass
