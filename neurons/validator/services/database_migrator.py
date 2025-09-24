"""
Database schema migration helper.

Ensures the validator's database is migrated to the latest Alembic head
on startup. Uses programmatic Alembic API and the validator configuration
file path so Alembic env.py resolves the URL from YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import bittensor as bt
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy import inspect as sqla_inspect

from neurons.validator.models.database import Base


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _alembic_config(alembic_ini_path: Optional[Path] = None) -> Config:
    """Create Alembic Config bound to this repo's migration directory."""
    if alembic_ini_path is None:
        alembic_ini_path = _project_root() / "alembic.ini"
    cfg = Config(str(alembic_ini_path))
    # Ensure script_location points at our migrations directory
    cfg.set_main_option("script_location", str(_project_root() / "migrations"))
    return cfg


def migrate_to_head_if_needed(
    database_url: str, config_path: Optional[str] = None
) -> bool:
    """
    Upgrade the database to Alembic head if current revision is behind.

    Returns True when schema is at head (either already or after upgrade),
    False if an unrecoverable error occurs.
    """
    if not database_url or not isinstance(database_url, str):
        raise ValueError("database_url required")
    if config_path:
        os.environ["VALIDATOR_CONFIG_PATH"] = config_path

    # Prevent Alembic env.py from reconfiguring root logging when running
    # migrations programmatically during service startup.
    os.environ["ALEMBIC_SKIP_LOG_CONFIG"] = "1"

    cfg = _alembic_config()

    # Determine target (head) revision from migration scripts
    script = ScriptDirectory.from_config(cfg)
    try:
        head_rev = script.get_current_head()
    except Exception:
        # Fall back to generic "head" label if script dir returns none
        head_rev = "head"

    # Inspect current DB revision and existing tables
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            current_rev = ctx.get_current_revision()
            inspector = sqla_inspect(conn)
            existing_tables = inspector.get_table_names()
        engine.dispose()
    except Exception as e:
        bt.logging.error(f"❌ Failed to inspect DB revision | error={e}")
        return False

    safe_db = database_url.split("@")[-1] if "@" in database_url else database_url

    # Bootstrap if no revision and no tables
    try:
        if current_rev is None and not existing_tables:
            bt.logging.info("🧱 Bootstrap DB schema from models (empty database)")
            engine = create_engine(database_url)
            Base.metadata.create_all(bind=engine)
            engine.dispose()
            command.stamp(cfg, "head")
            bt.logging.info("✅ Bootstrapped schema and stamped Alembic head")
            # Refresh current_rev for logging purposes
            try:
                head_rev = ScriptDirectory.from_config(cfg).get_current_head()
            except Exception:
                head_rev = "head"
    except Exception as e:
        bt.logging.error(f"❌ Bootstrap failed | error={e}")
        return False

    # If tables exist but Alembic hasn't been initialized, adopt schema by stamping head
    if current_rev is None and existing_tables:
        try:
            bt.logging.info(
                "📝 Existing schema without version table detected; stamping base revision"
            )
            # Stamp to initial revision to allow subsequent upgrade steps to run
            command.stamp(cfg, "a4e37089d772")
            bt.logging.info("✅ Stamped initial revision; will upgrade to head")
        except Exception as e:
            bt.logging.error(f"❌ Failed to stamp head | error={e}")
            return False
    # Always attempt to upgrade to head; Alembic is idempotent when already current
    try:
        if current_rev != head_rev:
            bt.logging.info(
                f"⛏️ DB migrate required | from={current_rev} -> to={head_rev}"
            )
        bt.logging.info(f"📦 Ensuring DB at head | url={safe_db}")
        command.upgrade(cfg, "head")
        bt.logging.info("✅ DB schema confirmed at head")
        return True
    except Exception as e:
        bt.logging.error(f"❌ Failed to apply DB migrations | error={e}")
        return False
