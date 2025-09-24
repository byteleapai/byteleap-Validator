"""
Alembic environment configuration
For database migration management
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import models
from neurons.validator.models.database import Base

# Alembic Config object
config = context.config

# Interpret logging configuration file
if config.config_file_name is not None and not os.environ.get(
    "ALEMBIC_SKIP_LOG_CONFIG"
):
    # Do not disable existing loggers to minimize side effects when used via CLI
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Target metadata
target_metadata = Base.metadata


def _read_yaml_database_url(cfg_path: str) -> str:
    """Load database URL from YAML if present, else return empty string.

    Fail-fast is enforced by caller to avoid hidden defaults.
    """
    try:
        import yaml

        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                db = data.get("database") or {}
                url = db.get("url")
                return url or ""
    except Exception:
        # Intentionally swallow and let the caller handle failure
        return ""
    return ""


def get_database_url() -> str:
    """Resolve database URL from YAML configuration only."""
    # Optional explicit config path via env
    cfg_path = os.environ.get("VALIDATOR_CONFIG_PATH")
    if cfg_path:
        url = _read_yaml_database_url(cfg_path)
        if url:
            return url

    # Conventional repo path
    default_cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config/validator_config.yaml"
    )
    url = _read_yaml_database_url(default_cfg_path)
    if url:
        return url

    raise RuntimeError(
        "Database URL not found. Provide database.url in validator_config.yaml."
    )


def run_migrations_offline() -> None:
    """Run migrations in offline mode"""
    url = get_database_url()
    is_sqlite = url.startswith("sqlite")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=is_sqlite,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode"""
    db_url = get_database_url()
    config.set_main_option("sqlalchemy.url", db_url)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=is_sqlite,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
