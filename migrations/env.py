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
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Interpret logging configuration file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata
target_metadata = Base.metadata


# Get database URL from environment variables
def get_database_url():
    """Get database URL from environment variables or configuration"""
    try:
        import yaml

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config/validator_config.yaml"
        )
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                db_url = config.get("database", {}).get("url")
                if db_url:
                    return db_url
    except Exception:
        pass

    # If no environment variable, use default configuration
    return "postgresql://neurons:PrygvMv3U5KjBEKtye7S@localhost:5432/neurons"


def run_migrations_offline() -> None:
    """Run migrations in offline mode"""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode"""
    # Set database URL
    config.set_main_option("sqlalchemy.url", get_database_url())

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
