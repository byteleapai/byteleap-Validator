#!/usr/bin/env python3
"""
Database Migration Management Script
Manage PostgreSQL database versions and migrations
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import bittensor as bt
import psycopg2
import yaml
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_database_config(config_path: str = None) -> dict:
    """Load database configuration"""
    if config_path is None:
        config_path = project_root / "config/validator_config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("database", {})
    except Exception as e:
        bt.logging.error(f"Failed to load configuration file: {e}")
        return {}


def parse_database_url(database_url: str) -> dict:
    """Parse database URL"""
    # Format: postgresql://user:password@host:port/database
    if not database_url.startswith("postgresql://"):
        raise ValueError("Database URL format error")

    url_parts = database_url.replace("postgresql://", "").split("/")
    database = url_parts[1] if len(url_parts) > 1 else "postgres"

    user_host = url_parts[0].split("@")
    host_port = user_host[1] if len(user_host) > 1 else "localhost:5432"
    user_pass = user_host[0] if len(user_host) > 1 else "postgres:"

    user, password = user_pass.split(":") if ":" in user_pass else (user_pass, "")
    host, port = host_port.split(":") if ":" in host_port else (host_port, "5432")

    return {
        "host": host,
        "port": int(port),
        "user": user,
        "password": password,
        "database": database,
    }


def check_postgresql_connection(db_config: dict) -> bool:
    """Check PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database="postgres",  # Connect to default database
        )
        conn.close()
        bt.logging.info("✓ PostgreSQL connection successful")
        return True
    except Exception as e:
        bt.logging.error(f"✗ PostgreSQL connection failed: {e}")
        return False


def database_exists(db_config: dict) -> bool:
    """Check if database exists"""
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (db_config["database"],)
        )
        exists = cursor.fetchone() is not None

        cursor.close()
        conn.close()

        return exists
    except Exception as e:
        bt.logging.error(f"Failed to check database existence: {e}")
        return False


def create_database(db_config: dict) -> bool:
    """Create database"""
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()
        cursor.execute(f'CREATE DATABASE "{db_config["database"]}"')

        cursor.close()
        conn.close()

        bt.logging.info(f"✓ Database '{db_config['database']}' created successfully")
        return True
    except Exception as e:
        bt.logging.error(f"✗ Failed to create database: {e}")
        return False


def drop_database(db_config: dict, force: bool = False) -> bool:
    """Delete database"""
    if not force:
        response = input(
            f"Are you sure you want to delete database '{db_config['database']}'? [y/N]: "
        )
        if response.lower() not in ["y", "yes"]:
            bt.logging.info("Delete operation cancelled")
            return False

    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()
        # Disconnect all connections
        cursor.execute(
            f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{db_config["database"]}'
            AND pid <> pg_backend_pid()
        """
        )

        cursor.execute(f'DROP DATABASE IF EXISTS "{db_config["database"]}"')

        cursor.close()
        conn.close()

        bt.logging.info(f"✓ Database '{db_config['database']}' deleted successfully")
        return True
    except Exception as e:
        bt.logging.error(f"✗ Failed to delete database: {e}")
        return False


def run_alembic_command(command: list, database_url: str) -> bool:
    """Run Alembic command"""
    try:
        env = os.environ.copy()
        env["DATABASE_URL"] = database_url

        os.chdir(project_root)
        result = subprocess.run(
            ["alembic"] + command, capture_output=True, text=True, env=env
        )

        if result.returncode == 0:
            bt.logging.info("✓ Alembic command executed successfully")
            if result.stdout:
                bt.logging.info(result.stdout)
            return True
        else:
            bt.logging.error("✗ Alembic command execution failed")
            if result.stderr:
                bt.logging.error(result.stderr)
            return False
    except Exception as e:
        bt.logging.error(f"Failed to execute Alembic command: {e}")
        return False


def init_migrations(database_url: str) -> bool:
    """Initialize migrations"""
    bt.logging.info("Initializing database migrations...")

    # Check if already initialized
    migrations_dir = project_root / "migrations/versions"
    if migrations_dir.exists() and list(migrations_dir.glob("*.py")):
        bt.logging.info(
            "Migrations already initialized, skipping initial migration creation"
        )
        return run_alembic_command(["upgrade", "head"], database_url)

    # Create initial migration
    success = run_alembic_command(
        ["revision", "--autogenerate", "-m", "Initial migration"], database_url
    )
    if success:
        return run_alembic_command(["upgrade", "head"], database_url)
    return False


def create_migration(message: str, database_url: str) -> bool:
    """Create new migration"""
    bt.logging.info(f"Creating migration: {message}")
    return run_alembic_command(
        ["revision", "--autogenerate", "-m", message], database_url
    )


def upgrade_database(database_url: str, revision: str = "head") -> bool:
    """Upgrade database"""
    bt.logging.info(f"Upgrading database to version: {revision}")
    return run_alembic_command(["upgrade", revision], database_url)


def downgrade_database(database_url: str, revision: str) -> bool:
    """Downgrade database"""
    bt.logging.info(f"Downgrading database to version: {revision}")
    return run_alembic_command(["downgrade", revision], database_url)


def show_migration_history(database_url: str) -> bool:
    """Show migration history"""
    bt.logging.info("Migration history:")
    return run_alembic_command(["history"], database_url)


def show_current_revision(database_url: str) -> bool:
    """Show current version"""
    bt.logging.info("Current database version:")
    return run_alembic_command(["current"], database_url)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Database migration management tool")

    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--database-url", type=str, help="Database connection URL")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    subparsers.add_parser("check", help="Check database connection")

    # Create database
    subparsers.add_parser("create-db", help="Create database")

    # Drop database
    drop_parser = subparsers.add_parser("drop-db", help="Drop database")
    drop_parser.add_argument(
        "--force", action="store_true", help="Force delete without confirmation"
    )

    # Initialize migrations
    subparsers.add_parser("init", help="Initialize migrations")

    # Create migration
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("message", help="Migration description message")

    # Upgrade database
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade database")
    upgrade_parser.add_argument(
        "revision", nargs="?", default="head", help="Target version"
    )

    # Downgrade database
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade database")
    downgrade_parser.add_argument("revision", help="Target version")

    # View history
    subparsers.add_parser("history", help="View migration history")

    # View current version
    subparsers.add_parser("current", help="View current database version")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Get database configuration
    if args.database_url:
        database_url = args.database_url
    else:
        db_config_dict = load_database_config(args.config)
        database_url = db_config_dict.get(
            "url", "postgresql://neurons:PrygvMv3U5KjBEKtye7S@localhost:5432/neurons"
        )

    bt.logging.info(
        f"Using database URL: {database_url.split('@')[-1] if '@' in database_url else database_url}"
    )

    # Parse database configuration
    db_config = parse_database_url(database_url)

    # Execute command
    if args.command == "check":
        if not check_postgresql_connection(db_config):
            sys.exit(1)

        if database_exists(db_config):
            bt.logging.info(f"✓ Database '{db_config['database']}' exists")
        else:
            bt.logging.warning(f"✗ Database '{db_config['database']}' does not exist")

    elif args.command == "create-db":
        if not check_postgresql_connection(db_config):
            sys.exit(1)

        if database_exists(db_config):
            bt.logging.info(f"Database '{db_config['database']}' already exists")
        else:
            if not create_database(db_config):
                sys.exit(1)

    elif args.command == "drop-db":
        if not check_postgresql_connection(db_config):
            sys.exit(1)

        if database_exists(db_config):
            if not drop_database(db_config, args.force):
                sys.exit(1)
        else:
            bt.logging.warning(f"Database '{db_config['database']}' does not exist")

    elif args.command == "init":
        if not database_exists(db_config):
            if not create_database(db_config):
                sys.exit(1)

        if not init_migrations(database_url):
            sys.exit(1)

    elif args.command == "create":
        if not create_migration(args.message, database_url):
            sys.exit(1)

    elif args.command == "upgrade":
        if not upgrade_database(database_url, args.revision):
            sys.exit(1)

    elif args.command == "downgrade":
        if not downgrade_database(database_url, args.revision):
            sys.exit(1)

    elif args.command == "history":
        if not show_migration_history(database_url):
            sys.exit(1)

    elif args.command == "current":
        if not show_current_revision(database_url):
            sys.exit(1)


if __name__ == "__main__":
    main()
