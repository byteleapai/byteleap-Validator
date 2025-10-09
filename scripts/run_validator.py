#!/usr/bin/env python3
"""
Validator Startup Script
Start and run Bittensor subnet validator
"""
import argparse
import asyncio
import logging
import logging.handlers
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import bittensor as bt
import yaml

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neurons.shared.config.config_manager import ConfigManager
from neurons.validator.core.validator import Validator
from neurons.validator.services.database_migrator import \
    migrate_to_head_if_needed
from neurons.validator.services.meshhub_client import MeshHubClient
from scripts.auto_updater import AutoUpdater, UpdateScheduler


def load_config(config_path: str) -> ConfigManager:
    """Load configuration file and return ConfigManager"""
    bt.logging.debug(f"Load config | path={config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return ConfigManager(config_data)
    except Exception as e:
        bt.logging.error(f"❌ Load config error | error={e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Bittensor Subnet Validator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the validator configuration file (e.g., config/validator_config.yaml)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate configuration without starting service",
    )

    parser.add_argument(
        "--no-auto-update",
        action="store_true",
        help="Skip automatic update check during startup",
    )

    return parser


def validate_config(config: ConfigManager) -> bool:
    """Validate configuration validity using fail-fast config methods"""
    try:
        # Test required fields by accessing them (will raise KeyError if missing)
        config.get("netuid")
        config.get("database.url")
        config.get("port")

        # Validate network ID
        netuid = config.get("netuid")
        if not isinstance(netuid, int) or netuid < 0:
            bt.logging.error("❌ Config error | netuid must be non-negative int")
            return False

        # Validate port
        port = config.get("port")
        if not isinstance(port, int) or port < 1024 or port > 65535:
            bt.logging.error("❌ Config error | port must be in 1024-65535")
            return False

        # Validate database cleanup retention
        try:
            config.get_positive_number("database.event_retention_days", int)
        except (KeyError, ValueError) as e:
            bt.logging.error(
                f"❌ Config error | database.event_retention_days invalid | error={e}"
            )
            return False

        # Validate critical GPU verification parameters
        try:
            config.get("validation.gpu.verification.coordinate_sample_count")
            config.get("validation.gpu.verification.coordinate_sample_count_variance")
            config.get("validation.gpu.verification.row_verification_count")
            config.get("validation.gpu.verification.row_verification_count_variance")
            config.get("validation.gpu.verification.row_sample_rate")
        except KeyError:
            pass  # GPU verification is optional if validation section exists

        # Validate critical CPU verification parameters
        try:
            config.get("validation.cpu.verification.row_verification_count")
            config.get("validation.cpu.verification.row_verification_count_variance")
        except KeyError:
            pass  # CPU verification is optional if validation section exists

        # Validate verification parameters to prevent "zero proof" issue
        try:
            # Check CPU verification
            cpu_rows = config.get("validation.cpu.verification.row_verification_count")
            if cpu_rows <= 0:
                bt.logging.error(
                    f"Error: CPU row_verification_count must be > 0 to avoid hanging challenges. "
                    f"Current value: {cpu_rows}"
                )
                return False

            # Check GPU verification - at least one method must be enabled
            gpu_coords = config.get(
                "validation.gpu.verification.coordinate_sample_count"
            )
            gpu_rows = config.get("validation.gpu.verification.row_verification_count")
            if gpu_coords <= 0 and gpu_rows <= 0:
                bt.logging.error(
                    f"Error: At least one GPU verification method must be enabled. "
                    f"coordinate_sample_count={gpu_coords}, row_verification_count={gpu_rows}"
                )
                return False

        except KeyError as e:
            bt.logging.warning(f"⚠️ Verify params | skip_check err={e}")

        # MeshHub required configuration
        try:
            MeshHubClient.validate_config(config)
        except (KeyError, ValueError) as e:
            bt.logging.error(f"❌ MeshHub config invalid | error={e}")
            return False

        return True

    except KeyError as e:
        bt.logging.error(f"❌ Config validation failed | error={e}")
        return False


def setup_logging(config: ConfigManager) -> None:
    """Setup complete logging configuration for validator"""
    log_dir = config.get("logging.log_dir")
    log_level = config.get("logging.log_level").upper()

    # Set bittensor logging level
    if log_level == "DEBUG":
        bt.logging.enable_debug()
    elif log_level == "INFO":
        bt.logging.enable_info()
    elif log_level == "WARNING":
        bt.logging.enable_warning()
    else:
        bt.logging.enable_default()

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Get the root logger used by bittensor
    root_logger = logging.getLogger()

    # Create rotating file handler (daily rotation, keep 3 days)
    log_filename = "validator.log"
    log_filepath = Path(log_dir) / log_filename

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_filepath,
        when="midnight",
        interval=1,
        backupCount=3,
        encoding="utf-8",
        utc=False,
    )

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # Set log level
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    file_handler.setLevel(log_level_map.get(log_level, logging.INFO))

    # Add handler to root logger
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)  # Allow all levels, handlers will filter

    # Set websockets logging to WARNING to suppress ping/pong debug messages
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.WARNING)

    # Suppress verbose SQLAlchemy internals unless explicitly needed
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)
    # Alembic migration logs at INFO are usually sufficient
    logging.getLogger("alembic").setLevel(logging.INFO)

    bt.logging.info(f"🧾 File logging | path={log_filepath} | level={log_level}")


def setup_environment(config: ConfigManager) -> None:
    """Set up runtime environment"""
    # Setup complete logging configuration
    setup_logging(config)


def check_database_connection(database_url: str) -> bool:
    """Check database connection"""
    try:
        from sqlalchemy import text

        from neurons.validator.models.database import DatabaseManager

        db_manager = DatabaseManager(database_url)

        # Try to connect
        with db_manager.get_session() as session:
            session.execute(text("SELECT 1"))

        bt.logging.info("✅ Database connection successful")
        return True

    except Exception as e:
        bt.logging.error(f"❌ Database connection failed | error={e}")
        return False


async def perform_startup_update_check() -> bool:
    """
    Perform automatic update check during validator startup

    Returns:
        bool: True if an update was performed and restart is needed, False otherwise
    """
    try:
        bt.logging.info("🔍 Performing startup update check")

        # Check if we just restarted due to an update (prevent infinite loops)
        restart_marker_file = project_root / ".update_restart_marker"
        if restart_marker_file.exists():
            bt.logging.info("✅ Validator restarted successfully after update")
            # Remove the marker file
            restart_marker_file.unlink()
            return False

        # Create auto-updater instance
        updater = AutoUpdater(str(project_root))

        # Get current version before update check
        current_version = updater.get_current_version()

        # Perform update check
        update_performed = await updater.check_and_update()

        if update_performed:
            bt.logging.info("🎉 Update completed successfully!")
            bt.logging.info("📝 IMPORTANT: Restarting to use new code")

            # Create marker file to indicate we're restarting due to update
            restart_marker_file.write_text(
                f"Updated from {current_version} at {datetime.now().isoformat()}"
            )

            updater.cleanup_old_backups()
            return True
        else:
            bt.logging.info("✅ No updates needed - already up to date")
            updater.cleanup_old_backups()
            return False

    except Exception as e:
        bt.logging.warning(
            f"⚠️ Startup update check failed (continuing with current version): {e}"
        )
        return False


def restart_validator_process():
    """
    Restart the validator process using subprocess in the current terminal
    This function will exec a new process replacing the current one
    """
    try:
        bt.logging.info("🔄 Restarting validator process with updated code")

        # Get current command line arguments
        current_args = sys.argv.copy()

        # Construct the restart command
        python_executable = sys.executable
        script_path = os.path.abspath(current_args[0])
        script_args = current_args[1:]

        restart_command = [python_executable, script_path] + script_args

        bt.logging.info(
            f"🚀 Restarting in current terminal: {' '.join(restart_command)}"
        )
        bt.logging.info(f"📁 Working directory: {project_root}")
        bt.logging.info(f"🐍 Python executable: {python_executable}")

        # Flush all logs before restart
        import logging

        for handler in logging.getLogger().handlers:
            handler.flush()

        # Small delay to ensure logs are written
        import time

        time.sleep(1)

        # Use os.execv to replace current process (Unix/Linux style restart)
        if hasattr(os, "execv"):
            bt.logging.info("🔄 Using os.execv to restart process")
            # Change to project directory
            os.chdir(str(project_root))
            # Replace current process with new one
            os.execv(python_executable, restart_command)
        else:
            # Fallback for Windows or systems without execv
            bt.logging.info("🔄 Using subprocess.call for restart")
            os.chdir(str(project_root))
            # Exit current process and start new one
            exit_code = subprocess.call(restart_command)
            sys.exit(exit_code)

    except Exception as e:
        bt.logging.error(f"❌ Failed to restart validator process: {e}")
        bt.logging.error("📝 Please restart manually to use updated code")
        # Continue with current process if restart fails
        bt.logging.warning(
            "⚠️ Continuing with current process (updated code will be used on next manual restart)"
        )
        return False


async def main():
    """Main function"""
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Initialize bt.logging with colors
    bt.logging.enable_default()

    # Validate configuration file existence
    if not args.config or not os.path.exists(args.config):
        bt.logging.error(f"❌ Config file not found | path={args.config}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Validate configuration
    if not validate_config(config):
        sys.exit(1)

    # Setup environment (including complete logging configuration)
    setup_environment(config)

    # Display configuration information
    bt.logging.info(f"🌐 Netuid | id={config.get('netuid')}")
    bt.logging.info(f"🔌 Port | port={config.get('port')}")
    bt.logging.info(f"👛 Wallet | name={config.get('wallet.name')}")
    bt.logging.info(f"🔑 Hotkey | name={config.get('wallet.hotkey')}")
    database_url = config.get("database.url")
    safe_db_url = database_url.split("@")[-1] if "@" in database_url else database_url
    bt.logging.info(f"DB | url={safe_db_url}")
    bt.logging.debug(f"Log level | level={config.get('logging.log_level')}")

    # Check database connection
    bt.logging.info("🔎 Checking database connection")
    if not check_database_connection(config.get("database.url")):
        bt.logging.error(
            "❌ DB connect failed | ensure PostgreSQL is running and configured"
        )
        sys.exit(1)

    # Dry run mode
    if args.dry_run:
        bt.logging.info("✅ Config validation done (dry run)")
        return

    # Perform startup update check (unless disabled)
    update_check_enabled = config.get_optional(
        "auto_update.enabled", True
    )  # Default to enabled
    if not args.no_auto_update and update_check_enabled:
        bt.logging.info("🚀 Starting automatic update check")
        update_performed = await perform_startup_update_check()

        if update_performed:
            bt.logging.info("🔄 Update completed - restarting validator with new code")
            # Restart the current process via Python native exec (no external manager required)
            restart_validator_process()
            # This line should never be reached due to os.execv in restart_validator_process()
            return
    else:
        if args.no_auto_update:
            bt.logging.info("⏭️ Automatic updates disabled via command line")
        else:
            bt.logging.info("⏭️ Automatic updates disabled in configuration")

    # Ensure database schema is up-to-date before service starts
    bt.logging.info("🛠️ Ensuring database schema is current")
    ok = migrate_to_head_if_needed(database_url, config_path=args.config)
    if not ok:
        bt.logging.error("❌ Database migration failed; aborting startup")
        sys.exit(1)

    validator = None
    try:
        # Create unified Bittensor config to avoid multiple internal loads
        bt_config = bt.config()
        # Populate required fields from our ConfigManager
        bt_config.netuid = config.get("netuid")
        from munch import DefaultMunch

        bt_config.wallet = DefaultMunch()
        bt_config.wallet.name = config.get("wallet.name")
        bt_config.wallet.hotkey = config.get("wallet.hotkey")
        bt_config.wallet.path = config.get("wallet.path")
        bt_config.subtensor = DefaultMunch()
        bt_config.subtensor.network = config.get("subtensor.network")

        # Create and start validator with shared bt_config
        validator = Validator(config, bt_config)
        await validator.run()

    except KeyboardInterrupt:
        bt.logging.info("⏹️ Interrupt | shutting down")
        if validator and validator.is_running:
            try:
                await asyncio.wait_for(validator.stop(), timeout=10.0)
            except asyncio.TimeoutError:
                bt.logging.warning("Shutdown timeout, forcing exit")
                os._exit(1)
    except Exception as e:
        bt.logging.error(f"❌ Runtime error | error={e}")
        if validator and validator.is_running:
            try:
                await validator.stop()
            except:
                pass
        sys.exit(1)
    finally:
        bt.logging.info("✅ Validator stopped")


if __name__ == "__main__":
    # Ensure predictable multiprocessing behavior across platforms
    try:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # Run main program
    asyncio.run(main())
