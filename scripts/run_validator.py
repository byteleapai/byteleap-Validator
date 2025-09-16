#!/usr/bin/env python3
"""
Validator Startup Script
Start and run Bittensor subnet validator
"""
import argparse
import asyncio
import logging
import os
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


def load_config(config_path: str) -> ConfigManager:
    """Load configuration file and return ConfigManager"""
    bt.logging.debug(f"🧾 Load config | path={config_path}")
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

    # Create log filename with current date
    current_date = datetime.now().strftime("%Y%m%d")
    log_filename = f"validator_{current_date}.log"
    log_filepath = Path(log_dir) / log_filename

    # Get the root logger used by bittensor
    root_logger = logging.getLogger()

    # Create file handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")

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

    bt.logging.debug(f"🧾 File logging | path={log_filepath}")


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
    bt.logging.info(f"🌐 Net | id={config.get('netuid')}")
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
    # Unix/Linux event loop policy (default)

    # Run main program
    asyncio.run(main())
