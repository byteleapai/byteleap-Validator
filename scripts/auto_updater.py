#!/usr/bin/env python3
"""
Auto-Updater for Subnet Miner
Checks for new releases on GitHub and automatically updates the codebase.
Preserves the config directory; restart behavior is up to the caller.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import aiofiles
import aiohttp


class AutoUpdater:
    """Handles automatic updates for the subnet miner (no process manager dependency)."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.version_file = self.project_root / "version.txt"
        self.config_dir = self.project_root / "config"
        self.backup_dir = self.project_root / "backups"
        self.log_dir = self.project_root / "logs" / "updater"

        # GitHub repository settings
        self.repo_owner = "byteleapai"
        self.repo_name = "byteleap-Validator"
        self.github_api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/releases/latest"

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the auto-updater"""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger("auto_updater")
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler with rotation
        log_file = self.log_dir / f"updater_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("🔧 Auto-updater logging initialized")

    def get_current_version(self) -> str:
        """Get the current version from version.txt"""
        try:
            if self.version_file.exists():
                version = self.version_file.read_text().strip()
                self.logger.debug(f"📖 Current version: {version}")
                return version
            else:
                self.logger.warning("⚠️ Version file not found, assuming v0.0.0")
                return "v0.0.0"
        except Exception as e:
            self.logger.error(f"❌ Failed to read version file: {e}")
            return "v0.0.0"

    def update_version_file(self, version: str) -> None:
        """Update the version.txt file with new version"""
        try:
            self.version_file.write_text(version)
            self.logger.info(f"✅ Version file updated to: {version}")
        except Exception as e:
            self.logger.error(f"❌ Failed to update version file: {e}")
            raise

    async def fetch_latest_release(self) -> Optional[Dict]:
        """Fetch latest release information from GitHub API"""
        try:
            self.logger.debug(f"🔍 Checking for latest release: {self.github_api_url}")

            async with aiohttp.ClientSession() as session:
                async with session.get(self.github_api_url) as response:
                    if response.status == 200:
                        release_data = await response.json()
                        self.logger.debug(
                            f"📦 Latest release: {release_data.get('tag_name')}"
                        )
                        return release_data
                    elif response.status == 404:
                        self.logger.warning("⚠️ Repository or releases not found")
                        return None
                    else:
                        self.logger.error(f"❌ GitHub API error: {response.status}")
                        return None

        except Exception as e:
            self.logger.error(f"❌ Failed to fetch release info: {e}")
            return None

    def compare_versions(self, current: str, latest: str) -> bool:
        """Compare version strings to determine if update is needed"""
        try:
            # Normalize version strings - remove 'release-v' or 'v' prefixes
            current_clean = current.replace("release-v", "").lstrip("v")
            latest_clean = latest.replace("release-v", "").lstrip("v")

            self.logger.debug(
                f"🔍 Version comparison: '{current}' -> '{current_clean}' vs '{latest}' -> '{latest_clean}'"
            )

            # Split version numbers
            current_parts = list(map(int, current_clean.split(".")))
            latest_parts = list(map(int, latest_clean.split(".")))

            # Pad shorter version with zeros
            max_length = max(len(current_parts), len(latest_parts))
            current_parts.extend([0] * (max_length - len(current_parts)))
            latest_parts.extend([0] * (max_length - len(latest_parts)))

            self.logger.debug(f"🔍 Comparing parts: {current_parts} vs {latest_parts}")

            # Compare versions
            for current_part, latest_part in zip(current_parts, latest_parts):
                if latest_part > current_part:
                    self.logger.debug(
                        f"✅ Update needed: {latest_part} > {current_part}"
                    )
                    return True
                elif latest_part < current_part:
                    self.logger.debug(
                        f"❌ Current version is newer: {current_part} > {latest_part}"
                    )
                    return False

            self.logger.debug("ℹ️ Versions are equal")
            return False  # Versions are equal

        except Exception as e:
            self.logger.error(f"❌ Version comparison failed: {e}")
            return False

    async def download_release(self, download_url: str, destination: Path) -> bool:
        """Download release zip file"""
        try:
            self.logger.info(f"⬇️ Downloading release from: {download_url}")

            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as response:
                    if response.status == 200:
                        total_size = int(response.headers.get("content-length", 0))
                        downloaded = 0

                        async with aiofiles.open(destination, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                                downloaded += len(chunk)

                                # Log progress every 1MB
                                if downloaded % (1024 * 1024) == 0:
                                    if total_size > 0:
                                        progress = (downloaded / total_size) * 100
                                        self.logger.debug(
                                            f"📥 Download progress: {progress:.1f}%"
                                        )

                        self.logger.info(f"✅ Download completed: {destination}")
                        return True
                    else:
                        self.logger.error(f"❌ Download failed: HTTP {response.status}")
                        return False

        except Exception as e:
            self.logger.error(f"❌ Download error: {e}")
            return False

    def create_backup(self) -> Optional[Path]:
        """Create backup of current installation"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            backup_path = self.backup_dir / backup_name

            self.logger.info(f"📦 Creating backup: {backup_path}")

            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)

            # Copy important files/directories (excluding logs, backups, temp files)
            exclude_patterns = {
                "logs",
                "backups",
                "__pycache__",
                ".git",
                "node_modules",
                "*.pyc",
                "*.log",
                ".DS_Store",
                "Thumbs.db",
            }

            for item in self.project_root.iterdir():
                if item.name not in exclude_patterns and not item.name.startswith("."):
                    if item.is_file():
                        shutil.copy2(item, backup_path / item.name)
                    elif item.is_dir():
                        shutil.copytree(
                            item,
                            backup_path / item.name,
                            ignore=shutil.ignore_patterns(*exclude_patterns),
                        )

            self.logger.info(f"✅ Backup created successfully: {backup_path}")
            return backup_path

        except Exception as e:
            self.logger.error(f"❌ Backup creation failed: {e}")
            return None

    def preserve_config(self) -> Optional[Path]:
        """Preserve config directory"""
        try:
            if not self.config_dir.exists():
                self.logger.warning("⚠️ Config directory not found, nothing to preserve")
                return None

            temp_config = Path(tempfile.mkdtemp()) / "config_backup"
            shutil.copytree(self.config_dir, temp_config)

            self.logger.info(f"💾 Config preserved: {temp_config}")
            return temp_config

        except Exception as e:
            self.logger.error(f"❌ Config preservation failed: {e}")
            return None

    def restore_config(self, preserved_config: Path) -> bool:
        """Restore preserved config directory"""
        try:
            if not preserved_config.exists():
                self.logger.warning("⚠️ Preserved config not found")
                return False

            # Remove current config if exists
            if self.config_dir.exists():
                shutil.rmtree(self.config_dir)

            # Restore preserved config
            shutil.copytree(preserved_config, self.config_dir)

            # Cleanup temporary config
            shutil.rmtree(preserved_config.parent)

            self.logger.info(f"✅ Config restored successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Config restoration failed: {e}")
            return False

    def extract_update(self, zip_path: Path) -> bool:
        """Extract update zip file"""
        try:
            self.logger.info(f"📂 Extracting update: {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Create temporary extraction directory
                temp_extract = Path(tempfile.mkdtemp())
                zip_ref.extractall(temp_extract)

                # Find the extracted directory (usually has repo name as prefix)
                extracted_dirs = [d for d in temp_extract.iterdir() if d.is_dir()]
                if not extracted_dirs:
                    self.logger.error("❌ No directories found in extracted zip")
                    return False

                source_dir = extracted_dirs[0]

                # Copy files to project root (excluding config directory)
                for item in source_dir.iterdir():
                    if item.name == "config":
                        self.logger.info(f"⏭️ Skipping config directory: {item.name}")
                        continue

                    dest_path = self.project_root / item.name

                    # Remove existing item if it exists
                    if dest_path.exists():
                        if dest_path.is_file():
                            dest_path.unlink()
                        else:
                            shutil.rmtree(dest_path)

                    # Copy new item
                    if item.is_file():
                        shutil.copy2(item, dest_path)
                    else:
                        shutil.copytree(item, dest_path)

                # Cleanup temporary directory
                shutil.rmtree(temp_extract)

            self.logger.info("✅ Update extraction completed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Update extraction failed: {e}")
            return False

    def cleanup_temp_files(self, *paths: Path) -> None:
        """Cleanup temporary files and directories"""
        for path in paths:
            try:
                if path and path.exists():
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path)
                    self.logger.debug(f"🧹 Cleaned up: {path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Cleanup failed for {path}: {e}")

    async def check_and_update(self) -> bool:
        """Main update check and execution logic (caller controls restart)."""
        try:
            self.logger.info("🔍 Starting update check")

            # Get current version
            current_version = self.get_current_version()
            self.logger.info(f"📖 Current version: {current_version}")

            # Fetch latest release
            release_data = await self.fetch_latest_release()
            if not release_data:
                self.logger.info("ℹ️ No release information available")
                return False

            latest_version = release_data.get("tag_name", "")
            self.logger.info(f"📦 Latest version: {latest_version}")

            # Compare versions
            if not self.compare_versions(current_version, latest_version):
                self.logger.info("✅ Already up to date")
                return False

            self.logger.info(f"🆕 New version available: {latest_version}")

            # Find download URL (prefer zipball_url)
            download_url = release_data.get("zipball_url")
            if not download_url:
                self.logger.error("❌ No download URL found in release")
                return False

            # Start update process
            self.logger.info("🚀 Starting update process")

            # Create backup
            backup_path = self.create_backup()
            if not backup_path:
                self.logger.error("❌ Backup creation failed, aborting update")
                return False

            # Preserve config
            preserved_config = self.preserve_config()

            # Download new version
            temp_zip = Path(tempfile.mktemp(suffix=".zip"))
            download_success = await self.download_release(download_url, temp_zip)

            if not download_success:
                self.logger.error("❌ Download failed, aborting update")
                self.cleanup_temp_files(temp_zip)
                return False

            # Extract update
            extract_success = self.extract_update(temp_zip)
            if not extract_success:
                self.logger.error("❌ Extraction failed, aborting update")
                self.cleanup_temp_files(temp_zip)
                return False

            # Restore config
            if preserved_config:
                self.restore_config(preserved_config)

            # Update version file
            self.update_version_file(latest_version)

            # Cleanup
            self.cleanup_temp_files(temp_zip)

            # Caller decides whether to restart (e.g., run_validator.py auto-restarts)
            self.logger.info(
                f"🎉 Update completed successfully to version {latest_version}"
            )
            self.logger.info(
                "📝 If the caller does not auto-restart, please restart the process to apply the new code"
            )

            return True

        except Exception as e:
            self.logger.error(f"❌ Update process failed: {e}")
            return False

    def cleanup_old_backups(self, keep_count: int = 5) -> None:
        """Cleanup old backup directories, keeping only the most recent ones"""
        try:
            if not self.backup_dir.exists():
                return

            # Get all backup directories sorted by creation time
            backups = [
                d
                for d in self.backup_dir.iterdir()
                if d.is_dir() and d.name.startswith("backup_")
            ]
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove old backups
            for backup in backups[keep_count:]:
                shutil.rmtree(backup)
                self.logger.debug(f"🧹 Removed old backup: {backup.name}")

            if len(backups) > keep_count:
                self.logger.info(
                    f"🧹 Cleaned up {len(backups) - keep_count} old backups"
                )

        except Exception as e:
            self.logger.warning(f"⚠️ Backup cleanup failed: {e}")


class UpdateScheduler:
    """Handles scheduling of update checks within the validator process"""

    def __init__(self, updater: AutoUpdater):
        self.updater = updater
        self.logger = updater.logger
        self.is_running = False
        self.last_check_time = 0
        self.check_interval = 12 * 60 * 60  # 12 hours in seconds

    async def scheduled_update_check(self) -> None:
        """Wrapper for scheduled update checks"""
        try:
            self.logger.info("⏰ Scheduled update check started")
            update_performed = await self.updater.check_and_update()
            if update_performed:
                self.logger.info(
                    "🎉 Update completed - restart the process to apply changes (caller-defined)"
                )
            self.updater.cleanup_old_backups()
            self.last_check_time = time.time()
        except Exception as e:
            self.logger.error(f"❌ Scheduled update check failed: {e}")

    async def run_initial_check(self) -> None:
        """Run initial update check on startup"""
        try:
            self.logger.info("🔍 Running initial update check on startup")
            update_performed = await self.updater.check_and_update()
            if update_performed:
                self.logger.info("🎉 Initial update completed")
                self.logger.info(
                    "📝 NOTICE: Caller should restart process to apply new code"
                )
            self.updater.cleanup_old_backups()
            self.last_check_time = time.time()
        except Exception as e:
            self.logger.error(f"❌ Initial update check failed: {e}")

    def start_background_scheduler(self) -> None:
        """Start the background scheduler thread"""

        def scheduler_loop():
            self.is_running = True
            self.logger.info(
                "⏰ Background update scheduler started - checking every 12 hours"
            )

            while self.is_running:
                try:
                    current_time = time.time()
                    if current_time - self.last_check_time >= self.check_interval:
                        # Run the update check in the background
                        asyncio.run(self.scheduled_update_check())
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    self.logger.error(f"❌ Background scheduler error: {e}")
                    time.sleep(300)  # Wait before retrying

        scheduler_thread = threading.Thread(
            target=scheduler_loop, daemon=True, name="UpdateScheduler"
        )
        scheduler_thread.start()

    def stop(self) -> None:
        """Stop the scheduler"""
        self.is_running = False


async def main():
    """Main function for testing and manual execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Updater for Subnet Miner")
    parser.add_argument(
        "--check", action="store_true", help="Check for updates once and exit"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        help="Project root directory (default: parent of script)",
    )

    args = parser.parse_args()

    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).parent.parent

    # Create updater
    updater = AutoUpdater(str(project_root))

    # Single update check
    await updater.check_and_update()
    updater.cleanup_old_backups()


if __name__ == "__main__":
    asyncio.run(main())
