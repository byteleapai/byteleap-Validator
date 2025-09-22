#!/usr/bin/env python3
"""
PM2 Management Script for Subnet Validator
Provides convenient commands for managing the validator with PM2
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


class PM2Manager:
    """PM2 process manager for subnet validator"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.pm2_config = self.project_root / "pm2.config.js"
        self.app_name = "subnet-validator"

    def run_command(
        self, cmd: List[str], capture_output: bool = False
    ) -> subprocess.CompletedProcess:
        """Execute a command and return the result"""
        try:
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            else:
                result = subprocess.run(cmd, check=True)
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {' '.join(cmd)}")
            if capture_output and e.stdout:
                print(f"stdout: {e.stdout}")
            if capture_output and e.stderr:
                print(f"stderr: {e.stderr}")
            sys.exit(1)

    def check_pm2_installed(self) -> bool:
        """Check if PM2 is installed"""
        try:
            subprocess.run(["pm2", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_pm2(self) -> None:
        """Install PM2 globally"""
        print("📦 Installing PM2...")
        self.run_command(["npm", "install", "-g", "pm2"])
        print("✅ PM2 installed successfully")

    def create_log_directory(self) -> None:
        """Create PM2 log directory"""
        log_dir = self.project_root / "logs" / "pm2"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created log directory: {log_dir}")

    def start(self, env: str = "production") -> None:
        """Start validator with PM2"""
        if not self.check_pm2_installed():
            print("❌ PM2 not installed. Installing PM2...")
            self.install_pm2()

        self.create_log_directory()

        print(f"🚀 Starting validator with PM2 (env: {env})...")
        cmd = ["pm2", "start", str(self.pm2_config), "--env", env]
        self.run_command(cmd)
        print("✅ Validator started with PM2")
        print(
            "📝 Auto-updater will check for updates and notify when manual restart needed"
        )

        # Show status
        self.status()

    def stop(self) -> None:
        """Stop validator"""
        print("⏹️ Stopping validator...")
        self.run_command(["pm2", "stop", self.app_name])
        print("✅ Validator stopped")

    def restart(self, env: str = "production") -> None:
        """Restart validator (use this after auto-updates)"""
        print("🔄 Restarting validator...")
        print("📝 This will use the latest code if auto-update was performed")
        cmd = ["pm2", "restart", self.app_name, "--env", env]
        self.run_command(cmd)
        print("✅ Validator restarted with latest code")

    def delete(self) -> None:
        """Delete validator from PM2"""
        print("🗑️ Removing validator from PM2...")
        self.run_command(["pm2", "delete", self.app_name])
        print("✅ Validator removed from PM2")

    def status(self) -> None:
        """Show validator status"""
        print("📊 Validator status:")
        try:
            result = self.run_command(["pm2", "list"], capture_output=True)
            print(result.stdout)
        except subprocess.CalledProcessError:
            print("❌ Failed to get PM2 status")

    def logs(self, lines: int = 100, follow: bool = False) -> None:
        """Show validator logs"""
        cmd = ["pm2", "logs", self.app_name, "--lines", str(lines)]
        if follow:
            cmd.append("--follow")

        print(f"📋 Showing validator logs (last {lines} lines)...")
        print("📝 Look for auto-updater messages and update notifications")
        try:
            self.run_command(cmd)
        except KeyboardInterrupt:
            print("\n✅ Log viewing stopped")

    def monit(self) -> None:
        """Open PM2 monitoring interface"""
        print("📊 Opening PM2 monitoring interface...")
        self.run_command(["pm2", "monit"])

    def save(self) -> None:
        """Save PM2 process list"""
        print("💾 Saving PM2 process list...")
        self.run_command(["pm2", "save"])
        print("✅ PM2 process list saved")

    def startup(self) -> None:
        """Generate startup script for system boot"""
        print("🔧 Generating PM2 startup script...")
        try:
            result = self.run_command(["pm2", "startup"], capture_output=True)
            print(result.stdout)
            print("⚠️ Please run the command above as root to enable auto-startup")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate startup script: {e}")

    def unstartup(self) -> None:
        """Remove PM2 from startup"""
        print("🔧 Removing PM2 from startup...")
        self.run_command(["pm2", "unstartup"])
        print("✅ PM2 removed from startup")

    def reload(self) -> None:
        """Reload validator (zero-downtime restart)"""
        print("🔄 Reloading validator (zero-downtime)...")
        print("📝 This will use the latest code if auto-update was performed")
        self.run_command(["pm2", "reload", self.app_name])
        print("✅ Validator reloaded with latest code")

    def reset(self) -> None:
        """Reset validator restart counter"""
        print("🔄 Resetting validator restart counter...")
        self.run_command(["pm2", "reset", self.app_name])
        print("✅ Restart counter reset")

    def describe(self) -> None:
        """Show detailed validator information"""
        print("📝 Validator detailed information:")
        try:
            result = self.run_command(
                ["pm2", "describe", self.app_name], capture_output=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError:
            print("❌ Failed to get validator description")

    def validate_config(self) -> bool:
        """Validate PM2 configuration file"""
        if not self.pm2_config.exists():
            print(f"❌ PM2 config file not found: {self.pm2_config}")
            return False

        # Check if config file path in PM2 config matches actual script location
        validator_script = self.project_root / "scripts" / "run_validator.py"
        if not validator_script.exists():
            print(f"❌ Validator script not found: {validator_script}")
            return False

        print("✅ PM2 configuration validated")
        return True

    def update_info(self) -> None:
        """Show auto-update information"""
        print("📋 Auto-Update Information:")
        print("=" * 50)

        # Check version file
        version_file = self.project_root / "version.txt"
        if version_file.exists():
            version = version_file.read_text().strip()
            print(f"📖 Current version: {version}")
        else:
            print("❌ Version file not found")

        # Check update logs
        update_log_dir = self.project_root / "logs" / "updater"
        if update_log_dir.exists():
            log_files = list(update_log_dir.glob("updater_*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                print(f"📄 Latest update log: {latest_log}")
            else:
                print("📄 No update logs found")
        else:
            print("📄 Update log directory not found")

        print("\n📝 Auto-Update Process:")
        print("1. Auto-updater runs initial check on validator startup")
        print("2. Background checks occur every 12 hours")
        print("3. When update is found, code is updated automatically")
        print("4. Config directory is preserved during updates")
        print("5. Manual PM2 restart required to use new code")
        print("6. Use 'pm2 restart subnet-validator' or this script's restart command")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="PM2 Manager for Subnet Validator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start validator with PM2")
    start_parser.add_argument(
        "--env",
        default="production",
        choices=["production", "development"],
        help="Environment to use",
    )

    # Stop command
    subparsers.add_parser("stop", help="Stop validator")

    # Restart command
    restart_parser = subparsers.add_parser(
        "restart", help="Restart validator (use after auto-update)"
    )
    restart_parser.add_argument(
        "--env",
        default="production",
        choices=["production", "development"],
        help="Environment to use",
    )

    # Delete command
    subparsers.add_parser("delete", help="Remove validator from PM2")

    # Status command
    subparsers.add_parser("status", help="Show validator status")

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show validator logs")
    logs_parser.add_argument(
        "--lines", type=int, default=100, help="Number of lines to show"
    )
    logs_parser.add_argument("--follow", action="store_true", help="Follow log output")

    # Monitoring commands
    subparsers.add_parser("monit", help="Open PM2 monitoring interface")
    subparsers.add_parser("save", help="Save PM2 process list")
    subparsers.add_parser("startup", help="Generate startup script for system boot")
    subparsers.add_parser("unstartup", help="Remove PM2 from startup")
    subparsers.add_parser(
        "reload", help="Reload validator (zero-downtime restart with latest code)"
    )
    subparsers.add_parser("reset", help="Reset validator restart counter")
    subparsers.add_parser("describe", help="Show detailed validator information")
    subparsers.add_parser("validate", help="Validate PM2 configuration")
    subparsers.add_parser("update-info", help="Show auto-update information and status")

    return parser


def main():
    """Main function"""
    # Get project root (parent of scripts directory)
    script_path = Path(__file__).parent
    project_root = script_path.parent

    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n📝 Common commands:")
        print("  start    - Start validator with auto-updater")
        print("  restart  - Restart validator (use after auto-update)")
        print("  logs     - View logs (including auto-updater messages)")
        print("  status   - Check validator status")
        print("  update-info - Show auto-update information")
        return

    manager = PM2Manager(str(project_root))

    # Validate configuration for most commands
    if (
        args.command not in ["validate", "update-info"]
        and not manager.validate_config()
    ):
        sys.exit(1)

    # Execute command
    try:
        if args.command == "start":
            manager.start(args.env)
        elif args.command == "stop":
            manager.stop()
        elif args.command == "restart":
            manager.restart(args.env)
        elif args.command == "delete":
            manager.delete()
        elif args.command == "status":
            manager.status()
        elif args.command == "logs":
            manager.logs(args.lines, args.follow)
        elif args.command == "monit":
            manager.monit()
        elif args.command == "save":
            manager.save()
        elif args.command == "startup":
            manager.startup()
        elif args.command == "unstartup":
            manager.unstartup()
        elif args.command == "reload":
            manager.reload()
        elif args.command == "reset":
            manager.reset()
        elif args.command == "describe":
            manager.describe()
        elif args.command == "validate":
            if manager.validate_config():
                print("✅ All configurations are valid")
            else:
                sys.exit(1)
        elif args.command == "update-info":
            manager.update_info()
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
