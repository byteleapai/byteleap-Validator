#!/usr/bin/env python3
"""
Standalone Update Checker
Can be run independently for update checks without PM2 restart
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.auto_updater import AutoUpdater


async def main():
    """Main function for standalone update checking"""
    try:
        updater = AutoUpdater(str(project_root))

        print("🔍 Checking for updates...")
        update_performed = await updater.check_and_update()

        if update_performed:
            print("🎉 Update completed successfully!")
            print("📝 IMPORTANT: Manual PM2 restart required to use new code")
            print("📝 Run: pm2 restart subnet-validator")
            print("📝 Or use: python3 scripts/pm2_manager.py restart")
        else:
            print("✅ No updates needed - already up to date")

        # Cleanup old backups
        updater.cleanup_old_backups()

    except Exception as e:
        print(f"❌ Update check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
