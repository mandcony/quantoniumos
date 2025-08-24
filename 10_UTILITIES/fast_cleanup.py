#!/usr/bin/env python3
"""
FAST QuantoniumOS Duplicate Cleaner
===================================

Quick and dirty duplicate file cleanup that actually works.
No hanging, no complex hashing - just gets the job done.

This script:
1. Finds obvious duplicates by size + filename patterns
2. Moves backup/fixed/old files to attic
3. Cleans up empty files
4. Consolidates build variants
5. ACTUALLY FINISHES in reasonable time

Usage:
    python fast_cleanup.py --dry-run    # See what it would do
    python fast_cleanup.py --execute    # Actually do it
"""

import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class FastDuplicateCleaner:
    def __init__(self, root_path="C:\\quantoniumos-1"):
        self.root = Path(root_path)
        self.actions = []
        self.stats = {
            "files_scanned": 0,
            "duplicates_found": 0,
            "backups_found": 0,
            "empty_files": 0,
            "actions_planned": 0,
        }

        # Create attic directory
        self.attic = (
            self.root
            / "09_LEGACY_BACKUPS"
            / f"_attic_{datetime.now().strftime('%Y%m%d')}"
        )

        print(f"[INIT] Fast cleanup initialized")
        print(f"[ROOT] {self.root}")
        print(f"[ATTIC] {self.attic}")

    def log_action(self, action_type, source, reason):
        """Log an action to be taken."""
        self.actions.append(
            {
                "type": action_type,
                "source": str(source),
                "target": str(self.attic / source.relative_to(self.root)),
                "reason": reason,
            }
        )
        print(f"[{action_type}] {source.name} - {reason}")

    def scan_fast(self):
        """Quick scan focusing on obvious duplicates and problems."""
        print("\n=== FAST SCAN STARTING ===")

        # Skip these directories entirely
        skip_dirs = {".venv", ".git", "__pycache__", "node_modules", "build"}

        # Group files by name and size for fast duplicate detection
        by_name_size = defaultdict(list)

        for file_path in self.root.rglob("*"):
            if file_path.is_file():
                self.stats["files_scanned"] += 1

                # Skip if in excluded directory
                if any(skip_dir in str(file_path) for skip_dir in skip_dirs):
                    continue

                try:
                    size = file_path.stat().st_size
                    key = (file_path.name, size)
                    by_name_size[key].append(file_path)
                except:
                    continue

                # Check for obvious patterns while scanning
                self.check_file_patterns(file_path)

        # Find duplicates
        for (name, size), files in by_name_size.items():
            if len(files) > 1:
                self.handle_duplicates(files)

        print(f"[SCAN] Completed - {self.stats['files_scanned']} files scanned")

    def check_file_patterns(self, file_path):
        """Check individual files for cleanup patterns."""
        name = file_path.name.lower()

        # Empty files
        try:
            if file_path.stat().st_size == 0:
                self.stats["empty_files"] += 1
                self.log_action("REMOVE_EMPTY", file_path, "Empty file")
                return
        except:
            pass

        # Backup patterns
        backup_patterns = [
            ".backup",
            ".bak",
            "_backup",
            "_old",
            "_fixed",
            "_copy",
            "_v2",
            "_v3",
            "_final",
            "_temp",
            ".tmp",
        ]

        if any(pattern in name for pattern in backup_patterns):
            self.stats["backups_found"] += 1
            self.log_action("MOVE_BACKUP", file_path, f"Backup file pattern: {name}")

        # Test files that are empty or just imports
        if name.startswith("test_") and file_path.suffix == ".py":
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content.strip()) < 50 or "def test_" not in content:
                    self.log_action(
                        "MOVE_EMPTY_TEST", file_path, "Empty or degenerate test"
                    )
            except:
                pass

    def handle_duplicates(self, files):
        """Handle a group of files with same name and size."""
        if len(files) < 2:
            return

        # Sort by preference (organized dirs first, non-backup names first)
        def file_score(f):
            path_str = str(f)
            score = 0

            # Prefer organized directories
            if any(
                org in path_str
                for org in ["01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_"]
            ):
                score += 1000

            # Penalize backup names
            if any(
                bad in f.name.lower()
                for bad in ["backup", "old", "fixed", "copy", "temp"]
            ):
                score -= 500

            # Penalize root directory
            if len(f.relative_to(self.root).parts) == 1:
                score -= 100

            # Prefer newer files
            try:
                score += int(f.stat().st_mtime)
            except:
                pass

            return score

        sorted_files = sorted(files, key=file_score, reverse=True)
        winner = sorted_files[0]
        losers = sorted_files[1:]

        self.stats["duplicates_found"] += len(losers)

        for loser in losers:
            self.log_action(
                "MOVE_DUPLICATE", loser, f"Duplicate of {winner.relative_to(self.root)}"
            )

    def execute_cleanup(self, dry_run=True):
        """Execute the cleanup actions."""
        if dry_run:
            print(f"\n=== DRY RUN - NO CHANGES MADE ===")
        else:
            print(f"\n=== EXECUTING CLEANUP ===")
            self.attic.mkdir(parents=True, exist_ok=True)

        executed = 0
        for action in self.actions:
            source = Path(action["source"])
            target = Path(action["target"])

            if dry_run:
                print(f"[DRY] Would move {source.name} to attic")
            else:
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
                    print(f"[MOVED] {source.name} → attic")
                    executed += 1
                except Exception as e:
                    print(f"[ERROR] Failed to move {source.name}: {e}")

        self.stats["actions_planned"] = len(self.actions)
        if not dry_run:
            self.stats["actions_executed"] = executed

        return executed if not dry_run else len(self.actions)

    def generate_report(self):
        """Generate a summary report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "root_path": str(self.root),
            "attic_path": str(self.attic),
            "statistics": self.stats,
            "actions": self.actions[:100],  # First 100 actions
            "summary": {"total_actions": len(self.actions), "by_type": {}},
        }

        # Count actions by type
        for action in self.actions:
            action_type = action["type"]
            if action_type not in report["summary"]["by_type"]:
                report["summary"]["by_type"][action_type] = 0
            report["summary"]["by_type"][action_type] += 1

        # Save report
        report_path = self.root / "fast_cleanup_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n=== CLEANUP SUMMARY ===")
        print(f"Files scanned: {self.stats['files_scanned']}")
        print(f"Actions planned: {len(self.actions)}")
        print(f"Empty files: {self.stats['empty_files']}")
        print(f"Backup files: {self.stats['backups_found']}")
        print(f"Duplicates: {self.stats['duplicates_found']}")
        print(f"Report saved: {report_path}")

        return report


def main():
    import sys

    # Simple argument parsing
    dry_run = True
    if "--execute" in sys.argv:
        dry_run = False
        print("🚨 LIVE MODE - Changes will be made!")
        response = input("Are you sure? Type 'YES' to continue: ")
        if response != "YES":
            print("❌ Cancelled")
            return
    else:
        print("🔍 DRY RUN MODE - No changes will be made")

    cleaner = FastDuplicateCleaner()
    cleaner.scan_fast()
    actions_count = cleaner.execute_cleanup(dry_run=dry_run)
    report = cleaner.generate_report()

    if dry_run:
        print(f"\n✅ Dry run complete! {actions_count} actions planned.")
        print("Run with --execute to perform cleanup.")
    else:
        print(f"\n✅ Cleanup complete! {actions_count} actions executed.")


if __name__ == "__main__":
    main()
