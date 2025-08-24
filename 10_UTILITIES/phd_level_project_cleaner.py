#!/usr/bin/env python3
"""
QuantoniumOS PhD-Level Project Cleanup Utility
Systematic Duplicate Consolidation & Organization

This utility implements evidence-based project cleanup with scientific rigor:
- Backup file analysis and consolidation  
- Duplicate detection and resolution
- Directory restructuring
- Build system unification
- Test infrastructure cleanup

All actions are logged with provenance tracking for doctoral-level documentation.
"""

import filecmp
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class PhDLevelProjectCleaner:
    """
    Scientific project cleanup with rigorous provenance tracking.
    """

    def __init__(self, base_path: str = "C:\\quantoniumos-1"):
        self.base_path = Path(base_path)
        self.cleanup_log = []
        self.duplicate_analysis = {}
        self.backup_analysis = {}
        self.timestamp = datetime.now().isoformat()

        # Cleanup strategy
        self.dry_run = True  # Safety first - log actions but don't execute
        self.preserve_working_files = True
        self.create_consolidated_backups = True

        print(f"[INIT] PhD-level project cleaner initialized")
        print(f"[MODE] Dry run: {self.dry_run}")
        print(f"[BASE] Working directory: {self.base_path}")

    def log_action(
        self,
        action: str,
        source: str,
        target: str,
        evidence: str,
        risk_level: str = "LOW",
    ):
        """Log cleanup action with scientific documentation."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "source": source,
            "target": target,
            "evidence": evidence,
            "risk_level": risk_level,
            "dry_run": self.dry_run,
            "source_hash": self._safe_file_hash(source),
            "target_hash": self._safe_file_hash(target)
            if target and os.path.exists(target)
            else None,
        }
        self.cleanup_log.append(entry)

        risk_icon = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}.get(risk_level, "❓")
        print(f"{risk_icon} [{action}] {source} → {target if target else 'DELETE'}")
        if evidence:
            print(f"    Evidence: {evidence}")

    def _safe_file_hash(self, filepath: str) -> Optional[str]:
        """Safely calculate file hash."""
        try:
            if filepath and os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()[:16]
        except:
            pass
        return None

    def analyze_backup_files(self) -> Dict[str, Any]:
        """Analyze all backup files and their relationships to current files."""
        print("\n=== BACKUP FILE ANALYSIS ===")

        backup_patterns = ["*.backup", "*_backup.*", "*_old.*", "*_orig.*"]
        backup_files = []

        for pattern in backup_patterns:
            backup_files.extend(self.base_path.glob(f"**/{pattern}"))

        analysis_results = {}

        for backup_file in backup_files:
            if ".venv" in str(backup_file) or "third_party" in str(backup_file):
                continue

            # Determine corresponding current file
            current_file = self._find_current_file(backup_file)

            # File comparison
            comparison = self._compare_files(backup_file, current_file)

            analysis_results[str(backup_file)] = {
                "backup_size": backup_file.stat().st_size
                if backup_file.exists()
                else 0,
                "current_file": str(current_file) if current_file else None,
                "current_exists": current_file.exists() if current_file else False,
                "current_size": current_file.stat().st_size
                if current_file and current_file.exists()
                else 0,
                "files_identical": comparison.get("identical", False),
                "backup_newer": comparison.get("backup_newer", False),
                "backup_larger": comparison.get("backup_larger", False),
                "recommendation": self._backup_recommendation(
                    backup_file, current_file, comparison
                ),
            }

            print(f"[BACKUP] {backup_file.name}")
            print(f"  Current: {current_file.name if current_file else 'None'}")
            print(
                f"  Recommendation: {analysis_results[str(backup_file)]['recommendation']}"
            )

        self.backup_analysis = analysis_results
        return analysis_results

    def _find_current_file(self, backup_file: Path) -> Optional[Path]:
        """Find the current file corresponding to a backup."""
        # Remove backup suffixes
        name = backup_file.name
        name = name.replace(".backup", "")
        name = name.replace("_backup", "")
        name = name.replace("_old", "")
        name = name.replace("_orig", "")

        # Look for current file in same directory
        current_file = backup_file.parent / name
        if current_file.exists():
            return current_file

        # Look in root directory
        root_file = self.base_path / name
        if root_file.exists():
            return root_file

        return None

    def _compare_files(
        self, backup_file: Path, current_file: Optional[Path]
    ) -> Dict[str, bool]:
        """Compare backup and current files."""
        if not current_file or not current_file.exists():
            return {"identical": False, "backup_newer": True, "backup_larger": True}

        try:
            # Size comparison
            backup_size = backup_file.stat().st_size
            current_size = current_file.stat().st_size
            backup_larger = backup_size > current_size

            # Modification time comparison
            backup_mtime = backup_file.stat().st_mtime
            current_mtime = current_file.stat().st_mtime
            backup_newer = backup_mtime > current_mtime

            # Content comparison (for reasonably sized files)
            identical = False
            if backup_size < 1024 * 1024:  # < 1MB
                try:
                    identical = filecmp.cmp(
                        str(backup_file), str(current_file), shallow=False
                    )
                except:
                    pass

            return {
                "identical": identical,
                "backup_newer": backup_newer,
                "backup_larger": backup_larger,
            }
        except:
            return {"identical": False, "backup_newer": False, "backup_larger": False}

    def _backup_recommendation(
        self,
        backup_file: Path,
        current_file: Optional[Path],
        comparison: Dict[str, bool],
    ) -> str:
        """Generate evidence-based recommendation for backup file."""
        if not current_file or not current_file.exists():
            return "RESTORE_BACKUP (current file missing)"

        if comparison.get("identical", False):
            return "DELETE_BACKUP (files identical)"

        if comparison.get("backup_newer", False) and comparison.get(
            "backup_larger", False
        ):
            return "REVIEW_BACKUP (newer and larger - potential improvements)"

        if comparison.get("backup_larger", False):
            return "REVIEW_BACKUP (larger file - potential additional content)"

        if current_file.stat().st_size == 0:
            return "RESTORE_BACKUP (current file empty)"

        return "KEEP_BACKUP (significant differences detected)"

    def analyze_duplicate_files(self) -> Dict[str, Any]:
        """Find and analyze duplicate files by content."""
        print("\n=== DUPLICATE FILE ANALYSIS ===")

        # Group files by hash
        hash_groups = {}
        file_count = 0

        for file_path in self.base_path.rglob("*"):
            if (
                file_path.is_file()
                and ".venv" not in str(file_path)
                and "third_party" not in str(file_path)
            ):
                file_hash = self._safe_file_hash(str(file_path))
                if file_hash:
                    if file_hash not in hash_groups:
                        hash_groups[file_hash] = []
                    hash_groups[file_hash].append(file_path)
                    file_count += 1

        # Find duplicates (groups with more than one file)
        duplicates = {
            hash_val: files for hash_val, files in hash_groups.items() if len(files) > 1
        }

        duplicate_analysis = {}
        total_duplicate_size = 0

        for hash_val, files in duplicates.items():
            files = sorted(files, key=lambda x: x.stat().st_size, reverse=True)
            primary_file = files[0]  # Largest file as primary
            duplicate_files = files[1:]

            file_size = primary_file.stat().st_size
            total_duplicate_size += file_size * len(duplicate_files)

            duplicate_analysis[hash_val] = {
                "primary_file": str(primary_file),
                "duplicate_files": [str(f) for f in duplicate_files],
                "file_size": file_size,
                "duplicate_count": len(duplicate_files),
                "space_wasted": file_size * len(duplicate_files),
                "recommendation": self._duplicate_recommendation(
                    primary_file, duplicate_files
                ),
            }

            print(f"[DUPLICATE] {primary_file.name} ({file_size} bytes)")
            for dup in duplicate_files:
                print(f"  → {dup}")

        summary = {
            "total_files_scanned": file_count,
            "unique_file_hashes": len(hash_groups),
            "duplicate_groups": len(duplicates),
            "total_duplicate_files": sum(
                len(files) - 1 for files in duplicates.values()
            ),
            "total_space_wasted": total_duplicate_size,
            "duplicate_analysis": duplicate_analysis,
        }

        print(f"[SUMMARY] {summary['duplicate_groups']} duplicate groups found")
        print(f"[SPACE] {total_duplicate_size / 1024 / 1024:.1f} MB wasted space")

        self.duplicate_analysis = summary
        return summary

    def _duplicate_recommendation(
        self, primary_file: Path, duplicates: List[Path]
    ) -> str:
        """Generate recommendation for handling duplicates."""
        # Preserve files in organized directories
        organized_dirs = [
            "01_START_HERE",
            "02_CORE_VALIDATORS",
            "03_RUNNING_SYSTEMS",
            "04_RFT_ALGORITHMS",
            "05_QUANTUM_ENGINES",
            "06_CRYPTOGRAPHY",
            "07_TESTS_BENCHMARKS",
            "08_RESEARCH_ANALYSIS",
        ]

        # Check if primary is in organized directory
        primary_organized = any(
            org_dir in str(primary_file) for org_dir in organized_dirs
        )

        if primary_organized:
            return "DELETE_DUPLICATES (keep organized directory version)"
        else:
            # Check if any duplicates are in organized directories
            organized_duplicates = [
                f
                for f in duplicates
                if any(org_dir in str(f) for org_dir in organized_dirs)
            ]
            if organized_duplicates:
                return "KEEP_ORGANIZED (move to organized directory, delete others)"
            else:
                return "KEEP_LARGEST (delete smaller duplicates)"

    def consolidate_build_system(self) -> Dict[str, Any]:
        """Consolidate multiple build systems into canonical version."""
        print("\n=== BUILD SYSTEM CONSOLIDATION ===")

        # Find all build-related files
        build_files = {
            "cmake": list(self.base_path.glob("**/CMakeLists*.txt")),
            "build_scripts": list(self.base_path.glob("**/build_*.py")),
            "setup_scripts": list(self.base_path.glob("**/setup*.py")),
        }

        consolidation_plan = {}

        for category, files in build_files.items():
            if len(files) <= 1:
                continue

            # Analyze each file
            file_analysis = []
            for file_path in files:
                size = file_path.stat().st_size
                mtime = file_path.stat().st_mtime
                is_main = (
                    "backup" not in file_path.name and "fixed" not in file_path.name
                )

                file_analysis.append(
                    {
                        "path": file_path,
                        "size": size,
                        "mtime": mtime,
                        "is_main": is_main,
                        "score": (2 if is_main else 1)
                        + (size / 1000)
                        + (mtime / 1000000000),
                    }
                )

            # Select canonical version (highest score)
            file_analysis.sort(key=lambda x: x["score"], reverse=True)
            canonical = file_analysis[0]
            others = file_analysis[1:]

            consolidation_plan[category] = {
                "canonical_file": str(canonical["path"]),
                "files_to_review": [str(f["path"]) for f in others],
                "recommendation": f"Keep {canonical['path'].name} as canonical, review others for improvements",
            }

            print(f"[BUILD] {category.upper()}")
            print(f"  Canonical: {canonical['path'].name}")
            for other in others:
                print(f"  Review: {other['path'].name}")

        return consolidation_plan

    def execute_safe_cleanup(self, execute: bool = False) -> Dict[str, Any]:
        """Execute cleanup actions based on analysis (dry run by default)."""
        print(f"\n=== CLEANUP EXECUTION ({'LIVE' if execute else 'DRY RUN'}) ===")

        if execute:
            self.dry_run = False
            print("🚨 LIVE EXECUTION MODE - Changes will be made to filesystem")
            response = input(
                "Are you sure you want to proceed? (type 'YES' to confirm): "
            )
            if response != "YES":
                print("❌ Cleanup cancelled")
                return {"status": "CANCELLED"}

        actions_planned = 0
        actions_executed = 0

        # Process backup files
        for backup_path, analysis in self.backup_analysis.items():
            recommendation = analysis["recommendation"]

            if recommendation == "DELETE_BACKUP (files identical)":
                self.log_action(
                    "DELETE_BACKUP",
                    backup_path,
                    "",
                    "Files are identical - backup redundant",
                    "LOW",
                )
                if not self.dry_run:
                    try:
                        os.remove(backup_path)
                        actions_executed += 1
                    except Exception as e:
                        print(f"  Error: {e}")
                actions_planned += 1

            elif recommendation.startswith("RESTORE_BACKUP"):
                current_file = analysis.get("current_file")
                if current_file:
                    self.log_action(
                        "RESTORE_FROM_BACKUP",
                        backup_path,
                        current_file,
                        recommendation,
                        "MEDIUM",
                    )
                    if not self.dry_run:
                        try:
                            shutil.copy2(backup_path, current_file)
                            actions_executed += 1
                        except Exception as e:
                            print(f"  Error: {e}")
                    actions_planned += 1

        # Process duplicates (low-risk only in this pass)
        for hash_val, analysis in self.duplicate_analysis.get(
            "duplicate_analysis", {}
        ).items():
            if (
                analysis["recommendation"]
                == "DELETE_DUPLICATES (keep organized directory version)"
            ):
                for dup_path in analysis["duplicate_files"]:
                    self.log_action(
                        "DELETE_DUPLICATE",
                        dup_path,
                        "",
                        "Keeping organized directory version",
                        "LOW",
                    )
                    if not self.dry_run:
                        try:
                            os.remove(dup_path)
                            actions_executed += 1
                        except Exception as e:
                            print(f"  Error: {e}")
                    actions_planned += 1

        summary = {
            "mode": "LIVE" if not self.dry_run else "DRY_RUN",
            "actions_planned": actions_planned,
            "actions_executed": actions_executed,
            "success_rate": actions_executed / actions_planned
            if actions_planned > 0
            else 0,
            "total_log_entries": len(self.cleanup_log),
        }

        print(
            f"[SUMMARY] {actions_planned} actions planned, {actions_executed} executed"
        )
        return summary

    def generate_cleanup_report(self) -> Dict[str, Any]:
        """Generate comprehensive cleanup analysis report."""
        print("\n=== GENERATING COMPREHENSIVE CLEANUP REPORT ===")

        # Run all analyses
        backup_analysis = self.analyze_backup_files()
        duplicate_analysis = self.analyze_duplicate_files()
        build_consolidation = self.consolidate_build_system()
        cleanup_summary = self.execute_safe_cleanup(execute=False)  # Dry run

        # Compile comprehensive report
        report = {
            "cleanup_metadata": {
                "timestamp": self.timestamp,
                "base_path": str(self.base_path),
                "total_actions_logged": len(self.cleanup_log),
            },
            "backup_analysis": backup_analysis,
            "duplicate_analysis": duplicate_analysis,
            "build_consolidation": build_consolidation,
            "cleanup_summary": cleanup_summary,
            "action_log": self.cleanup_log,
            "recommendations": {
                "immediate_actions": [
                    "Review backup files marked for restoration",
                    "Delete identical backup files to save space",
                    "Consolidate build system variants",
                ],
                "medium_term": [
                    "Move root directory files to organized subdirectories",
                    "Compress legacy archive directory",
                    "Standardize test infrastructure",
                ],
                "risk_assessment": "Current cleanup plan is low-risk with full provenance tracking",
            },
        }

        # Save report
        report_path = self.base_path / "PhD_LEVEL_CLEANUP_REPORT.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[REPORT] Comprehensive cleanup report saved: {report_path}")
        print(f"[BACKUP] {len(backup_analysis)} backup files analyzed")
        print(
            f"[DUPLICATES] {duplicate_analysis['duplicate_groups']} duplicate groups found"
        )
        print(
            f"[SPACE] {duplicate_analysis['total_space_wasted'] / 1024 / 1024:.1f} MB recoverable"
        )

        return report


if __name__ == "__main__":
    print("=== QUANTONIUMOS PhD-LEVEL PROJECT CLEANUP ===")
    print("Scientific Duplicate Consolidation & Organization")
    print("=" * 60)

    cleaner = PhDLevelProjectCleaner()
    report = cleaner.generate_cleanup_report()

    print("\n=== CLEANUP ANALYSIS COMPLETE ===")
    print("All actions logged with scientific provenance tracking")
    print("Review cleanup report before executing any changes")
