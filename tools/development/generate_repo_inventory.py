#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Generate a structured inventory of the repository:
- Classify files into categories (source, tests, docs, tools, data/models, artifacts, misc)
- Summarize sizes per directory
- Emit JSON and Markdown catalogs under results/inventory/

Non-destructive: does not move or modify any files.
"""
from __future__ import annotations
import os
import sys
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "inventory"

CATEGORIES = {
    "source": ["src/", "core/"],
    "tests": ["tests/", "validation/", "test_", "direct_bell_test.py"],
    "docs": ["docs/", "README.md", "*.md"],
    "tools": ["tools/", "dev/tools/", "dev/scripts/", "dev/examples/"],
    "ui": ["ui/"],
    "models": ["ai/", "encoded_models/", "decoded_models/", "hf_models/", "hf_cache/"],
    "artifacts": ["logs/", "fast_validation_*.json", "*.log", "tests/results/", "validation/results/"],
    "configs": ["requirements.txt", "pytest.ini", ".gitignore", "Makefile"],
}

IGNORE_DIRS = { ".git", ".venv", "env", "venv", "node_modules", ".pytest_cache", "__pycache__" }

@dataclass
class FileEntry:
    path: str
    size: int
    category: str

@dataclass
class DirSummary:
    path: str
    total_size: int
    file_count: int


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < 1024 or u == units[-1]:
            return f"{s:.1f} {u}"
        s /= 1024


def classify(path: Path) -> str:
    rel = path.as_posix()
    # Priority ordering: models/artifacts before generic docs
    if any(rel.startswith(p.rstrip("/")) for p in CATEGORIES["models"] if p.endswith("/")):
        return "models"
    if any(rel.startswith(p.rstrip("/")) for p in CATEGORIES["source"] if p.endswith("/")):
        return "source"
    if any(rel.startswith(p.rstrip("/")) for p in CATEGORIES["tests"] if p.endswith("/")) or rel.startswith("tests/"):
        return "tests"
    if any(rel.startswith(p.rstrip("/")) for p in CATEGORIES["tools"] if p.endswith("/")):
        return "tools"
    if any(rel.startswith(p.rstrip("/")) for p in CATEGORIES["docs"] if p.endswith("/")) or rel.endswith(".md"):
        return "docs"
    if any(rel.startswith(p.rstrip("/")) for p in CATEGORIES["artifacts"] if p.endswith("/")):
        return "artifacts"
    if rel in ("requirements.txt", "pytest.ini", ".gitignore", "Makefile"):
        return "configs"
    return "misc"


def scan_repo(root: Path) -> Tuple[List[FileEntry], Dict[str, DirSummary]]:
    files: List[FileEntry] = []
    dir_summaries: Dict[str, DirSummary] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith('.') or d in (".github",)]
        dpath = Path(dirpath)
        total_size = 0
        file_count = 0
        rel_dir = dpath.relative_to(root).as_posix() or "."
        for fn in filenames:
            fpath = dpath / fn
            try:
                size = fpath.stat().st_size
            except FileNotFoundError:
                continue
            # Skip .git internals explicitly
            if "/.git/" in fpath.as_posix():
                continue
            rel_path = fpath.relative_to(root)
            category = classify(rel_path)
            files.append(FileEntry(path=rel_path.as_posix(), size=size, category=category))
            total_size += size
            file_count += 1
        dir_summaries[rel_dir] = DirSummary(path=rel_dir, total_size=total_size, file_count=file_count)
    return files, dir_summaries


def write_outputs(files: List[FileEntry], dirs: Dict[str, DirSummary]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON inventory
    payload = {
        "root": str(ROOT),
        "totals": {
            "files": len(files),
            "bytes": sum(f.size for f in files),
        },
        "by_category": {},
        "files": [asdict(f) for f in files],
        "dirs": {k: asdict(v) for k, v in sorted(dirs.items())},
    }
    # Category aggregation
    by_cat: Dict[str, Dict[str, int]] = {}
    for f in files:
        agg = by_cat.setdefault(f.category, {"files": 0, "bytes": 0})
        agg["files"] += 1
        agg["bytes"] += f.size
    payload["by_category"] = by_cat

    (OUT_DIR / "repo_inventory.json").write_text(json.dumps(payload, indent=2))

    # Markdown summary
    lines: List[str] = []
    lines.append("# Repository Inventory Summary\n")
    lines.append(f"Root: `{ROOT}`\n")
    lines.append("## Category Totals\n")
    for cat, agg in sorted(by_cat.items(), key=lambda kv: kv[1]["bytes"], reverse=True):
        lines.append(f"- {cat}: {agg['files']} files, {human_bytes(agg['bytes'])}")
    lines.append("\n## Top-level Directories (by size)\n")
    top_dirs = [(p, d) for p, d in dirs.items() if "/" not in p or p == "."]
    for p, d in sorted(top_dirs, key=lambda kv: kv[1].total_size, reverse=True):
        lines.append(f"- {p}: {d.file_count} files, {human_bytes(d.total_size)}")

    lines.append("\n## Notable Files\n")
    # Largest 30 files
    largest = sorted(files, key=lambda f: f.size, reverse=True)[:30]
    for f in largest:
        lines.append(f"- {f.path} — {human_bytes(f.size)} ({f.category})")

    (OUT_DIR / "repo_inventory.md").write_text("\n".join(lines))


def main() -> int:
    files, dirs = scan_repo(ROOT)
    write_outputs(files, dirs)
    print(f"Inventory written to: {OUT_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
