#!/usr/bin/env python3
"""
Dry-run restructure: prints planned moves/renames based on docs/RESTRUCTURE_PLAN.md.
Non-destructive. Edit the PLAN to change the mapping rules.
"""
from __future__ import annotations
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "RESTRUCTURE_PLAN.md"

RULES = [
    (re.compile(r"^src/apps/(.*)$"), r"apps/\1"),
    (re.compile(r"^src/core/(.*)$"), r"core/\1"),
    (re.compile(r"^src/assembly/(.*)$"), r"kernels/\1"),
    (re.compile(r"^src/engine/(.*)$"), r"engine/\1"),
    (re.compile(r"^src/frontend/(.*)$"), r"ui/\1"),
    (re.compile(r"^ai/models/(.*)$"), r"models/\1"),
    (re.compile(r"^encoded_models/(.*)$"), r"models/encoded/\1"),
    (re.compile(r"^decoded_models/(.*)$"), r"models/decoded/\1"),
    (re.compile(r"^dev/tools/(.*)$"), r"tools/\1"),
    (re.compile(r"^quantonium_boot.py$"), r"boot.py"),
]

# Paths to ignore (already in target, vcs, caches)
SKIP_PREFIX = (
    ".git/", "results/", ".pytest_cache/", "__pycache__/", "models/",
)


def suggest_moves() -> list[tuple[str, str]]:
    moves = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(SKIP_PREFIX):
            continue
        for pattern, repl in RULES:
            m = pattern.match(rel)
            if m:
                dest = pattern.sub(repl, rel)
                if dest != rel:
                    moves.append((rel, dest))
                break
    return moves


def main() -> int:
    print(f"Loading plan: {PLAN}")
    if not PLAN.exists():
        print("Plan not found. Aborting.")
        return 1
    moves = suggest_moves()
    if not moves:
        print("No moves suggested.")
        return 0
    print("\nPlanned moves (dry-run):")
    for src, dst in moves:
        print(f"  {src}  ->  {dst}")
    print(f"\nTotal planned moves: {len(moves)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
