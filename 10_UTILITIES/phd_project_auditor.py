#!/usr/bin/env python3
"""
PhD Project Auditor — QuantoniumOS Edition
==========================================

Scans a repository to:
- Find exact duplicates (size+SHA-256) and propose a canonical "winner"
- Detect backups/nonsensical files (e.g., *.backup, *_fixed.*, *old*, temp)
- Flag empty files, empty/degenerate tests, root-directory clutter
- Spot orphan artifacts (.pyc without .py, zero-byte .pyd/.so)
- Enumerate build system duplicates (CMakeLists, build_* scripts)
- Generate:
    - JSON report (machine-readable)
    - Markdown report (human-readable)
    - Cleanup plan shell script (dry-run friendly; moves to attic)
      with provenance-safe "git mv" operations

Default behavior is read-only (dry run). Use --execute to emit a plan that
you can run (still non-destructive; moves to attic directory).

Usage
-----
# Audit the current repo, excluding common virtualenv/build caches:
python phd_project_auditor.py --root . --export-dir artifacts/audit --name run_$(date +%F)

# Produce a cleanup plan you can review & run:
python phd_project_auditor.py --root . --export-dir artifacts/audit --execute

Notes
-----
- The auditor uses a deterministic policy to pick canonical files among duplicates:
  1) Prefer files NOT matching backup/fixed/old/copy patterns
  2) Prefer files OUTSIDE legacy/attic/archive dirs
  3) Prefer files deeper inside organized dirs over repo root
  4) Prefer newer mtime
  5) Prefer larger file (heuristic for non-stubs)
  6) As tiebreaker, prefer shorter path

- All "losers" are slated to move to: 09_LEGACY_BACKUPS/_attic_<YYYYMMDD>/

- The script never deletes; it only *plans* moves. You can edit the plan before running.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------- Configuration ---------------------------

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "out",
    "build",
    "cmake-build",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".cache",
}

LEGACY_HINT_DIRS = {"09_LEGACY_BACKUPS", "legacy", "archive", "_attic", "attic"}

# backup / duplicate-y name patterns
BACKUP_PATTERNS = [
    r"\.backup$",
    r"\.bak$",
    r"\.old(\.\w+)?$",
    r"~$",
    r"\.tmp$",
    r"(^|[_\-])(backup|bak|copy|final|old|old\d*|draft|dup|duplicate|copy\d*|v\d+)(\.\w+)?$",
    r"(^|[_\-])(fixed|fix|patched)(\.\w+)?$",
]

# test file patterns
TEST_PATTERNS = [r"(^|/|\\)test_.*\.py$", r"(^|/|\\).*_test\.py$"]

# suspicious single-file names in repo root
SUSPICIOUS_ROOT_FILES = {
    "tmp",
    "temp",
    "notes.txt",
    "scratch.py",
    "script.py",
    "run.py",
}

# Extensions considered compiled/binary modules
BINARY_EXTS = {".pyd", ".so", ".dll"}

# --------------------------- Helpers ---------------------------


def sha256_file(path: Path, chunk_size: int = 1048576) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_text_file(path: Path, sniff_bytes: int = 4096) -> bool:
    try:
        b = path.read_bytes()[:sniff_bytes]
        if b"\x00" in b:
            return False
        # Heuristic: try decode as UTF-8
        try:
            b.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    except Exception:
        return False


def matches_any(patterns, name: str) -> bool:
    for p in patterns:
        if re.search(p, name, re.IGNORECASE):
            return True
    return False


def classify_test_emptiness(p: Path) -> str:
    """Return 'empty', 'degenerate', or 'ok' for test files."""
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "unknown"
    stripped = re.sub(r"#.*", "", txt).strip()
    if len(stripped) == 0:
        return "empty"
    # Degenerate: just imports/pass/no tests discovered
    if re.fullmatch(
        r"(?:from\s+\S+\s+import\s+\S+|import\s+\S+|pass|\s+|\n)+",
        stripped,
        re.MULTILINE,
    ):
        return "degenerate"
    # Has at least one "def test_" or unittest TestCase subclass
    if re.search(r"def\s+test_", stripped) or re.search(
        r"class\s+\w*\(.*TestCase.*\):", stripped
    ):
        return "ok"
    # Might still be a helper; mark as degenerate to encourage consolidation
    return "degenerate"


def in_any_dir(path: Path, names: set[str]) -> bool:
    parts = {p.name for p in path.parents}
    return bool(parts & names)


def depth_score(path: Path) -> int:
    # deeper is "more organized" → higher score
    return len(path.parts)


def root_clutter_score(path: Path, root: Path) -> int:
    rel = path.relative_to(root)
    return 0 if len(rel.parts) > 1 else 1  # 1 if at repo root, else 0


def is_legacy_path(path: Path) -> bool:
    return in_any_dir(path, LEGACY_HINT_DIRS)


def is_backupish_name(path: Path) -> bool:
    return matches_any(BACKUP_PATTERNS, path.name)


def winner_score(path: Path, root: Path) -> tuple:
    """Higher tuple sorts earlier (better)."""
    # Prefer non-backupish
    non_backup = 0 if is_backupish_name(path) else 1
    # Prefer non-legacy
    non_legacy = 0 if is_legacy_path(path) else 1
    # Prefer not at repo root
    not_root = 0 if root_clutter_score(path, root) else 1
    # Depth score (deeper better)
    dscore = depth_score(path)
    # Newer mtime, larger size read at call site
    stat = path.stat()
    mtime = int(stat.st_mtime)
    size = stat.st_size
    # Prefer shorter path (invert)
    neg_len = -len(str(path))
    return (non_backup, non_legacy, not_root, dscore, mtime, size, -neg_len)


# --------------------------- Core Audit ---------------------------


def scan_files(root: Path, exclude_dirs: set[str]) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_dir():
            # skip excluded directories
            if p.name in exclude_dirs:
                # prune traversal by skipping subdirs under excluded
                # rglob doesn't allow prune; we filter on file iteration
                continue
            else:
                continue
        # files only
        skip = False
        for parent in p.parents:
            if parent.name in exclude_dirs:
                skip = True
                break
        if skip:
            continue
        files.append(p)
    return files


def group_duplicates(files: list[Path]) -> dict[str, list[Path]]:
    by_size = defaultdict(list)
    for f in files:
        try:
            by_size[f.stat().st_size].append(f)
        except FileNotFoundError:
            continue
    dup_groups = {}
    for size, group in by_size.items():
        if len(group) < 2:
            continue
        # hash files of this size
        hashes = defaultdict(list)
        for g in group:
            try:
                h = sha256_file(g)
                hashes[h].append(g)
            except Exception:
                continue
        for h, paths in hashes.items():
            if len(paths) > 1:
                dup_groups[h] = paths
    return dup_groups


def find_empty_files(files: list[Path]) -> list[Path]:
    empties = []
    for f in files:
        try:
            if f.stat().st_size == 0:
                empties.append(f)
        except FileNotFoundError:
            pass
    return empties


def find_backupish(files: list[Path]) -> list[Path]:
    out = []
    for f in files:
        if is_backupish_name(f):
            out.append(f)
    return out


def find_orphans(files: list[Path], root: Path) -> dict[str, list[Path]]:
    py_files = {f for f in files if f.suffix == ".py"}
    pyc_files = [f for f in files if f.suffix == ".pyc"]
    pyd_so = [f for f in files if f.suffix in {".pyd", ".so"}]
    orphans = {"pyc_without_py": [], "empty_binaries": [], "py_empty_but_pyc": []}

    py_names = {str(f.with_suffix(".py")) for f in files}

    for c in pyc_files:
        src = None
        # locate corresponding .py (pycache layout varies; be conservative)
        name = c.name
        base = name.split(".")[0]  # module
        # best effort: walk up two dirs, search for base.py
        for up in [c.parent, c.parent.parent, root]:
            cand = list(up.rglob(f"{base}.py"))
            if cand:
                src = cand[0]
                break
        if not src:
            orphans["pyc_without_py"].append(c)

    for b in pyd_so:
        try:
            if b.stat().st_size == 0:
                orphans["empty_binaries"].append(b)
        except FileNotFoundError:
            pass

    for p in py_files:
        try:
            if p.stat().st_size == 0:
                # check any .pyc nearby
                pyc = list(p.parent.glob(f"__pycache__/{p.stem}*.pyc"))
                if pyc:
                    orphans["py_empty_but_pyc"].append(p)
        except FileNotFoundError:
            pass

    return orphans


def find_build_system_duplicates(files: list[Path]) -> dict[str, list[Path]]:
    cmakes = [f for f in files if f.name.lower().startswith("cmakelists")]
    build_scripts = [
        f
        for f in files
        if f.name.startswith(("build_", "Build_", "BUILD_"))
        and f.suffix in {".py", ".sh", ""}
    ]
    setup_variants = [
        f
        for f in files
        if f.name.startswith(("setup_", "install_", "configure_"))
        and f.suffix in {".py", ".sh"}
    ]
    return {
        "cmake_variants": cmakes,
        "build_script_variants": build_scripts,
        "setup_variants": setup_variants,
    }


def find_tests(files: list[Path]) -> dict:
    tests = []
    for f in files:
        s = str(f).replace("\\", "/")
        if matches_any(TEST_PATTERNS, s):
            tests.append(f)

    empty_tests, degenerate_tests = [], []
    for t in tests:
        cls = classify_test_emptiness(t)
        if cls == "empty":
            empty_tests.append(t)
        elif cls == "degenerate":
            degenerate_tests.append(t)
    return {
        "all_tests": tests,
        "empty_tests": empty_tests,
        "degenerate_tests": degenerate_tests,
    }


def choose_winners(dup_groups: dict[str, list[Path]], root: Path) -> dict[str, dict]:
    plan = {}
    for h, paths in dup_groups.items():
        scored = sorted(paths, key=lambda p: winner_score(p, root), reverse=True)
        winner = scored[0]
        losers = scored[1:]
        plan[h] = {
            "hash": h,
            "winner": str(winner),
            "losers": [str(x) for x in losers],
            "group_size": len(paths),
        }
    return plan


def make_attic_path(root: Path) -> Path:
    date = datetime.datetime.now().strftime("%Y%m%d")
    attic = root / "09_LEGACY_BACKUPS" / f"_attic_{date}"
    return attic


def render_cleanup_plan(
    sh: Path, plan: dict, root: Path, execute: bool = False
) -> None:
    attic = make_attic_path(root)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'mkdir -p "{attic}"',
        'echo "Moving duplicate/backup files to attic (provenance-preserving)..."',
    ]
    n_moves = 0

    # duplicate losers → attic
    for _, entry in plan.get("duplicate_resolution", {}).items():
        for loser in entry.get("losers", []):
            los = Path(loser)
            rel = los.relative_to(root)
            target = attic / rel
            lines.append(f'mkdir -p "{target.parent}"')
            lines.append(f'git mv "{los}" "{target}"')
            n_moves += 1

    # backupish files (non-duplicates but suspicious) → attic
    for b in plan.get("backupish", []):
        bpath = Path(b)
        rel = bpath.relative_to(root)
        target = attic / rel
        lines.append(f'mkdir -p "{target.parent}"')
        lines.append(f'git mv "{bpath}" "{target}"')
        n_moves += 1

    # empty/degenerate tests → attic (not delete)
    for t in plan.get("empty_tests", []):
        tpath = Path(t)
        rel = tpath.relative_to(root)
        target = attic / rel
        lines.append(f'mkdir -p "{target.parent}"')
        lines.append(f'git mv "{tpath}" "{target}"')
        n_moves += 1

    for t in plan.get("degenerate_tests", []):
        tpath = Path(t)
        rel = tpath.relative_to(root)
        target = attic / rel
        lines.append(f'mkdir -p "{target.parent}"')
        lines.append(f'git mv "{tpath}" "{target}"')
        n_moves += 1

    # build/setup variants → attic (leave 1 canonical to be decided manually)
    for v in plan.get("build_variants_to_attic", []):
        vpath = Path(v)
        rel = vpath.relative_to(root)
        target = attic / rel
        lines.append(f'mkdir -p "{target.parent}"')
        lines.append(f'git mv "{vpath}" "{target}"')
        n_moves += 1

    lines.append(f'echo "Planned moves: {n_moves}"')

    sh.write_text("\n".join(lines), encoding="utf-8")
    sh.chmod(0o755)


# --------------------------- Markdown/JSON Reports ---------------------------


def write_reports(export_dir: Path, summary: dict) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "audit_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = []
    md.append("# QuantoniumOS PhD Project Audit Report")
    md.append("")
    md.append(f"**Timestamp:** {summary['timestamp']}  ")
    md.append(f"**Root:** `{summary['root']}`  ")
    md.append(f"**Files scanned:** {summary['stats']['files_scanned']}  ")
    md.append("")
    md.append("## Highlights")
    md.append(f"- Empty files: **{summary['stats']['empty_files']}**")
    md.append(
        f"- Duplicate groups: **{summary['stats']['duplicate_groups']}** (exact SHA-256 matches)"
    )
    md.append(f"- Backup-ish files: **{len(summary['backupish'])}**")
    md.append(
        f"- Empty tests: **{len(summary['empty_tests'])}**, Degenerate tests: **{len(summary['degenerate_tests'])}**"
    )
    md.append("")
    md.append("## Duplicate Resolution Plan")
    md.append("Winner → losers")
    for h, grp in summary.get("duplicate_resolution", {}).items():
        md.append(f"- `{grp['winner']}`  ")
        for los in grp.get("losers", []):
            md.append(f"  - loser: `{los}`")
    md.append("")
    md.append("## Backup-ish Files")
    for b in summary["backupish"][:100]:
        md.append(f"- `{b}`")
    if len(summary["backupish"]) > 100:
        md.append(f"... and {len(summary['backupish'])-100} more")
    md.append("")
    md.append("## Problematic Tests")
    md.append("### Empty tests")
    for t in summary["empty_tests"][:50]:
        md.append(f"- `{t}`")
    if len(summary["empty_tests"]) > 50:
        md.append(f"... and {len(summary['empty_tests'])-50} more")
    md.append("### Degenerate tests")
    for t in summary["degenerate_tests"][:50]:
        md.append(f"- `{t}`")
    if len(summary["degenerate_tests"]) > 50:
        md.append(f"... and {len(summary['degenerate_tests'])-50} more")
    md.append("")
    md.append("## Orphans")
    for k, arr in summary.get("orphans", {}).items():
        md.append(f"### {k}")
        for p in arr[:50]:
            md.append(f"- `{p}`")
        if len(arr) > 50:
            md.append(f"... and {len(arr)-50} more")
    md.append("")
    md.append("## Build/Setup Variants")
    for k, arr in summary.get("build_system_duplicates", {}).items():
        md.append(f"### {k}")
        for p in arr:
            md.append(f"- `{p}`")
    md.append("")
    md.append("## Cleanup Plan (shell script)")
    md.append(f"- Generated at: `{summary['cleanup_script']}`")
    (export_dir / "audit_report.md").write_text("\n".join(md), encoding="utf-8")


# --------------------------- Main ---------------------------


def main():
    ap = argparse.ArgumentParser(
        description="PhD-level repository auditor for QuantoniumOS (duplicates, backups, empties, tests, build variants)."
    )
    ap.add_argument("--root", type=str, default=".", help="Repository root to scan.")
    ap.add_argument(
        "--export-dir",
        type=str,
        default="artifacts/audit",
        help="Directory to put reports and plan.",
    )
    ap.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Extra directory names to exclude.",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Generate cleanup plan shell script (still moves to attic; does not delete).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    export_dir = Path(args.export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | set(args.exclude)

    all_files = scan_files(root, exclude_dirs)
    files_scanned = len(all_files)

    empties = find_empty_files(all_files)
    backups = find_backupish(all_files)
    dup_groups = group_duplicates(all_files)
    tests_info = find_tests(all_files)
    orphans = find_orphans(all_files, root)
    build_dups = find_build_system_duplicates(all_files)

    # winner/loser plan for duplicates
    winner_plan = choose_winners(dup_groups, root)

    # Decide which build/setup variants to attic by default (keep lexicographically first as temp winner)
    build_variants_to_attic = []
    for key, arr in build_dups.items():
        if not arr:
            continue
        sorted_arr = sorted(arr)
        keep = sorted_arr[0]
        for f in sorted_arr[1:]:
            build_variants_to_attic.append(str(f))

    cleanup_script = export_dir / "cleanup_plan.sh"
    plan = {
        "duplicate_resolution": winner_plan,
        "backupish": [
            str(b)
            for b in backups
            if str(b) not in {w["winner"] for w in winner_plan.values()}
        ],
        "empty_tests": [str(p) for p in tests_info["empty_tests"]],
        "degenerate_tests": [str(p) for p in tests_info["degenerate_tests"]],
        "build_variants_to_attic": build_variants_to_attic,
    }
    render_cleanup_plan(cleanup_script, plan, root, execute=args.execute)

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "root": str(root),
        "stats": {
            "files_scanned": files_scanned,
            "empty_files": len(empties),
            "duplicate_groups": len(dup_groups),
        },
        "empty_files": [str(p) for p in empties],
        "backupish": [str(p) for p in backups],
        "orphans": {k: [str(p) for p in v] for k, v in orphans.items()},
        "tests": {
            "all_tests": [str(p) for p in tests_info["all_tests"]],
        },
        "empty_tests": [str(p) for p in tests_info["empty_tests"]],
        "degenerate_tests": [str(p) for p in tests_info["degenerate_tests"]],
        "build_system_duplicates": {
            k: [str(p) for p in v] for k, v in build_dups.items()
        },
        "duplicate_resolution": winner_plan,
        "cleanup_script": str(cleanup_script),
    }

    write_reports(export_dir, summary)

    print(
        json.dumps(
            {
                "ok": True,
                "export_dir": str(export_dir),
                "cleanup_script": str(cleanup_script),
                "files_scanned": files_scanned,
                "empty_files": len(empties),
                "duplicate_groups": len(dup_groups),
                "backupish_files": len(backups),
                "empty_tests": len(tests_info["empty_tests"]),
                "degenerate_tests": len(tests_info["degenerate_tests"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
