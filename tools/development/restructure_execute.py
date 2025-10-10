#!/usr/bin/env python3
"""
Execute repository restructure into a distribution profile (non-destructive by default).

Usage examples:
  - Dry run (show actions only):
      python tools/restructure_execute.py --profile compression-only --dry-run

  - Materialize a dist/compression_only tree by copying needed files:
      python tools/restructure_execute.py --profile compression-only --output dist/compression_only

  - Dangerous: delete in-place everything not part of the profile (not recommended):
      python tools/restructure_execute.py --profile compression-only --delete-in-place

Profiles:
  compression-only: keep assembly kernels + Python compression codecs + minimal tests/docs/proofs.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]


def glob_many(base: Path, patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(base.glob(pat))
    return sorted(set([p for p in out if p.exists()]))


def profile_compression_only() -> dict:
    """Return include/exclude patterns for the compression-only distribution."""
    include = [
        # Kernels (assembly)
    "src/assembly/**",
        # Core compression algorithms and dependencies
        "src/core/rft_vertex_codec.py",
        "src/core/rft_hybrid_codec.py",
        "src/core/canonical_true_rft.py",
        "src/core/ans.py",
        "src/core/hybrid_residual_predictor.py",
        # CLI tools for compression
        "tools/rft_encode_model.py",
        "tools/rft_decode_model.py",
        "tools/rft_hybrid_compress.py",
        # Tests and proofs
    "tests/tests/test_rft_vertex_codec.py",
        "proofs/**",
        # Docs and license
        "docs/RFT_VALIDATION_GUIDE.md",
        "docs/TECHNICAL_SUMMARY.md",
        "docs/RFT_AUTOMATION_README.md",
        "LICENSE.md",
        # Configs
        "requirements.txt",
        "pytest.ini",
        # README (root) for context
        "README.md",
    ]
    exclude = [
        "ai/**",
        "encoded_models/**",
        "decoded_models/**",
        "hf_models/**",
        "hf_cache/**",
        "dev/**",
        "validation/**",
        "ui/**",
        "src/apps/**",
        "src/engine/**",
        # assembly build artifacts and caches
        "src/assembly/build/**",
        "src/assembly/compiled/**",
        "src/assembly/python_bindings/__pycache__/**",
        "docs/**/__pycache__/**",
        "**/__pycache__/**",
        ".pytest_cache/**",
        "logs/**",
        "results/**",
    ]
    return {"include": include, "exclude": exclude}


SKIP_SUFFIXES = {".pyc", ".pyo", ".o", ".so", ".a"}


def build_file_list(profile: str) -> List[Path]:
    if profile != "compression-only":
        raise ValueError(f"Unsupported profile: {profile}")
    spec = profile_compression_only()
    inc = glob_many(ROOT, spec["include"])
    # Filter out matches that are directories; we'll copy directories as needed
    files: List[Path] = []
    for p in inc:
        if p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file() and sub.suffix not in SKIP_SUFFIXES and "__pycache__" not in sub.as_posix():
                    files.append(sub)
        else:
            if p.suffix not in SKIP_SUFFIXES and "__pycache__" not in p.as_posix():
                files.append(p)
    return sorted(set(files))


def transform_rel_path(rel: Path, flatten: bool) -> Path:
    if not flatten:
        return rel
    rel_str = rel.as_posix()
    if rel_str.startswith("src/assembly/"):
        return Path("kernels") / rel_str.split("src/assembly/", 1)[1]
    if rel_str.startswith("src/core/"):
        return Path("core") / rel_str.split("src/core/", 1)[1]
    if rel_str.startswith("tests/tests/"):
        return Path("tests") / rel_str.split("tests/tests/", 1)[1]
    # tools/, proofs/, docs/, configs, README remain at top
    return rel


def materialize(files: List[Path], output: Path, dry_run: bool = True, flatten: bool = False) -> None:
    for f in files:
        rel = f.relative_to(ROOT)
        rel_out = transform_rel_path(rel, flatten)
        dest = output / rel_out
        if dry_run:
            print(f"COPY {rel} -> {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)


def delete_in_place(profile: str, dry_run: bool = True) -> None:
    """Dangerous: delete everything not included by the profile from the repo."""
    keep = set(build_file_list(profile))
    for p in ROOT.rglob("*"):
        if p.is_dir():
            continue
        if p.resolve() in keep:
            continue
        rel = p.relative_to(ROOT)
        if dry_run:
            print(f"DELETE {rel}")
        else:
            try:
                p.unlink()
            except Exception as exc:
                print(f"WARN cannot delete {rel}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="compression-only", choices=["compression-only"], help="Distribution profile")
    ap.add_argument("--dry-run", action="store_true", help="Print actions only (default when deleting)")
    ap.add_argument("--output", type=str, help="Output directory to copy the distribution (non-destructive)")
    ap.add_argument("--flatten", action="store_true", help="Flatten layout (src/assembly->kernels, src/core->core, tests/tests->tests)")
    ap.add_argument("--delete-in-place", action="store_true", help="Dangerous: delete everything not in profile")
    args = ap.parse_args()

    if args.delete_in_place and not args.dry_run:
        print("⚠️  In-place deletion without --dry-run is dangerous. Re-run with --dry-run to preview first.")
    files = build_file_list(args.profile)
    print(f"Profile '{args.profile}': {len(files)} files selected")

    if args.output:
        out_dir = (ROOT / args.output).resolve()
        print(f"Materializing distribution to: {out_dir}")
        materialize(files, out_dir, dry_run=args.dry_run, flatten=args.flatten)

    if args.delete_in_place:
        print("In-place deletion plan:")
        delete_in_place(args.profile, dry_run=True if args.dry_run else False)

    if not args.output and not args.delete_in_place:
        print("No action specified. Use --output to copy or --delete-in-place to prune (with --dry-run first).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
