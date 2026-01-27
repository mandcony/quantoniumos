# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Build a multi-domain local text corpus from the QuantoniumOS repo.

Goal
- Create a *local* training dataset you can use to fine-tune your local LLM (LoRA)
  across the repo's domains (RFT, theory, manuals, safety posture, etc.).

This script produces a JSONL file with records:
  {"text": "..."}
which can be fed directly into TRL's SFTTrainer as plain causal-LM training text.

Important
- This is not a magical "make it omniscient" step. It mainly makes the model more fluent
  in YOUR repo's terminology and docs.
- Keep the system non-agentic: this script doesn't add tool use.

Usage
- python src/apps/build_domain_corpus.py --out ai/training/datasets/domain_corpus.jsonl

Then train:
- python src/apps/train_local_chat_lora.py --model distilgpt2 --corpus ai/training/datasets/domain_corpus.jsonl --out ai/training/models/domain_lora
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_ROOTS = [
    "docs",
    "algorithms/rft",
    "demos",
    "README.md",
    "SYSTEM_STATUS_REPORT.md",
]

RFT_PROFILE_ROOTS = [
    # RFTPU spec + key supporting docs
    "docs/RFTPU_TECHNICAL_SPECIFICATION_V2.md",
    "docs/ARCHITECTURE.md",
    "docs/ARCHITECTURE_QUICKREF.md",
    "docs/theory",
    "docs/technical",
    "docs/research",
    "algorithms/rft",
]

DEFAULT_EXTS = [".md", ".txt", ".py"]

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".hypothesis",
    "__pycache__",
    "node_modules",
    "results",
    "data/compressed_models",
    "papers",
    "figures",
}

DEFAULT_EXCLUDE_GLOBS = [
    "*.ipynb",
    "*.pdf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.pt",
    "*.bin",
    "*.safetensors",
    "*.onnx",
]


def _should_exclude_path(p: Path, exclude_dirs: set[str], exclude_globs: Sequence[str]) -> bool:
    parts = set(p.parts)
    if parts.intersection(exclude_dirs):
        return True
    name = p.name
    for g in exclude_globs:
        if fnmatch.fnmatch(name, g):
            return True
    return False


def iter_text_files(
    repo_root: Path,
    roots: Sequence[str],
    exts: Sequence[str],
    exclude_dirs: set[str],
    exclude_globs: Sequence[str],
    max_files: Optional[int],
) -> Iterator[Path]:
    count = 0
    for r in roots:
        rp = (repo_root / r).resolve()
        if not rp.exists():
            continue

        if rp.is_file():
            if rp.suffix.lower() in exts and not _should_exclude_path(rp, exclude_dirs, exclude_globs):
                yield rp
                count += 1
                if max_files is not None and count >= max_files:
                    return
            continue

        for p in rp.rglob("*"):
            if max_files is not None and count >= max_files:
                return
            if not p.is_file():
                continue
            if _should_exclude_path(p, exclude_dirs, exclude_globs):
                continue
            if p.suffix.lower() not in exts:
                continue
            yield p
            count += 1


def read_text_safely(p: Path, max_chars: int) -> Optional[str]:
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars]
    # Strip NULs or weird control chars
    txt = "".join(ch for ch in txt if ch != "\x00")
    return txt.strip() or None


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    overlap_chars = max(0, min(overlap_chars, chunk_chars - 1))
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = j - overlap_chars
    return chunks


def write_jsonl(out_path: Path, rows: Iterable[dict]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default="ai/training/datasets/domain_corpus.jsonl")
    ap.add_argument(
        "--profile",
        default="default",
        choices=["default", "rft"],
        help="Corpus profile: 'default' (broad) or 'rft' (RFT specs/docs focused)",
    )
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--ext", nargs="*", default=DEFAULT_EXTS)
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--max_chars_per_file", type=int, default=200_000)
    ap.add_argument("--chunk_chars", type=int, default=2000)
    ap.add_argument("--overlap_chars", type=int, default=200)
    args = ap.parse_args()

    if args.profile == "rft":
        args.roots = list(RFT_PROFILE_ROOTS)

    repo_root = Path(args.repo).resolve()
    out_path = Path(args.out)

    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)
    exclude_globs = list(DEFAULT_EXCLUDE_GLOBS)

    files = list(
        iter_text_files(
            repo_root=repo_root,
            roots=args.roots,
            exts=[e.lower() for e in args.ext],
            exclude_dirs=exclude_dirs,
            exclude_globs=exclude_globs,
            max_files=args.max_files,
        )
    )

    def rows_iter():
        for p in files:
            txt = read_text_safely(p, max_chars=args.max_chars_per_file)
            if not txt:
                continue
            rel = str(p.relative_to(repo_root))
            # light header to preserve provenance inside the corpus
            header = f"[FILE: {rel}]\n"
            for ch in chunk_text(txt, args.chunk_chars, args.overlap_chars):
                yield {"text": header + ch}

    n_rows = write_jsonl(out_path, rows_iter())
    print(f"Wrote {n_rows} chunks from {len(files)} files -> {out_path}")


if __name__ == "__main__":
    main()
