#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Report how much disk space your local AI stack is using.

This is meant to answer: "how much space do we save" when serving locally with
OSS models (e.g., Ollama) and/or HF caches + LoRA adapters.

What it measures (if present)
- HF cache (default: $HF_HOME or ai/hf_cache)
- Quantonium training outputs (ai/training/models)
- Ollama models (common default: ~/.ollama)

Output
- Writes logs/space_report.json and prints a short summary.

Usage
- python src/apps/space_report.py
- python src/apps/space_report.py --json logs/space_report.json

Notes
- In a Codespace, ~/.ollama probably won't exist unless you installed Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

from atomic_io import atomic_write_json


def _bytes_dir(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            continue
    return total


def _human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(f)} {u}"
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{n} B"


def _default_hf_home() -> Path:
    return Path(os.getenv("HF_HOME", "ai/hf_cache"))


def _default_ollama_home() -> Path:
    # Ollama default on Linux/macOS is ~/.ollama
    return Path(os.path.expanduser(os.getenv("OLLAMA_HOME", "~/.ollama")))


def _maybe_model_param_estimate() -> Optional[Dict[str, object]]:
    """Best-effort: if QUANTONIUM_MODEL_ID is set and transformers can load config, estimate params."""
    model_id = os.getenv("QUANTONIUM_MODEL_ID")
    if not model_id:
        return None
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, local_files_only=(os.getenv("QUANTONIUM_LOCAL_ONLY") == "1"))
        # We cannot reliably compute exact params from config for every architecture without loading weights.
        # Provide a lightweight summary instead.
        return {
            "quantonium_model_id": model_id,
            "architecture": getattr(cfg, "model_type", None),
            "hidden_size": getattr(cfg, "n_embd", getattr(cfg, "hidden_size", None)),
            "n_layers": getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", None)),
            "n_heads": getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", None)),
        }
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default=None, help="HF cache directory to measure (default: $HF_HOME or ai/hf_cache)")
    ap.add_argument("--ollama", default=None, help="Ollama directory to measure (default: ~/.ollama)")
    ap.add_argument("--models", default="ai/training/models", help="Quantonium LoRA/models directory")
    ap.add_argument("--json", default="logs/space_report.json", help="Where to write JSON report")
    args = ap.parse_args()

    hf_dir = Path(args.hf) if args.hf else _default_hf_home()
    ollama_dir = Path(args.ollama) if args.ollama else _default_ollama_home()
    models_dir = Path(args.models)

    report: Dict[str, object] = {
        "hf_cache_dir": str(hf_dir),
        "hf_cache_bytes": _bytes_dir(hf_dir),
        "quantonium_models_dir": str(models_dir),
        "quantonium_models_bytes": _bytes_dir(models_dir),
        "ollama_dir": str(ollama_dir),
        "ollama_bytes": _bytes_dir(ollama_dir),
        "notes": "Bytes are measured by summing file sizes under the directory.",
    }

    model_meta = _maybe_model_param_estimate()
    if model_meta:
        report["model_meta"] = model_meta

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, report, indent=2)

    print("Disk usage:")
    print(f"  HF cache     : {_human(int(report['hf_cache_bytes']))}  ({hf_dir})")
    print(f"  Quantonium   : {_human(int(report['quantonium_models_bytes']))}  ({models_dir})")
    print(f"  Ollama       : {_human(int(report['ollama_bytes']))}  ({ollama_dir})")
    total = int(report["hf_cache_bytes"]) + int(report["quantonium_models_bytes"]) + int(report["ollama_bytes"])
    print(f"  Total        : {_human(total)}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
