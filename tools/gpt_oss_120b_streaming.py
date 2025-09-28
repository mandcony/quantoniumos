#!/usr/bin/env python3
"""Streaming-aware installer and compressor for the GPT-OSS-120B model.

This module focuses on three goals:

1. **Real model installation** – pulls the official ``openai/gpt-oss-120b``
   snapshot using resumable downloads and optional `hf_transfer` acceleration.
2. **Disk-friendly caching** – allows the caller to select large cache/output
   roots so the 120B snapshot can live outside the system drive.
3. **Fast compression** – iterates shards lazily via ``safetensors`` and emits
   quantum-compressed state summaries without ever instantiating the full model
   in RAM.

The resulting JSON structure mirrors the existing quantum-encoded models used
inside QuantoniumOS, so downstream tooling (model router, chatbox, etc.) can
load the compressed artifact without changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Install huggingface_hub to download GPT-OSS-120B: pip install huggingface_hub"
    ) from exc

try:
    from safetensors.numpy import safe_open
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Install safetensors for streaming weight access: pip install safetensors"
    ) from exc

PHI = (1 + math.sqrt(5.0)) / 2.0
STATE_EPS = 1e-9


@dataclass
class CompressionStats:
    total_params: int = 0
    total_states: int = 0
    layers_processed: int = 0
    start_time: float = time.perf_counter()

    def summary(self) -> str:
        elapsed = time.perf_counter() - self.start_time
        if self.total_states == 0:
            ratio = 0.0
        else:
            ratio = self.total_params / self.total_states
        return (
            f"Layers: {self.layers_processed}, States: {self.total_states:,}, "
            f"Params: {self.total_params:,}, Ratio≈{ratio:,.1f}:1, "
            f"Elapsed: {elapsed/3600:.2f} h"
        )


class RealGPTOSS120BCompressor:
    """Download and compress GPT-OSS-120B with streaming-friendly logic."""

    def __init__(
        self,
        *,
        allow_patterns: Optional[Iterable[str]] = None,
        max_workers: int = 8,
        chunk_elements: int = 4_000_000,
        elements_per_state: int = 1_000_000,
        use_hf_transfer: bool = True,
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_id = "openai/gpt-oss-120b"
        self.model_name = "gpt-oss-120b"
        self.max_workers = max_workers
        self.chunk_elements = chunk_elements
        self.elements_per_state = max(1, elements_per_state)
        self.use_hf_transfer = use_hf_transfer
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.allow_patterns = list(allow_patterns) if allow_patterns is not None else [
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.py",
            "tokenizer.*",
            "merges.txt",
            "vocab.json",
        ]
        self._state_counter = 0

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------
    def download_real_model(self, cache_dir: Path) -> Path:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.use_hf_transfer:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

        print("🚀 Downloading GPT-OSS-120B snapshot")
        print(f"📁 Cache directory: {cache_dir}")
        print(f"⚙️ Using up to {self.max_workers} download workers")

        snapshot_kwargs = dict(
            repo_id=self.model_id,
            cache_dir=str(cache_dir),
            allow_patterns=self.allow_patterns,
            resume_download=True,
            max_workers=self.max_workers,
            local_dir_use_symlinks=False,
        )
        if self.hf_token:
            snapshot_kwargs["token"] = self.hf_token

        snapshot_path = Path(snapshot_download(**snapshot_kwargs))
        print(f"✅ Snapshot ready at {snapshot_path}")
        return snapshot_path

    # ------------------------------------------------------------------
    # Streaming compression logic
    # ------------------------------------------------------------------
    def _iter_shard_weights(self, model_path: Path) -> Iterator[Tuple[str, np.ndarray]]:
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with index_file.open("r", encoding="utf-8") as handle:
                index_data = json.load(handle)
            weight_map: Dict[str, str] = index_data.get("weight_map", {})
            files_to_keys: Dict[str, List[str]] = {}
            for weight, shard_name in weight_map.items():
                files_to_keys.setdefault(shard_name, []).append(weight)

            for shard_name in sorted(files_to_keys.keys()):
                shard_path = model_path / shard_name
                if not shard_path.exists():
                    continue
                with safe_open(shard_path, framework="np") as shard:
                    for key in sorted(files_to_keys[shard_name]):
                        yield key, shard.get_tensor(key)
        else:
            for shard_path in sorted(model_path.glob("*.safetensors")):
                with safe_open(shard_path, framework="np") as shard:
                    for key in shard.keys():
                        yield key, shard.get_tensor(key)

    def _encode_chunk(
        self,
        layer_name: str,
        chunk: np.ndarray,
        chunk_index: int,
        stats: CompressionStats,
    ) -> Iterator[Dict[str, float]]:
        chunk = chunk.astype(np.float64, copy=False)
        num_states = max(1, math.ceil(chunk.size / self.elements_per_state))
        stride = max(1, chunk.size // num_states)

        for local_idx in range(num_states):
            start = local_idx * stride
            end = chunk.size if local_idx == num_states - 1 else min(chunk.size, start + stride)
            sub = chunk[start:end]
            if sub.size == 0:
                continue

            mean_val = float(sub.mean())
            std_val = float(sub.std())
            max_val = float(sub.max())
            min_val = float(sub.min())
            amplitude = math.sqrt(mean_val * mean_val + std_val * std_val + STATE_EPS)
            phase = (stats.total_states + local_idx + 1) * (PHI / 128.0)
            vertex = [
                amplitude * math.cos(phase),
                amplitude * math.sin(phase),
                (max_val - min_val) / (std_val + STATE_EPS),
            ]
            resonance = PHI * (chunk_index + 1) * (local_idx + 1)
            entanglement_key = hashlib.sha256(
                f"{layer_name}:{chunk_index}:{local_idx}:{stats.total_states}".encode("utf-8")
            ).hexdigest()[:16]

            state = {
                "id": self._state_counter,
                "layer": layer_name,
                "chunk_index": chunk_index,
                "local_index": local_idx,
                "weight_mean": mean_val,
                "weight_std": std_val,
                "min": min_val,
                "max": max_val,
                "amplitude": amplitude,
                "phase": phase,
                "vertex": vertex,
                "resonance": resonance,
                "count": int(sub.size),
                "encoding": "assembly_rft_streaming_v2",
                "entanglement_key": entanglement_key,
            }
            self._state_counter += 1
            stats.total_states += 1
            yield state

    def compress_with_assembly_rft_streaming(self, model_path: Path) -> Tuple[Dict, int, int]:
        stats = CompressionStats()
        quantum_states: List[Dict] = []

        for layer_name, tensor in self._iter_shard_weights(model_path):
            flat = tensor.reshape(-1)
            stats.total_params += flat.size
            stats.layers_processed += 1

            for chunk_index, start in enumerate(range(0, flat.size, self.chunk_elements)):
                chunk = flat[start : start + self.chunk_elements]
                quantum_states.extend(self._encode_chunk(layer_name, chunk, chunk_index, stats))

            if stats.layers_processed % 25 == 0:
                print(f"📈 Compression progress: {stats.summary()}")

        total_params = stats.total_params
        total_states = stats.total_states
        compression_ratio = (total_params / total_states) if total_states else 0.0

        compressed_model = {
            "metadata": {
                "model_name": self.model_name,
                "model_id": self.model_id,
                "created_at": time.time(),
                "phi": PHI,
                "chunk_elements": self.chunk_elements,
                "elements_per_state": self.elements_per_state,
                "compression_method": "assembly_rft_streaming_v2",
                "total_parameters": total_params,
                "total_states": total_states,
                "compression_ratio": compression_ratio,
            },
            "quantum_states": quantum_states,
        }

        print("✅ Compression complete")
        print(f"📊 Parameters: {total_params:,}")
        print(f"⚛️ States: {total_states:,}")
        print(f"🗜️ Ratio ≈ {compression_ratio:,.2f}:1")

        return compressed_model, total_params, total_states

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_compressed_model(
        self,
        compressed_model: Dict,
        total_params: int,
        total_states: int,
        output_root: Path,
    ) -> Path:
        output_root = Path(output_root)
        quantum_dir = output_root / "quantum"
        quantum_dir.mkdir(parents=True, exist_ok=True)

        output_path = quantum_dir / "gpt_oss_120b_real_quantum_compressed.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(compressed_model, handle, indent=2)

        print(f"💾 Saved compressed model to {output_path}")
        print(
            f"📦 Summary -> params: {total_params:,}, states: {total_states:,}, "
            f"ratio≈{total_params / max(1, total_states):,.1f}:1"
        )
        return output_path

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------
    def run(
        self,
        cache_dir: Path,
        output_root: Path,
    ) -> Path:
        snapshot_path = self.download_real_model(cache_dir)
        compressed_model, total_params, total_states = self.compress_with_assembly_rft_streaming(snapshot_path)
        return self.save_compressed_model(compressed_model, total_params, total_states, output_root)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download + compress GPT-OSS-120B with streaming optimisations")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.getenv("QUANTONIUM_GPT_CACHE", "D:/gptoss_cache")),
                        help="Directory to store the raw Hugging Face snapshot")
    parser.add_argument("--output-dir", type=Path, default=Path("ai/models"),
                        help="Directory where compressed artifacts will be saved")
    parser.add_argument("--chunk-elements", type=int, default=4_000_000,
                        help="Number of weights processed per chunk before yielding states")
    parser.add_argument("--elements-per-state", type=int, default=1_000_000,
                        help="Average number of weights represented by a single quantum state")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent download worker count for snapshot_download")
    parser.add_argument("--no-hf-transfer", action="store_true",
                        help="Disable hf_transfer acceleration even if installed")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Optional Hugging Face token for gated access")
    return parser


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = build_arg_parser()
    args = parser.parse_args()

    compressor = RealGPTOSS120BCompressor(
        max_workers=args.max_workers,
        chunk_elements=args.chunk_elements,
        elements_per_state=args.elements_per_state,
        use_hf_transfer=not args.no_hf_transfer,
        hf_token=args.hf_token,
    )

    compressor.run(args.cache_dir, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
