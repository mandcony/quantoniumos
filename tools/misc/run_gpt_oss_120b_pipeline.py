#!/usr/bin/env python3
"""Utility to run the GPT-OSS-120B compression pipeline with custom cache/output paths.

This wrapper avoids the hard-coded devcontainer paths inside `add_gpt_oss_120b_to_brain.py`
so we can place the 120B download cache on a spacious drive (e.g. D:) while directing the
compressed artifacts wherever we need them.

⚠️ WARNING ⚠️
Downloading and loading GPT-OSS-120B requires hundreds of gigabytes of disk and many hours
of CPU/GPU time. Make sure you have >800 GB free and plenty of RAM before launching.
"""

import argparse
import os
import tempfile
from pathlib import Path

from gpt_oss_120b_streaming import RealGPTOSS120BCompressor


def run_pipeline(
    cache_root: Path,
    output_root: Path,
    *,
    chunk_elements: int,
    elements_per_state: int,
    max_workers: int,
    use_hf_transfer: bool,
    hf_token: str | None,
) -> None:
    compressor = RealGPTOSS120BCompressor(
        chunk_elements=chunk_elements,
        elements_per_state=elements_per_state,
        max_workers=max_workers,
        use_hf_transfer=use_hf_transfer,
        hf_token=hf_token,
    )

    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"💽 Using cache root: {cache_root}")
    print(f"📦 Outputs will be written under: {output_root.resolve()}")

    with tempfile.TemporaryDirectory(dir=str(cache_root), prefix="gpt_oss_120b_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        model_path = compressor.download_real_model(temp_dir_path)
        compressed_model, total_params, total_states = compressor.compress_with_assembly_rft_streaming(model_path)

        # Save JSON artifact and create an additional gzipped pickle for fast loading
        output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        json_path = compressor.save_compressed_model(compressed_model, total_params, total_states, output_root)

        assembly_dir = output_root / "compressed"
        assembly_dir.mkdir(parents=True, exist_ok=True)
        assembly_file = assembly_dir / "gpt_oss_120b_compressed.pkl.gz"

        import gzip
        import pickle

        with gzip.open(assembly_file, "wb") as handle:
            pickle.dump(compressed_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"💾 Assembly-compressed artifact stored at {assembly_file}")
        print(
            f"📊 JSON size: {json_path.stat().st_size / (1024 * 1024):.2f} MB | "
            f"PKL size: {assembly_file.stat().st_size / (1024 * 1024):.2f} MB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-OSS-120B compression with custom paths")
    parser.add_argument("--cache-root", type=Path, default=Path(os.getenv("QUANTONIUM_GPT_CACHE", "D:/gptoss_cache")),
                        help="Directory on a large drive to hold the Hugging Face snapshot")
    parser.add_argument("--output-root", type=Path, default=Path("ai/models"),
                        help="Where to place compressed artifacts (quantum JSON + optional pickle)")
    parser.add_argument("--chunk-elements", type=int, default=4_000_000,
                        help="How many raw weights to stream before recalculating state summaries")
    parser.add_argument("--elements-per-state", type=int, default=1_000_000,
                        help="Average number of weights represented by each quantum state")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent worker count for snapshot downloads")
    parser.add_argument("--no-hf-transfer", action="store_true",
                        help="Disable hf_transfer acceleration even if available")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Optional Hugging Face token for gated repositories")
    args = parser.parse_args()

    run_pipeline(
        args.cache_root,
        args.output_root,
        chunk_elements=args.chunk_elements,
        elements_per_state=args.elements_per_state,
        max_workers=args.max_workers,
        use_hf_transfer=not args.no_hf_transfer,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
