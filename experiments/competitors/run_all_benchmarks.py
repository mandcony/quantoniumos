#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Run All Competitor Benchmarks
=============================

Master runner for all competitor benchmarks:
1. Transform benchmark (RFT vs FFT/DCT/Wavelets)
2. Compression benchmark (RFTMW+ANS vs zlib/zstd/brotli)
3. Crypto throughput (RFT cipher vs AES-GCM/ChaCha20)

Usage:
    python run_all_benchmarks.py [--preset laptop|desktop|quick]
    python run_all_benchmarks.py --benchmarks transforms,compression
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]


# Benchmark presets
PRESETS = {
    'quick': {
        'description': 'Fast run for CI/testing',
        'transforms': {'sizes': '256,1024', 'runs': 3},
        'compression': {'runs': 1},
        'crypto': {'sizes': '1024,4096', 'runs': 3},
    },
    'laptop': {
        'description': 'Balanced run for laptops',
        'transforms': {'sizes': '256,1024,4096', 'runs': 5},
        'compression': {'runs': 3},
        'crypto': {'sizes': '1024,4096,16384', 'runs': 5},
    },
    'desktop': {
        'description': 'Full run for desktops/servers',
        'transforms': {'sizes': '256,1024,4096,16384,65536', 'runs': 10},
        'compression': {'runs': 5},
        'crypto': {'sizes': '1024,4096,16384,65536,262144', 'runs': 10},
    },
    'full': {
        'description': 'Comprehensive benchmark (slow)',
        'transforms': {'sizes': '128,256,512,1024,2048,4096,8192,16384,32768,65536', 'runs': 20},
        'compression': {'runs': 10},
        'crypto': {'sizes': '256,1024,4096,16384,65536,262144,1048576', 'runs': 20},
    },
}


def get_system_info() -> Dict:
    """Collect system information."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'system': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
    }
    
    try:
        import psutil
        info['cpu_count_physical'] = psutil.cpu_count(logical=False)
        info['cpu_count_logical'] = psutil.cpu_count(logical=True)
        info['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info['cpu_count_logical'] = os.cpu_count()
    
    return info


def run_benchmark(
    script: str,
    args: List[str],
    output_dir: Path,
    name: str
) -> bool:
    """Run a benchmark script and capture output."""
    script_path = _HERE / script
    
    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)] + args + ['--output-dir', str(output_dir)]
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            capture_output=False,  # Show output in real-time
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def generate_final_report(
    output_dir: Path,
    system_info: Dict,
    benchmark_results: Dict[str, bool]
) -> Path:
    """Generate combined final report."""
    report_path = output_dir / f"competitor_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    lines = [
        "# QuantoniumOS Competitor Benchmark Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## System Information",
        "",
        f"- **Platform:** {system_info.get('platform', 'N/A')}",
        f"- **Processor:** {system_info.get('processor', 'N/A')}",
        f"- **CPU Cores:** {system_info.get('cpu_count_logical', 'N/A')}",
        f"- **Memory:** {system_info.get('memory_gb', 'N/A')} GB",
        f"- **Python:** {system_info.get('python_version', 'N/A')}",
        "",
        "## Benchmark Status",
        "",
        "| Benchmark | Status |",
        "|-----------|--------|",
    ]
    
    for name, success in benchmark_results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        lines.append(f"| {name} | {status} |")
    
    lines.extend([
        "",
        "## Results Files",
        "",
        "Detailed results are available in the following files:",
        "",
    ])
    
    # List all result files
    for f in sorted(output_dir.glob("*.csv")) + sorted(output_dir.glob("*.json")) + sorted(output_dir.glob("*.md")):
        if f.name != report_path.name:
            lines.append(f"- `{f.name}`")
    
    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "### Transform Benchmark",
        "- **Round-trip error:** Should be < 1e-10 for lossless transforms",
        "- **Speed:** Lower µs is better; compare RFT vs FFT",
        "- **Sparsity:** Lower ratio indicates sparser representation",
        "",
        "### Compression Benchmark",
        "- **Entropy gap:** R - H(X), lower is better (0 is Shannon limit)",
        "- **Throughput:** MB/s for encode/decode",
        "- **Ratio:** Higher compression ratio is better",
        "",
        "### Crypto Benchmark",
        "- **⚠️ NOT A SECURITY COMPARISON** - throughput only!",
        "- **Avalanche:** Should be ~50% for good diffusion",
        "- **Throughput:** MB/s for encrypt/decrypt",
        "",
        "## Disclaimer",
        "",
        "This benchmark suite is designed for performance comparison only.",
        "The RFT cipher is a research tool and should NOT be used for production security.",
    ])
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Run all QuantoniumOS competitor benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  quick     Fast run for CI (~1 minute)
  laptop    Balanced run (~5 minutes)
  desktop   Full run (~15 minutes)
  full      Comprehensive (~30+ minutes)

Examples:
  python run_all_benchmarks.py --preset laptop
  python run_all_benchmarks.py --benchmarks transforms,compression
  python run_all_benchmarks.py --preset quick --benchmarks crypto
        """
    )
    
    parser.add_argument(
        '--preset', '-p',
        choices=list(PRESETS.keys()),
        default='laptop',
        help='Benchmark preset (default: laptop)'
    )
    parser.add_argument(
        '--benchmarks', '-b',
        type=str,
        default='transforms,compression,crypto',
        help='Comma-separated list of benchmarks to run'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=_PROJECT_ROOT.parent / 'results' / 'competitors',
        help='Output directory for results'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available presets'
    )
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available presets:")
        for name, config in PRESETS.items():
            print(f"  {name:10} - {config['description']}")
        return
    
    preset = PRESETS[args.preset]
    benchmarks = [b.strip() for b in args.benchmarks.split(',')]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("QUANTONIUMOS COMPETITOR BENCHMARKS")
    print("=" * 70)
    print(f"Preset: {args.preset} - {preset['description']}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Output: {output_dir}")
    print()
    
    # Collect system info
    print("Collecting system information...")
    system_info = get_system_info()
    print(f"  Platform: {system_info.get('platform', 'N/A')}")
    print(f"  Python: {system_info.get('python_version', 'N/A')}")
    
    # Run benchmarks
    results = {}
    
    if 'transforms' in benchmarks:
        tf_args = [
            '--sizes', preset['transforms']['sizes'],
            '--runs', str(preset['transforms']['runs']),
        ]
        results['transforms'] = run_benchmark(
            'benchmark_transforms_vs_fft.py',
            tf_args,
            output_dir,
            'Transform Benchmark (RFT vs FFT/DCT)'
        )
    
    if 'compression' in benchmarks:
        comp_args = [
            '--runs', str(preset['compression']['runs']),
        ]
        results['compression'] = run_benchmark(
            'benchmark_compression_vs_codecs.py',
            comp_args,
            output_dir,
            'Compression Benchmark (RFTMW vs zlib/zstd/brotli)'
        )
    
    if 'crypto' in benchmarks:
        crypto_args = [
            '--sizes', preset['crypto']['sizes'],
            '--runs', str(preset['crypto']['runs']),
        ]
        results['crypto'] = run_benchmark(
            'benchmark_crypto_throughput.py',
            crypto_args,
            output_dir,
            'Crypto Benchmark (RFT cipher vs AES/ChaCha)'
        )
    
    # Generate final report
    report_path = generate_final_report(output_dir, system_info, results)
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    all_passed = all(results.values())
    print(f"Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    print()
    
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print(f"\nFinal report: {report_path}")
    print(f"All results in: {output_dir}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
