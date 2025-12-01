#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Shannon Entropy Measurement
===========================

Compute Shannon entropy H(X) = -Σ p(x) log₂ p(x) for various data sources.
Core utility for entropy gap analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Add project root to path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.entropy.datasets import load_dataset, list_datasets


def estimate_entropy(
    symbols: np.ndarray,
    alphabet_size: Optional[int] = None,
    base: float = 2.0,
    method: str = 'plugin'
) -> float:
    """
    Estimate Shannon entropy of a symbol sequence.
    
    H(X) = -Σ p(x) log(p(x))
    
    Args:
        symbols: 1D array of discrete symbols (integers)
        alphabet_size: Size of alphabet (auto-detected if None)
        base: Logarithm base (2 for bits, e for nats)
        method: 'plugin' (MLE), 'miller_madow', or 'grassberger'
    
    Returns:
        Entropy in bits (or nats if base=e)
    """
    symbols = np.asarray(symbols).ravel()
    n = len(symbols)
    if n == 0:
        return 0.0
    
    # Count symbol occurrences
    unique, counts = np.unique(symbols, return_counts=True)
    k_obs = len(unique)  # Number of observed symbols
    
    if alphabet_size is None:
        alphabet_size = k_obs
    
    # Probability estimates
    probs = counts / n
    
    # Entropy with correction for zero probabilities
    # H = -Σ p log p (skip p=0 terms)
    nonzero = probs > 0
    H = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
    
    # Apply bias correction if requested
    if method == 'miller_madow':
        # Miller-Madow correction: H_mm = H_plugin + (k-1)/(2n)
        H += (k_obs - 1) / (2 * n)
    
    elif method == 'grassberger':
        # Grassberger correction (more accurate for small samples)
        # Uses digamma function correction
        from scipy.special import digamma
        H = np.log(n) - np.sum(counts * digamma(counts + 1)) / n
        if base != np.e:
            H /= np.log(base)
        return float(H)
    
    # Convert to desired base
    if base != np.e:
        H /= np.log(base)
    
    return float(H)


def entropy_per_symbol(data: np.ndarray, quantization_bits: int = 8) -> Dict[str, float]:
    """
    Compute entropy metrics for continuous or quantized data.
    
    Args:
        data: Input data (will be quantized)
        quantization_bits: Number of bits for quantization
    
    Returns:
        Dictionary with entropy metrics
    """
    data = np.asarray(data).ravel()
    n = len(data)
    
    # Quantize to discrete symbols
    if data.dtype == np.uint8:
        symbols = data
        alphabet_size = 256
    else:
        # Normalize and quantize
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)
        
        n_levels = 2 ** quantization_bits
        symbols = np.clip(normalized * (n_levels - 1), 0, n_levels - 1).astype(np.int64)
        alphabet_size = n_levels
    
    # Compute entropy with different estimators
    H_plugin = estimate_entropy(symbols, alphabet_size, method='plugin')
    H_mm = estimate_entropy(symbols, alphabet_size, method='miller_madow')
    
    # Theoretical maximum entropy
    H_max = np.log2(alphabet_size)
    
    # Redundancy
    redundancy = 1.0 - H_plugin / H_max if H_max > 0 else 0.0
    
    return {
        'H_plugin': H_plugin,
        'H_miller_madow': H_mm,
        'H_max': H_max,
        'redundancy': redundancy,
        'n_symbols': n,
        'n_unique': len(np.unique(symbols)),
        'alphabet_size': alphabet_size,
        'quantization_bits': quantization_bits,
    }


def measure_dataset_entropy(
    dataset_name: str,
    block_size: int = 256,
    quantization_bits: int = 8,
    max_blocks: int = 100
) -> Dict[str, Any]:
    """
    Measure entropy of an entire dataset.
    
    Returns:
        Dictionary with per-block and aggregate entropy statistics
    """
    data = load_dataset(dataset_name, block_size=block_size, max_blocks=max_blocks)
    
    # Per-block entropy
    block_entropies = []
    for i in range(len(data)):
        block = data[i]
        metrics = entropy_per_symbol(block, quantization_bits)
        block_entropies.append(metrics['H_plugin'])
    
    block_entropies = np.array(block_entropies)
    
    # Aggregate over all data
    aggregate = entropy_per_symbol(data.ravel(), quantization_bits)
    
    return {
        'dataset': dataset_name,
        'block_size': block_size,
        'n_blocks': len(data),
        'quantization_bits': quantization_bits,
        'aggregate': aggregate,
        'per_block': {
            'mean': float(block_entropies.mean()),
            'std': float(block_entropies.std()),
            'min': float(block_entropies.min()),
            'max': float(block_entropies.max()),
            'values': block_entropies.tolist(),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Measure Shannon entropy of datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure entropy of ASCII corpus
  python measure_entropy.py --dataset ascii --block-size 256

  # Measure entropy of audio with 12-bit quantization
  python measure_entropy.py --dataset audio --quantization-bits 12

  # Measure all datasets
  python measure_entropy.py --all

  # Save results to JSON
  python measure_entropy.py --dataset golden --output results.json
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        choices=list_datasets(),
        help='Dataset to measure'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Measure all datasets'
    )
    parser.add_argument(
        '--block-size', '-b',
        type=int,
        default=256,
        help='Block size for segmentation (default: 256)'
    )
    parser.add_argument(
        '--quantization-bits', '-q',
        type=int,
        default=8,
        help='Quantization bits (default: 8)'
    )
    parser.add_argument(
        '--max-blocks', '-n',
        type=int,
        default=100,
        help='Maximum blocks to process (default: 100)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output JSON file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        parser.error("Specify --dataset or --all")
    
    datasets = list_datasets() if args.all else [args.dataset]
    results = {}
    
    for ds in datasets:
        print(f"Measuring entropy: {ds}...")
        try:
            result = measure_dataset_entropy(
                ds,
                block_size=args.block_size,
                quantization_bits=args.quantization_bits,
                max_blocks=args.max_blocks
            )
            results[ds] = result
            
            # Print summary
            agg = result['aggregate']
            per = result['per_block']
            print(f"  H(X) = {agg['H_plugin']:.4f} bits/symbol (aggregate)")
            print(f"  Per-block: mean={per['mean']:.4f}, std={per['std']:.4f}")
            print(f"  Redundancy: {agg['redundancy']*100:.1f}%")
            print(f"  Max entropy: {agg['H_max']:.4f} bits (for {agg['alphabet_size']} symbols)")
            
            if args.verbose:
                print(f"  Unique symbols: {agg['n_unique']} / {agg['alphabet_size']}")
                print(f"  Total symbols: {agg['n_symbols']}")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[ds] = {'error': str(e)}
    
    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
