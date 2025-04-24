"""
QuantoniumOS - Resonance Metrics

This module implements the Resonance Metrics for QuantoniumOS:
- Resonance Fourier Transform (RFT)
- Symbolic Avalanche Metric
- Waveform Coherence calculation
- 64-Perturbation Benchmark (reproducible avalanche tests)

All proprietary algorithms are securely delegated to the core engine.
"""

import os
import csv
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from secure_core.python_bindings import engine_core
from encryption.resonance_encrypt import encrypt_symbolic

logger = logging.getLogger("quantonium_api.metrics")

def compute_rft(payload: str) -> Dict[str, Any]:
    """
    Compute the Resonance Fourier Transform for input payload
    
    Args:
        payload: Input string to analyze
        
    Returns:
        Dictionary with RFT bins, harmonic ratio, and metrics
    """
    try:
        # Convert payload string to bytes for core engine
        if isinstance(payload, str):
            bytes_data = payload.encode('utf-8')
        else:
            bytes_data = bytes(payload)
            
        # Call into the secure core engine to compute RFT
        bins, hr = engine_core.rft_run(bytes_data)
        
        # Calculate harmonic peaks (ratio of max to mean)
        if bins and len(bins) > 0:
            max_value = max(bins)
            mean_value = sum(bins) / len(bins)
            harmonic_peak_ratio = max_value / mean_value if mean_value > 0 else 0
        else:
            harmonic_peak_ratio = 0
        
        # Return complete result
        return {
            "rft": bins,
            "hr": hr,
            "bin_count": len(bins),
            "harmonic_peak_ratio": harmonic_peak_ratio
        }
        
    except Exception as e:
        logger.error(f"Error computing RFT: {str(e)}")
        # In production, we don't want to expose internal errors
        if os.environ.get("FASTAPI_ENV") == "development":
            raise
        # Return a minimal result indicating failure
        return {
            "rft": None,
            "hr": 0.0,
            "error": "Failed to compute RFT"
        }

def compute_avalanche(plaintext: str, key: str, perturbation_count: int = 64) -> List[Dict[str, Any]]:
    """
    Compute the Symbolic Avalanche metric for a given plaintext/key pair
    
    This measures how small changes in input produce large changes in output,
    a key property of cryptographic algorithms.
    
    Args:
        plaintext: Original plaintext (hex)
        key: Original key (hex)
        perturbation_count: Number of perturbations to test (default: 64)
        
    Returns:
        List of dictionaries containing test results for each perturbation
    """
    results = []
    
    # First, encrypt the original plaintext/key pair to get baseline
    baseline = encrypt_symbolic(plaintext, key)
    baseline_hr = baseline["hr"]
    baseline_wc = baseline["wc"]
    
    # Helper to flip a bit at position
    def flip_bit(hex_str: str, bit_pos: int) -> str:
        # Convert to binary
        bin_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)
        # Flip the bit
        bin_list = list(bin_str)
        bin_list[bit_pos] = '1' if bin_list[bit_pos] == '0' else '0'
        # Convert back to hex
        return hex(int(''.join(bin_list), 2))[2:].zfill(len(hex_str))
    
    # Test perturbations
    for i in range(min(perturbation_count, 128)):  # Max 128 bits in each input (plaintext/key)
        # Determine if we're perturbing plaintext or key
        if i < 64:
            # Perturb plaintext
            perturbed_pt = flip_bit(plaintext, i)
            perturbed_key = key
            perturb_type = "plaintext"
            bit_pos = i
        else:
            # Perturb key
            perturbed_pt = plaintext
            perturbed_key = flip_bit(key, i - 64)
            perturb_type = "key"
            bit_pos = i - 64
        
        # Encrypt with perturbed input
        result = encrypt_symbolic(perturbed_pt, perturbed_key)
        
        # Calculate deltas
        delta_hr = abs(result["hr"] - baseline_hr)
        delta_wc = abs(result["wc"] - baseline_wc)
        
        # Store result
        results.append({
            "test_id": i,
            "bit_pos": bit_pos,
            "perturb_type": perturb_type,
            "hr": result["hr"],
            "wc": result["wc"],
            "sa": result.get("sa", 0.0),
            "entropy": result.get("entropy", 0.0),
            "delta_hr": delta_hr,
            "delta_wc": delta_wc,
            "signature": result.get("signature", "")
        })
    
    return results

def calculate_waveform_coherence(waveform1: List[float], waveform2: List[float]) -> float:
    """
    Calculate the Waveform Coherence between two waveforms
    
    This measures the similarity between two waveforms based on their
    phase and amplitude alignment.
    
    Args:
        waveform1: First waveform as list of float values
        waveform2: Second waveform as list of float values
        
    Returns:
        Coherence value between 0.0 and 1.0
    """
    # Ensure equal length
    min_len = min(len(waveform1), len(waveform2))
    wf1 = waveform1[:min_len]
    wf2 = waveform2[:min_len]
    
    # Convert to numpy arrays for vector operations
    arr1 = np.array(wf1)
    arr2 = np.array(wf2)
    
    # Calculate coherence
    # Normalized dot product measure
    dot_product = np.sum(arr1 * arr2)
    norm1 = np.sqrt(np.sum(arr1 * arr1))
    norm2 = np.sqrt(np.sum(arr2 * arr2))
    
    if norm1 > 0 and norm2 > 0:
        coherence = abs(dot_product) / (norm1 * norm2)
    else:
        coherence = 0.0
    
    return float(coherence)
    
BITS = 128

def _flip_bit(hex_str: str, bit_pos: int) -> str:
    """
    Flips a specific bit in a hex string
    
    Args:
        hex_str: Hex string to modify
        bit_pos: Bit position to flip (0-indexed)
        
    Returns:
        Modified hex string
    """
    b = bytearray.fromhex(hex_str)
    byte_i, offset = divmod(bit_pos, 8)
    b[byte_i] ^= 1 << (7-offset)
    return b.hex()

def run_symbolic_benchmark(base_pt: str, base_key: str) -> Tuple[str, Dict[str, Any]]:
    """
    Executes the 64-vector avalanche test on the server side
    
    Args:
        base_pt: Base plaintext to use
        base_key: Base key to use
        
    Returns:
        Tuple of (csv_path, summary_dict)
    """
    # Ensure inputs are valid hex (32 chars)
    def validate_hex(input_str: str, default: str = None) -> str:
        """Validate and fix hex input"""
        if not input_str:
            return default or "0" * 32
        
        # Keep only hex characters
        hex_only = ''.join(c for c in input_str if c in '0123456789abcdefABCDEF')
        
        # Pad or truncate
        if len(hex_only) < 32:
            return hex_only.ljust(32, '0')
        return hex_only[:32]
    
    # Validate inputs for benchmark (don't change originals)
    base_pt = validate_hex(base_pt)
    base_key = validate_hex(base_key)
    tstamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = Path("logs") / f"benchmark_{tstamp}.csv"
    csv_path.parent.mkdir(exist_ok=True)
    
    max_wc_delta = 0.0
    max_hr_delta = 0.0
    
    with csv_path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["TestID", "Vector", "BitPos", "PT", "KEY", "HR", "WC", "Entropy"])
        
        # Helper to encrypt once and write
        def _process(idx, pt, key, label, bitpos):
            out = encrypt_symbolic(pt, key)
            w.writerow([idx, label, bitpos, pt, key, out["hr"], out["wc"], out.get("entropy", 0.0)])
            return out["hr"], out["wc"]
        
        # 0) base
        base_hr, base_wc = _process(0, base_pt, base_key, "base", -1)
        
        row = 1
        # 32 plaintext flips (even bits 0-124)
        for pos in range(0, BITS, 4):
            hr, wc = _process(row, _flip_bit(base_pt, pos), base_key, "pt_flip", pos)
            max_wc_delta = max(max_wc_delta, abs(wc-base_wc))
            max_hr_delta = max(max_hr_delta, abs(hr-base_hr))
            row += 1
        
        # 31 key flips (odd bits 1-125)
        for pos in range(1, BITS, 4):
            hr, wc = _process(row, base_pt, _flip_bit(base_key, pos), "key_flip", pos)
            max_wc_delta = max(max_wc_delta, abs(wc-base_wc))
            max_hr_delta = max(max_hr_delta, abs(hr-base_hr))
            row += 1
    
    # Calculate SHA-256 of the CSV for integrity verification
    import hashlib
    csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    
    return str(csv_path), {
        "rows": row,
        "max_wc_delta": round(max_wc_delta, 3),
        "max_hr_delta": round(max_hr_delta, 3),
        "sha256": csv_hash
    }