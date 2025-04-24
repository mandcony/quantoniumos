"""
QuantoniumOS - Resonance Manager

This module orchestrates the resonance-based operations for the patent validation system.
It manages benchmark tests and maintains state related to resonance metrics.
"""

import os
import time
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Will import these when modules are defined
# from encryption.resonance_encrypt import encrypt_symbolic
# from encryption.geometric_waveform_hash import wave_hash

logger = logging.getLogger("quantonium_api.manager")

# Store recent hash values for replay protection
MAX_HASH_HISTORY = 64
hash_history = []

def run_benchmark(plaintext_seed: str = None, key_seed: str = None) -> List[Dict[str, Any]]:
    """
    Run a 64-test perturbation suite for symbolic avalanche testing
    
    Args:
        plaintext_seed: Optional seed for generating plaintext
        key_seed: Optional seed for generating key
        
    Returns:
        List of 64 test results with metrics
    """
    results = []
    
    # Generate base plaintext and key if not provided
    if not plaintext_seed:
        plaintext_seed = hashlib.sha256(str(time.time()).encode()).hexdigest()
    if not key_seed:
        key_seed = hashlib.sha256((plaintext_seed + str(time.time())).encode()).hexdigest()
    
    # Use first 32 chars of each seed as base plaintext and key
    base_plaintext = plaintext_seed[:32]
    base_key = key_seed[:32]
    
    # Get baseline metrics
    baseline = encrypt_symbolic(base_plaintext, base_key)
    baseline_wc = baseline["wc"]
    baseline_hr = baseline["hr"]
    
    # Helper to flip a bit at position
    def flip_bit(hex_str: str, bit_pos: int) -> str:
        # Convert to binary
        bin_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)
        # Flip the bit
        bin_list = list(bin_str)
        bin_list[bit_pos] = '1' if bin_list[bit_pos] == '0' else '0'
        # Convert back to hex
        return hex(int(''.join(bin_list), 2))[2:].zfill(len(hex_str))
    
    # Run 64 perturbation tests
    for i in range(64):
        # Determine if we're perturbing plaintext or key
        if i < 32:
            # Perturb plaintext
            perturbed_pt = flip_bit(base_plaintext, i)
            perturbed_key = base_key
            perturb_type = "plaintext"
            bit_pos = i
        else:
            # Perturb key
            perturbed_pt = base_plaintext
            perturbed_key = flip_bit(base_key, i - 32)
            perturb_type = "key"
            bit_pos = i - 32
        
        # Encrypt with perturbed input
        result = encrypt_symbolic(perturbed_pt, perturbed_key)
        
        # Calculate deltas
        delta_hr = abs(result["hr"] - baseline_hr)
        delta_wc = abs(result["wc"] - baseline_wc)
        
        # Get timestamp
        timestamp = datetime.now().isoformat()
        
        # Create test record
        test_record = {
            "test_id": i,
            "bit_pos": bit_pos,
            "perturb_type": perturb_type,
            "hr": result["hr"],
            "wc": result["wc"],
            "sa": result.get("sa", 0.0),  # SA may not always be present
            "entropy": result.get("entropy", 0.0),
            "delta_hr": delta_hr,
            "delta_wc": delta_wc,
            "timestamp": timestamp,
            "signature": result.get("signature", "")
        }
        
        results.append(test_record)
        
    # Also write to CSV log file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{log_dir}/benchmark_{timestamp}.csv"
    
    with open(csv_path, 'w') as f:
        # Write header
        f.write("test_id,bit_pos,perturb_type,hr,wc,sa,entropy,signature\n")
        
        # Write data rows
        for r in results:
            f.write(f"{r['test_id']},{r['bit_pos']},{r['perturb_type']},{r['hr']:.6f},{r['wc']:.6f},{r['sa']:.6f},{r['entropy']:.6f},{r['signature']}\n")
    
    logger.info(f"Benchmark data written to {csv_path}")
    
    return results

def add_to_hash_history(hash_value: str):
    """
    Add a hash to the history for replay protection
    
    Args:
        hash_value: Hash to add
    """
    global hash_history
    
    hash_history.append(hash_value)
    
    # Trim if necessary
    if len(hash_history) > MAX_HASH_HISTORY:
        hash_history = hash_history[-MAX_HASH_HISTORY:]

def check_hash_replay(hash_value: str) -> bool:
    """
    Check if a hash has been seen before (replay protection)
    
    Args:
        hash_value: Hash to check
        
    Returns:
        True if hash is new, False if it's a replay
    """
    return hash_value not in hash_history

def validate_avalanche(results: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate if the avalanche test results meet the requirements
    
    According to the patent claims:
    - ≥ 70% rows have WC < 0.55
    - At least one row has WC < 0.05 (symbolic avalanche)
    
    Args:
        results: List of benchmark test results
        
    Returns:
        Tuple of (passed, stats) where stats contains detailed validation info
    """
    # Count rows with WC < 0.55
    low_wc_count = sum(1 for r in results if r["wc"] < 0.55)
    low_wc_percent = (low_wc_count / len(results)) * 100
    
    # Check for at least one row with WC < 0.05
    has_avalanche = any(r["wc"] < 0.05 for r in results)
    
    # Validation result
    passed = (low_wc_percent >= 70.0) and has_avalanche
    
    stats = {
        "low_wc_count": low_wc_count,
        "low_wc_percent": low_wc_percent,
        "has_avalanche": has_avalanche,
        "passed": passed
    }
    
    return passed, stats