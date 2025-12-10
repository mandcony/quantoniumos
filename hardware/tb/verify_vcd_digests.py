#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
VCD Digest Verification Script
==============================
Parses VCD waveform outputs and compares to Python golden reference.
Validates numerical accuracy of the hardware Φ-RFT implementation.
"""

import sys
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Use operator-based canonical RFT (matches hardware kernel ROM)
from hardware.tb.rftpu_test_vectors import python_rft_forward, BLOCK_SAMPLES


@dataclass
class VCDSignalChange:
    """Single signal value change"""
    time: int
    value: str  # Binary string


@dataclass 
class DigestCapture:
    """Captured digest from simulation"""
    test_idx: int
    time_ns: int
    digest_hex: str
    resonance: bool
    input_samples: List[int]


def parse_vcd_simple(vcd_path: Path) -> Dict[str, List[VCDSignalChange]]:
    """
    Simple VCD parser - extracts signal changes for key signals.
    """
    signals = {}
    signal_map = {}  # shorthand -> signal name
    current_time = 0
    
    with open(vcd_path, 'r') as f:
        in_definitions = True
        
        for line in f:
            line = line.strip()
            
            if line == '$enddefinitions $end':
                in_definitions = False
                continue
            
            if in_definitions:
                # Parse signal definitions: $var wire WIDTH SHORTHAND NAME [WIDTH] $end
                match = re.match(r'\$var\s+\w+\s+(\d+)\s+(\S+)\s+(\S+)', line)
                if match:
                    width, shorthand, name = match.groups()
                    # Clean name
                    name = name.split()[0]
                    signal_map[shorthand] = name
                    signals[name] = []
            else:
                # Time change
                if line.startswith('#'):
                    current_time = int(line[1:])
                # Binary value change: bVALUE SHORTHAND
                elif line.startswith('b'):
                    parts = line.split()
                    if len(parts) >= 2:
                        value = parts[0][1:]  # Remove 'b' prefix
                        shorthand = parts[1]
                        if shorthand in signal_map:
                            name = signal_map[shorthand]
                            signals[name].append(VCDSignalChange(current_time, value))
                # Single-bit: 0/1 followed by shorthand
                elif line and line[0] in '01xXzZ':
                    value = line[0]
                    shorthand = line[1:]
                    if shorthand in signal_map:
                        name = signal_map[shorthand]
                        signals[name].append(VCDSignalChange(current_time, value))
    
    return signals


def extract_digest_captures(signals: Dict[str, List[VCDSignalChange]]) -> List[DigestCapture]:
    """
    Extract digest values when digest_valid goes high.
    """
    captures = []
    
    digest_valid_changes = signals.get('digest_valid', [])
    digest_changes = signals.get('digest', [])
    resonance_changes = signals.get('resonance_flag', [])
    samples_changes = signals.get('samples', [])
    
    # Build time -> value maps
    def build_value_at_time(changes: List[VCDSignalChange]) -> Dict[int, str]:
        result = {}
        current = '0'
        for change in changes:
            current = change.value
            result[change.time] = current
        return result
    
    digest_valid_map = build_value_at_time(digest_valid_changes)
    
    # Find rising edges of digest_valid
    prev_valid = '0'
    test_idx = 0
    
    for change in digest_valid_changes:
        if change.value == '1' and prev_valid == '0':
            # Rising edge - capture digest
            time = change.time
            
            # Find most recent digest value
            digest_val = None
            for dc in reversed(digest_changes):
                if dc.time <= time:
                    digest_val = dc.value
                    break
            
            # Find most recent resonance value
            resonance_val = False
            for rc in reversed(resonance_changes):
                if rc.time <= time:
                    resonance_val = rc.value == '1'
                    break
            
            # Find most recent samples value
            samples_val = None
            for sc in reversed(samples_changes):
                if sc.time <= time:
                    samples_val = sc.value
                    break
            
            if digest_val:
                # Convert binary to hex
                digest_int = int(digest_val, 2) if digest_val else 0
                digest_hex = f"{digest_int:064X}"
                
                # Parse samples (128-bit -> 8x16-bit)
                input_samples = []
                if samples_val:
                    samples_int = int(samples_val, 2)
                    for i in range(8):
                        sample = (samples_int >> (i * 16)) & 0xFFFF
                        # Convert to signed
                        if sample >= 0x8000:
                            sample -= 0x10000
                        input_samples.append(sample)
                
                captures.append(DigestCapture(
                    test_idx=test_idx,
                    time_ns=time // 1000,  # ps to ns
                    digest_hex=digest_hex,
                    resonance=resonance_val,
                    input_samples=input_samples
                ))
                test_idx += 1
        
        prev_valid = change.value
    
    return captures


def compute_golden_digest(samples_q15: List[int]) -> Tuple[str, bool]:
    """
    Compute golden reference digest using Python RFT implementation.
    Matches the hardware compute_block function exactly.
    """
    # Convert Q1.15 to float
    samples_float = np.array([s / 32768.0 for s in samples_q15], dtype=np.float64)
    if len(samples_float) != BLOCK_SAMPLES:
        raise ValueError(f"Expected {BLOCK_SAMPLES} samples, got {len(samples_float)}")
    
    # Apply operator-based RFT forward transform
    rft_out = python_rft_forward(samples_float)
    
    # Quantize to match hardware (32-bit accumulator, take upper 16 bits)
    # Hardware does: rft_real[k] = sum(sample[n] * kernel_real[k,n])
    # Then: q_real[m] = rft_real[m][31:16]
    
    # For this golden model, we'll simulate the fixed-point behavior
    rft_real = np.real(rft_out) * 32768  # Scale up
    rft_imag = np.imag(rft_out) * 32768
    
    # Quantize to 16-bit
    q_real = np.clip(rft_real.astype(int), -32768, 32767)
    q_imag = np.clip(rft_imag.astype(int), -32768, 32767)
    
    # s_vals = 8-bit from (q_real + q_imag)
    s_vals = [(int(q_real[i]) + int(q_imag[i])) & 0xFF for i in range(8)]
    
    # Lattice sliding window sums
    lat = [
        (s_vals[0] + s_vals[1] + s_vals[2] + s_vals[3]) & 0xFFFF,
        (s_vals[1] + s_vals[2] + s_vals[3] + s_vals[4]) & 0xFFFF,
        (s_vals[2] + s_vals[3] + s_vals[4] + s_vals[5]) & 0xFFFF,
        (s_vals[3] + s_vals[4] + s_vals[5] + s_vals[6]) & 0xFFFF,
        (s_vals[4] + s_vals[5] + s_vals[6] + s_vals[7]) & 0xFFFF,
        (s_vals[5] + s_vals[6] + s_vals[7] + s_vals[0]) & 0xFFFF,
        (s_vals[6] + s_vals[7] + s_vals[0] + s_vals[1]) & 0xFFFF,
        (s_vals[7] + s_vals[0] + s_vals[1] + s_vals[2]) & 0xFFFF,
    ]
    
    # Build 256-bit digest (sis_lo = sis_hi pattern)
    # sis_lo = {lat[3], lat[2], lat[1], lat[0], lat[7], lat[6], lat[5], lat[4]}
    sis_lo = (lat[3] << 112) | (lat[2] << 96) | (lat[1] << 80) | (lat[0] << 64) | \
             (lat[7] << 48) | (lat[6] << 32) | (lat[5] << 16) | lat[4]
    digest = (sis_lo << 128) | sis_lo
    
    # Resonance detection (total_energy > 1000)
    total_energy = 0
    for i in range(8):
        mag_real = abs(int(rft_real[i]))
        mag_imag = abs(int(rft_imag[i]))
        amplitude = mag_real + mag_imag
        amp_scaled = amplitude >> 16
        total_energy += amp_scaled * amp_scaled
    
    resonance = total_energy > 1000
    
    return f"{digest:064X}", resonance


def verify_captures(captures: List[DigestCapture]) -> Tuple[int, int, List[str]]:
    """
    Verify captured digests against golden reference.
    Returns (pass_count, fail_count, error_messages).
    """
    passes = 0
    fails = 0
    errors = []
    
    test_names = [
        "dc_signal", "impulse", "alternating", "zeros",
        "ramp", "random_1", "max_positive", "max_negative",
        "back2back_0", "back2back_1", "back2back_2", "back2back_3",
        "reset_recovery"
    ]
    
    for cap in captures:
        name = test_names[cap.test_idx] if cap.test_idx < len(test_names) else f"test_{cap.test_idx}"
        
        if not cap.input_samples:
            errors.append(f"[{name}] No input samples captured")
            fails += 1
            continue
        
        # Compute golden reference
        try:
            golden_hex, golden_resonance = compute_golden_digest(cap.input_samples)
        except Exception as e:
            errors.append(f"[{name}] Golden computation failed: {e}")
            fails += 1
            continue
        
        # Compare
        digest_match = cap.digest_hex == golden_hex
        resonance_match = cap.resonance == golden_resonance
        
        if digest_match and resonance_match:
            passes += 1
            print(f"[PASS] {name}: Digest and resonance match golden reference")
        else:
            fails += 1
            if not digest_match:
                # Find bit difference
                actual = int(cap.digest_hex, 16)
                expected = int(golden_hex, 16)
                diff = actual ^ expected
                diff_bits = bin(diff).count('1')
                errors.append(f"[{name}] Digest mismatch: {diff_bits} bits differ")
                errors.append(f"  Actual:   {cap.digest_hex}")
                errors.append(f"  Expected: {golden_hex}")
            if not resonance_match:
                errors.append(f"[{name}] Resonance mismatch: got {cap.resonance}, expected {golden_resonance}")
    
    return passes, fails, errors


def analyze_intermediate_values(captures: List[DigestCapture]):
    """
    Analyze the intermediate RFT values to verify lattice structure.
    """
    print("\n" + "="*60)
    print("Intermediate Value Analysis (SIS Lattice)")
    print("="*60)
    
    for cap in captures[:4]:  # First 4 tests
        if not cap.input_samples:
            continue
        
        samples = np.array(cap.input_samples) / 32768.0
        rft_out = python_rft_forward(samples)
        
        print(f"\nTest {cap.test_idx} @ {cap.time_ns}ns:")
        print(f"  Input samples (Q1.15): {[hex(s & 0xFFFF) for s in cap.input_samples]}")
        print(f"  RFT output magnitudes: {[f'{abs(c):.4f}' for c in rft_out]}")
        
        # Show lattice values
        digest_int = int(cap.digest_hex, 16)
        lat_values = []
        for i in range(8):
            lat_val = (digest_int >> (i * 16)) & 0xFFFF
            lat_values.append(lat_val)
        print(f"  Lattice values: {[hex(v) for v in lat_values]}")


def main():
    print("="*60)
    print("VCD Digest Verification")
    print("="*60)
    
    vcd_path = Path(__file__).parent / "obj_phi_rft" / "tb_phi_rft_core.vcd"
    
    if not vcd_path.exists():
        print(f"ERROR: VCD file not found: {vcd_path}")
        return 1
    
    print(f"\nParsing: {vcd_path}")
    signals = parse_vcd_simple(vcd_path)
    print(f"Found {len(signals)} signals")
    
    # Extract digest captures
    captures = extract_digest_captures(signals)
    print(f"Extracted {len(captures)} digest captures")
    
    if not captures:
        print("ERROR: No digest captures found in VCD")
        return 1
    
    # Display captures
    print("\n" + "-"*60)
    print("Captured Digests:")
    print("-"*60)
    for cap in captures:
        print(f"  Test {cap.test_idx:2d} @ {cap.time_ns:6d}ns: "
              f"Digest={cap.digest_hex[:16]}... Resonance={cap.resonance}")
    
    # Verify against golden reference
    print("\n" + "="*60)
    print("Verification Against Golden Reference")
    print("="*60)
    
    passes, fails, errors = verify_captures(captures)
    
    for err in errors:
        print(err)
    
    # Analyze intermediate values
    analyze_intermediate_values(captures)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"  Passed: {passes}")
    print(f"  Failed: {fails}")
    print(f"  Total:  {len(captures)}")
    print("="*60)
    
    if fails == 0:
        print("\n✅ ALL DIGESTS MATCH GOLDEN REFERENCE")
        return 0
    else:
        print("\n❌ SOME DIGESTS DO NOT MATCH")
        return 1


if __name__ == "__main__":
    sys.exit(main())
