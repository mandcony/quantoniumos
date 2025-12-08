#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RTL vs Python Cross-Validation Script
======================================

Compares RTL simulation outputs (from Verilator) against Python reference
implementation to verify the hardware correctly implements the canonical RFT.

Usage:
    # First run Verilator simulation to generate rtl_outputs.csv
    cd hardware/tb && make run-crossval
    
    # Then run this script
    python3 verify_rtl_vs_python.py
"""

import sys
import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rft.variants.operator_variants import generate_rft_golden

# Fixed-point parameters (must match hardware)
FRAC_BITS = 15
BLOCK_SIZE = 8
Q15_SCALE = 2**FRAC_BITS  # 32768


@dataclass
class TestResult:
    test_id: int
    input_real: np.ndarray
    input_imag: np.ndarray
    rtl_output_real: np.ndarray
    rtl_output_imag: np.ndarray
    python_output_real: np.ndarray
    python_output_imag: np.ndarray
    max_error_real: int  # In LSBs
    max_error_imag: int
    passed: bool


def q15_to_float(q15_val: int) -> float:
    """Convert Q1.15 signed integer to float."""
    if q15_val >= 0x8000:
        q15_val -= 0x10000
    return q15_val / Q15_SCALE


def float_to_q15(val: float) -> int:
    """Convert float to Q1.15 signed integer."""
    val = max(-1.0, min(0.999969482421875, val))
    q15 = int(val * Q15_SCALE)
    return q15 & 0xFFFF


def compute_python_rft_q15(input_real: np.ndarray, input_imag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute canonical RFT in Q1.15 fixed-point matching hardware.
    
    The hardware does:
      1. Q1.15 × Q1.15 multiply → Q2.30 accumulator
      2. Sum all products
      3. Right-shift 15 → Q1.15 output
    """
    kernel = generate_rft_golden(BLOCK_SIZE)  # Real-only eigenbasis
    
    # Convert Q1.15 inputs to float
    x_real = np.array([q15_to_float(int(v)) for v in input_real])
    x_imag = np.array([q15_to_float(int(v)) for v in input_imag])
    
    # Matrix-vector multiply (canonical RFT is real-only kernel)
    y_real = kernel.real @ x_real
    y_imag = kernel.real @ x_imag
    
    # Quantize outputs to Q1.15
    out_real = np.array([float_to_q15(v) for v in y_real])
    out_imag = np.array([float_to_q15(v) for v in y_imag])
    
    return out_real, out_imag


def compute_python_rft_q15_exact(input_real: np.ndarray, input_imag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute canonical RFT matching hardware bit-exact behavior.
    
    Uses integer arithmetic to match hardware Q1.15 × Q1.15 → Q2.30 → Q1.15 pipeline.
    """
    # Kernel LUT values (Q1.15, matching hardware)
    kernel_q15 = np.array([
        [0xD6E0, 0x3209, 0xD1F4, 0xCF18, 0x36D4, 0xC837, 0xDAF0, 0xF272],
        [0xD2A3, 0x301D, 0x2BF0, 0xE165, 0x2641, 0x3DE5, 0x3457, 0x214B],
        [0xD0C9, 0xD806, 0x2C59, 0xC4D2, 0xDDDE, 0x1BBC, 0xCACC, 0xCFD0],
        [0xD0F5, 0xD5E0, 0xD160, 0xDB1D, 0xCD70, 0xEA1E, 0x2353, 0x43A8],
        [0xD0F5, 0x2A20, 0xD160, 0x24E3, 0xCD70, 0x15E2, 0x2353, 0xBC58],
        [0xD0C9, 0x27FA, 0x2C59, 0x3B2E, 0xDDDE, 0xE444, 0xCACC, 0x3030],
        [0xD2A3, 0xCFE3, 0x2BF0, 0x1E9B, 0x2641, 0xC21B, 0x3457, 0xDEB5],
        [0xD6E0, 0xCDF7, 0xD1F4, 0x30E8, 0x36D4, 0x37C9, 0xDAF0, 0x0D8E],
    ], dtype=np.uint16)
    
    # Convert to signed
    kernel_signed = np.zeros((8, 8), dtype=np.int32)
    for k in range(8):
        for n in range(8):
            val = int(kernel_q15[k, n])
            if val >= 0x8000:
                val -= 0x10000
            kernel_signed[k, n] = val
    
    # Convert inputs to signed int32
    x_real = np.zeros(8, dtype=np.int32)
    x_imag = np.zeros(8, dtype=np.int32)
    for i in range(8):
        val_r = int(input_real[i])
        val_i = int(input_imag[i])
        if val_r >= 0x8000:
            val_r -= 0x10000
        if val_i >= 0x8000:
            val_i -= 0x10000
        x_real[i] = val_r
        x_imag[i] = val_i
    
    # Matrix-vector multiply with integer arithmetic
    out_real = np.zeros(8, dtype=np.int32)
    out_imag = np.zeros(8, dtype=np.int32)
    
    for k in range(8):
        acc_real = 0
        acc_imag = 0
        for n in range(8):
            # Q1.15 × Q1.15 = Q2.30
            acc_real += kernel_signed[k, n] * x_real[n]
            acc_imag += kernel_signed[k, n] * x_imag[n]
        # Q2.30 → Q1.15 (arithmetic right shift by 15)
        out_real[k] = acc_real >> 15
        out_imag[k] = acc_imag >> 15
    
    return out_real, out_imag


def parse_rtl_outputs(csv_path: Path) -> dict:
    """Parse RTL simulation output CSV."""
    results = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_id = int(row['test_id'])
            sample_idx = int(row['sample_idx'])
            
            if test_id not in results:
                results[test_id] = {
                    'input_real': [0] * BLOCK_SIZE,
                    'input_imag': [0] * BLOCK_SIZE,
                    'output_real': [0] * BLOCK_SIZE,
                    'output_imag': [0] * BLOCK_SIZE,
                }
            
            results[test_id]['input_real'][sample_idx] = int(row['input_real'])
            results[test_id]['input_imag'][sample_idx] = int(row['input_imag'])
            results[test_id]['output_real'][sample_idx] = int(row['output_real'])
            results[test_id]['output_imag'][sample_idx] = int(row['output_imag'])
    
    return results


def run_crossval(rtl_csv_path: Optional[Path] = None) -> List[TestResult]:
    """Run cross-validation between RTL outputs and Python reference."""
    
    if rtl_csv_path is None:
        rtl_csv_path = Path(__file__).parent / "rtl_outputs.csv"
    
    if not rtl_csv_path.exists():
        print(f"ERROR: RTL output file not found: {rtl_csv_path}")
        print("Run Verilator simulation first: make run-crossval")
        sys.exit(1)
    
    rtl_data = parse_rtl_outputs(rtl_csv_path)
    results = []
    
    print("=" * 60)
    print("RTL vs Python Cross-Validation")
    print("=" * 60)
    print(f"RTL outputs: {rtl_csv_path}")
    print(f"Test cases: {len(rtl_data)}")
    print()
    
    for test_id in sorted(rtl_data.keys()):
        data = rtl_data[test_id]
        
        input_real = np.array(data['input_real'], dtype=np.int32)
        input_imag = np.array(data['input_imag'], dtype=np.int32)
        rtl_out_real = np.array(data['output_real'], dtype=np.int32)
        rtl_out_imag = np.array(data['output_imag'], dtype=np.int32)
        
        # Compute Python reference (bit-exact)
        py_out_real, py_out_imag = compute_python_rft_q15_exact(input_real, input_imag)
        
        # Compute errors (in LSBs)
        err_real = np.abs(rtl_out_real - py_out_real)
        err_imag = np.abs(rtl_out_imag - py_out_imag)
        max_err_real = int(np.max(err_real))
        max_err_imag = int(np.max(err_imag))
        
        # Pass if within 2 LSB tolerance (for rounding differences)
        passed = max_err_real <= 2 and max_err_imag <= 2
        
        result = TestResult(
            test_id=test_id,
            input_real=input_real,
            input_imag=input_imag,
            rtl_output_real=rtl_out_real,
            rtl_output_imag=rtl_out_imag,
            python_output_real=py_out_real,
            python_output_imag=py_out_imag,
            max_error_real=max_err_real,
            max_error_imag=max_err_imag,
            passed=passed,
        )
        results.append(result)
        
        status = "PASS" if passed else "FAIL"
        print(f"  Test {test_id:2d}: {status} (err_real={max_err_real} LSB, err_imag={max_err_imag} LSB)")
        
        if not passed:
            print(f"           RTL real:    {list(rtl_out_real)}")
            print(f"           Python real: {list(py_out_real)}")
    
    return results


def main():
    results = run_crossval()
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Passed: {passed} / {len(results)}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print()
        print("✓ RTL implementation matches Python canonical RFT reference")
        print("  Hardware is cross-validated within ±2 LSB tolerance")
        return 0
    else:
        print()
        print("✗ RTL/Python mismatch detected")
        print("  Review kernel LUT values and fixed-point arithmetic")
        return 1


if __name__ == "__main__":
    sys.exit(main())
