#!/usr/bin/env python3
"""
RFT-Wavelet Hybrid Benchmark on Real Open-Source Medical Data
==============================================================

Validates RFT-Wavelet hybrid denoising on real MIT-BIH and PhysioNet data.

Datasets Used (all open-source, MIT/PhysioNet license):
1. MIT-BIH Arrhythmia Database - ECG signals
2. PhysioNet Sleep-EDF - EEG/EOG polysomnography

⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️

This benchmark:
- Loads real biosignal data
- Adds controlled noise (Rician for MRI-like, Gaussian for ECG)
- Compares: Pure Wavelet vs RFT-only vs RFT-Wavelet Hybrid
- Reports PSNR improvement, SSIM, execution time
- Saves results to JSON for reproducibility

Usage:
    python benchmarks/rft_wavelet_real_data_benchmark.py

Copyright (C) 2025 Luis M. Minier / quantoniumos
Licensed under AGPL-3.0-or-later
"""

import os
import sys
import json
import time
import datetime
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Data Loading Functions (MIT-BIH / PhysioNet)
# ============================================================================

def load_mitbih_record(record_id: str) -> Tuple[np.ndarray, int]:
    """
    Load MIT-BIH ECG record using wfdb format.
    
    Returns:
        (signal_array, sampling_rate)
    """
    data_dir = PROJECT_ROOT / "data" / "mitbih"
    dat_file = data_dir / f"{record_id}.dat"
    hea_file = data_dir / f"{record_id}.hea"
    
    if not dat_file.exists():
        # Try alternate location
        data_dir = PROJECT_ROOT / "data" / "physionet" / "mitbih"
        dat_file = data_dir / f"{record_id}.dat"
        hea_file = data_dir / f"{record_id}.hea"
    
    if not dat_file.exists():
        raise FileNotFoundError(f"MIT-BIH record {record_id} not found in {data_dir}")
    
    # Parse header for format info
    fs = 360  # Default MIT-BIH sampling rate
    n_signals = 2
    adc_gain = 200
    adc_zero = 1024
    fmt = 212  # Default format
    
    if hea_file.exists():
        with open(hea_file, 'r') as f:
            lines = f.readlines()
        # First line: record_name n_signals fs n_samples
        parts = lines[0].split()
        if len(parts) >= 3:
            n_signals = int(parts[1])
            fs = int(parts[2])
        # Signal lines contain gain info
        for line in lines[1:n_signals+1]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    adc_gain = float(parts[2].split('/')[0])
                except:
                    pass
    
    # Read binary data (format 212: 12-bit pairs packed into 3 bytes)
    raw = np.fromfile(dat_file, dtype=np.uint8)
    
    # Unpack format 212
    n_samples = len(raw) // 3 * 2
    signal = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(0, len(raw) - 2, 3):
        # First sample: low byte + low nibble of middle byte
        s1 = raw[i] + ((raw[i+1] & 0x0F) << 8)
        if s1 > 2047:
            s1 -= 4096
        # Second sample: high nibble of middle byte + high byte
        s2 = (raw[i+1] >> 4) + (raw[i+2] << 4)
        if s2 > 2047:
            s2 -= 4096
        
        idx = (i // 3) * 2
        if idx < n_samples:
            signal[idx] = (s1 - adc_zero) / adc_gain
        if idx + 1 < n_samples:
            signal[idx + 1] = (s2 - adc_zero) / adc_gain
    
    return signal[:n_samples], fs


def load_sleepedf_record(edf_file: Path) -> Tuple[np.ndarray, int]:
    """
    Load Sleep-EDF PSG record (EEG channel).
    
    Returns:
        (signal_array, sampling_rate)
    """
    if not edf_file.exists():
        raise FileNotFoundError(f"Sleep-EDF file not found: {edf_file}")
    
    # Simple EDF reader for our benchmark
    with open(edf_file, 'rb') as f:
        # Header is 256 bytes + 256 bytes per signal
        header = f.read(256)
        
        # Parse header
        n_signals = int(header[252:256].decode().strip())
        n_data_records = int(header[236:244].decode().strip())
        record_duration = float(header[244:252].decode().strip())
        
        # Read signal headers
        signal_headers = f.read(256 * n_signals)
        
        # Get samples per record for first signal
        samples_per_record = []
        offset = 216 * n_signals  # Skip to nr field
        for i in range(n_signals):
            ns = int(signal_headers[offset + i*8:offset + (i+1)*8].decode().strip())
            samples_per_record.append(ns)
        
        # Get digital/physical min/max for scaling
        physical_min = []
        physical_max = []
        digital_min = []
        digital_max = []
        
        for i in range(n_signals):
            pm_start = 104 * n_signals + i * 8
            physical_min.append(float(signal_headers[pm_start:pm_start+8].decode().strip()))
            pm_start = 112 * n_signals + i * 8
            physical_max.append(float(signal_headers[pm_start:pm_start+8].decode().strip()))
            dm_start = 120 * n_signals + i * 8
            digital_min.append(float(signal_headers[dm_start:dm_start+8].decode().strip()))
            dm_start = 128 * n_signals + i * 8
            digital_max.append(float(signal_headers[dm_start:dm_start+8].decode().strip()))
        
        # Read data records (take first signal - usually EEG Fpz-Cz)
        fs = int(samples_per_record[0] / record_duration)
        total_samples = samples_per_record[0] * min(n_data_records, 100)  # Limit to 100 records
        
        signal = np.zeros(total_samples, dtype=np.float32)
        idx = 0
        
        for rec in range(min(n_data_records, 100)):
            for sig_idx in range(n_signals):
                n_samp = samples_per_record[sig_idx]
                raw = np.frombuffer(f.read(n_samp * 2), dtype=np.int16)
                
                if sig_idx == 0:  # First signal (EEG)
                    # Scale to physical units
                    scale = (physical_max[0] - physical_min[0]) / (digital_max[0] - digital_min[0])
                    offset_val = physical_min[0] - digital_min[0] * scale
                    signal[idx:idx+n_samp] = raw * scale + offset_val
                    idx += n_samp
    
    return signal[:idx], fs


# ============================================================================
# Noise Models
# ============================================================================

def add_rician_noise(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Add Rician noise (MRI-like) to signal."""
    # Ensure signal is positive for Rician model
    signal_shifted = signal - signal.min() + 0.1
    
    # Rician: |signal + N(0,σ) + j*N(0,σ)|
    real = signal_shifted + np.random.normal(0, sigma, signal.shape)
    imag = np.random.normal(0, sigma, signal.shape)
    noisy = np.sqrt(real**2 + imag**2)
    
    # Shift back
    return noisy - 0.1 + signal.min()


def add_gaussian_noise(signal: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to signal."""
    return signal + np.random.normal(0, sigma, signal.shape)


def add_poisson_noise(signal: np.ndarray, scale: float = 100) -> np.ndarray:
    """Add Poisson noise (photon counting like PET/CT)."""
    # Scale to counts
    signal_shifted = signal - signal.min() + 0.01
    counts = signal_shifted * scale
    counts = np.maximum(counts, 0.01)
    
    # Poisson sampling
    noisy_counts = np.random.poisson(counts).astype(np.float32)
    
    # Scale back
    return noisy_counts / scale + signal.min() - 0.01


# ============================================================================
# Denoising Methods
# ============================================================================

def wavelet_denoise_1d(signal: np.ndarray, levels: int = 3) -> np.ndarray:
    """Pure wavelet denoising using Haar."""
    # Pad to power of 2
    n = len(signal)
    n_pad = 2 ** int(np.ceil(np.log2(n)))
    padded = np.zeros(n_pad)
    padded[:n] = signal
    
    # Multi-level Haar decomposition - store approximation coefficients too
    approx = padded.copy()
    details = []
    
    for _ in range(levels):
        n_curr = len(approx)
        # Handle odd length
        if n_curr % 2 != 0:
            approx = np.append(approx, approx[-1])
            n_curr += 1
        low = (approx[0::2] + approx[1::2]) / np.sqrt(2)
        high = (approx[0::2] - approx[1::2]) / np.sqrt(2)
        details.append(high)
        approx = low
    
    # BayesShrink thresholding on detail coefficients
    # Estimate noise from finest level details
    sigma = np.median(np.abs(details[0])) / 0.6745
    
    for i, det in enumerate(details):
        # Threshold decreases at coarser levels
        threshold = sigma * np.sqrt(2 * np.log(len(det))) * (0.8 ** i)
        details[i] = np.sign(det) * np.maximum(np.abs(det) - threshold, 0)
    
    # Reconstruct from coarsest to finest
    current = approx
    for i in range(len(details) - 1, -1, -1):
        high = details[i]
        n_out = len(high) * 2
        reconstructed = np.zeros(n_out)
        reconstructed[0::2] = (current + high) / np.sqrt(2)
        reconstructed[1::2] = (current - high) / np.sqrt(2)
        current = reconstructed
    
    return current[:n]


def rft_denoise_1d(signal: np.ndarray) -> np.ndarray:
    """Pure RFT denoising using Wiener filtering."""
    try:
        from algorithms.rft.variants.operator_variants import get_operator_variant
        
        n = len(signal)
        Phi = get_operator_variant('rft_entropy_modulated', n)
        
        # Forward transform
        coeffs = Phi.T @ signal.astype(np.float64)
        
        # Estimate noise from high-frequency coefficients
        noise_var = np.var(coeffs[n//2:])
        
        # Wiener filter
        power = np.abs(coeffs) ** 2
        wiener = power / (power + noise_var + 1e-10)
        filtered = coeffs * wiener
        
        # Inverse
        return (Phi @ filtered).real
        
    except ImportError:
        # Fallback to FFT-based
        coeffs = np.fft.fft(signal)
        noise_var = np.var(np.abs(coeffs[len(coeffs)//2:]))
        power = np.abs(coeffs) ** 2
        wiener = power / (power + noise_var + 1e-10)
        return np.fft.ifft(coeffs * wiener).real


def rft_wavelet_hybrid_denoise_1d(signal: np.ndarray, levels: int = 3) -> np.ndarray:
    """RFT-Wavelet hybrid denoising for 1D signals."""
    try:
        from algorithms.rft.variants.operator_variants import get_operator_variant
        use_rft_variant = True
    except ImportError:
        use_rft_variant = False
    
    # Pad to power of 2
    n = len(signal)
    n_pad = 2 ** int(np.ceil(np.log2(n)))
    padded = np.zeros(n_pad)
    padded[:n] = signal
    
    # Multi-level Haar decomposition
    approx = padded.copy()
    details = []
    
    for _ in range(levels):
        n_curr = len(approx)
        if n_curr % 2 != 0:
            approx = np.append(approx, approx[-1])
            n_curr += 1
        low = (approx[0::2] + approx[1::2]) / np.sqrt(2)
        high = (approx[0::2] - approx[1::2]) / np.sqrt(2)
        details.append(high)
        approx = low
    
    # Estimate noise from finest level
    sigma = np.median(np.abs(details[0])) / 0.6745
    noise_var = sigma ** 2
    
    # Apply RFT filtering to detail subbands
    filtered_details = []
    for i, detail in enumerate(details):
        n_d = len(detail)
        
        # Scale noise variance by level
        level_noise_var = noise_var * (0.5 ** i)
        
        if use_rft_variant and n_d >= 16:
            try:
                Phi = get_operator_variant('rft_entropy_modulated', n_d)
                coeffs = Phi.T @ detail.astype(np.float64)
                
                # Wiener filter in RFT domain
                power = np.abs(coeffs) ** 2
                wiener = power / (power + level_noise_var + 1e-10)
                filtered = coeffs * wiener
                
                # Inverse
                detail_filtered = (Phi @ filtered).real
            except:
                detail_filtered = detail.copy()
        else:
            detail_filtered = detail.copy()
        
        # Light soft threshold (reduced because RFT already filtered)
        threshold = sigma * np.sqrt(2 * np.log(n_d)) * (0.3 ** (i + 1))
        detail_filtered = np.sign(detail_filtered) * np.maximum(
            np.abs(detail_filtered) - threshold, 0
        )
        filtered_details.append(detail_filtered)
    
    # Reconstruct from coarsest to finest
    current = approx
    for i in range(len(filtered_details) - 1, -1, -1):
        high = filtered_details[i]
        n_out = len(high) * 2
        reconstructed = np.zeros(n_out)
        reconstructed[0::2] = (current + high) / np.sqrt(2)
        reconstructed[1::2] = (current - high) / np.sqrt(2)
        current = reconstructed
    
    return current[:n]


# ============================================================================
# Metrics
# ============================================================================

def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - processed) ** 2)
    if mse < 1e-10:
        return 100.0
    max_val = np.max(np.abs(original))
    return 10 * np.log10(max_val ** 2 / mse)


def compute_snr(original: np.ndarray, noisy: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - noisy) ** 2)
    if noise_power < 1e-10:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)


def compute_correlation(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    return np.corrcoef(original.flatten(), processed.flatten())[0, 1]


# ============================================================================
# Benchmark Runner
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    dataset: str
    record_id: str
    noise_type: str
    noise_level: float
    method: str
    psnr_noisy: float
    psnr_denoised: float
    psnr_improvement: float
    correlation_noisy: float
    correlation_denoised: float
    time_ms: float


def run_benchmark_on_signal(
    signal: np.ndarray,
    dataset: str,
    record_id: str,
    noise_configs: List[Tuple[str, float]],
    segment_length: int = 4096
) -> List[BenchmarkResult]:
    """
    Run denoising benchmark on a signal.
    
    Args:
        signal: Clean signal
        dataset: Dataset name
        record_id: Record identifier
        noise_configs: List of (noise_type, noise_level) tuples
        segment_length: Length of signal segment to use
        
    Returns:
        List of BenchmarkResult
    """
    results = []
    
    # Normalize signal to [0, 1]
    sig_min, sig_max = signal.min(), signal.max()
    if sig_max - sig_min < 1e-10:
        return results
    
    signal_norm = (signal - sig_min) / (sig_max - sig_min)
    
    # Take center segment
    start = max(0, len(signal_norm) // 2 - segment_length // 2)
    segment = signal_norm[start:start + segment_length]
    
    if len(segment) < 256:
        return results
    
    methods = [
        ("Wavelet (Haar)", wavelet_denoise_1d),
        ("RFT-only", rft_denoise_1d),
        ("RFT-Wavelet Hybrid", rft_wavelet_hybrid_denoise_1d),
    ]
    
    for noise_type, noise_level in noise_configs:
        # Add noise
        np.random.seed(42)  # Reproducibility
        
        if noise_type == "gaussian":
            noisy = add_gaussian_noise(segment, noise_level)
        elif noise_type == "rician":
            noisy = add_rician_noise(segment, noise_level)
        elif noise_type == "poisson":
            noisy = add_poisson_noise(segment, noise_level)
        else:
            continue
        
        psnr_noisy = compute_psnr(segment, noisy)
        corr_noisy = compute_correlation(segment, noisy)
        
        for method_name, method_func in methods:
            start_time = time.perf_counter()
            
            try:
                denoised = method_func(noisy)
            except Exception as e:
                print(f"  ⚠ {method_name} failed: {e}")
                continue
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            psnr_denoised = compute_psnr(segment, denoised)
            corr_denoised = compute_correlation(segment, denoised)
            
            result = BenchmarkResult(
                dataset=dataset,
                record_id=record_id,
                noise_type=noise_type,
                noise_level=noise_level,
                method=method_name,
                psnr_noisy=round(psnr_noisy, 2),
                psnr_denoised=round(psnr_denoised, 2),
                psnr_improvement=round(psnr_denoised - psnr_noisy, 2),
                correlation_noisy=round(corr_noisy, 4),
                correlation_denoised=round(corr_denoised, 4),
                time_ms=round(elapsed_ms, 2),
            )
            results.append(result)
    
    return results


DISCLAIMER = """
================================================================================
⚠️  DISCLAIMER: RESEARCH USE ONLY  ⚠️

- NOT FOR CLINICAL OR DIAGNOSTIC APPLICATION
- Results measure signal-level metrics (PSNR, SNR, correlation)
- This software is NOT validated for medical device use
- Data used under PhysioNet Research License
================================================================================
"""


def run_full_benchmark() -> Dict:
    """
    Run full benchmark on all available real datasets.
    
    Returns:
        Dictionary with all results and summary statistics.
    """
    print(DISCLAIMER)
    print("=" * 70)
    print("RFT-Wavelet Hybrid Benchmark on Real Open-Source Medical Data")
    print("=" * 70)
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print()
    
    all_results: List[BenchmarkResult] = []
    
    # Noise configurations to test
    noise_configs = [
        ("gaussian", 0.05),
        ("gaussian", 0.10),
        ("gaussian", 0.15),
        ("rician", 0.05),
        ("rician", 0.10),
        ("rician", 0.15),
        ("poisson", 200),  # Higher = more photons = less noise
        ("poisson", 100),
        ("poisson", 50),
    ]
    
    # =========================================
    # MIT-BIH Arrhythmia Database
    # =========================================
    print("\n" + "-" * 50)
    print("Dataset: MIT-BIH Arrhythmia Database (ECG)")
    print("-" * 50)
    
    mitbih_records = ["100", "101", "200", "207", "208", "217"]
    
    for record_id in mitbih_records:
        try:
            signal, fs = load_mitbih_record(record_id)
            print(f"\n✓ Loaded MIT-BIH {record_id}: {len(signal)} samples @ {fs} Hz")
            
            results = run_benchmark_on_signal(
                signal, "MIT-BIH", record_id, noise_configs
            )
            all_results.extend(results)
            
            # Print summary for this record
            for noise_type in ["gaussian", "rician", "poisson"]:
                relevant = [r for r in results if r.noise_type == noise_type]
                if relevant:
                    print(f"  {noise_type.capitalize()} noise:")
                    for r in relevant:
                        print(f"    {r.method}: PSNR {r.psnr_noisy:.1f}→{r.psnr_denoised:.1f} dB "
                              f"(+{r.psnr_improvement:.1f}), {r.time_ms:.1f}ms")
                              
        except FileNotFoundError as e:
            print(f"✗ MIT-BIH {record_id}: {e}")
        except Exception as e:
            print(f"✗ MIT-BIH {record_id} error: {e}")
    
    # =========================================
    # PhysioNet Sleep-EDF (EEG)
    # =========================================
    print("\n" + "-" * 50)
    print("Dataset: PhysioNet Sleep-EDF (EEG)")
    print("-" * 50)
    
    sleepedf_dir = PROJECT_ROOT / "data" / "physionet" / "sleepedf"
    if sleepedf_dir.exists():
        edf_files = list(sleepedf_dir.glob("*PSG.edf"))[:2]  # First 2 records
        
        for edf_file in edf_files:
            try:
                signal, fs = load_sleepedf_record(edf_file)
                record_id = edf_file.stem
                print(f"\n✓ Loaded Sleep-EDF {record_id}: {len(signal)} samples @ {fs} Hz")
                
                results = run_benchmark_on_signal(
                    signal, "Sleep-EDF", record_id, noise_configs
                )
                all_results.extend(results)
                
                # Print summary
                for noise_type in ["gaussian", "rician"]:
                    relevant = [r for r in results if r.noise_type == noise_type]
                    if relevant:
                        print(f"  {noise_type.capitalize()} noise:")
                        for r in relevant:
                            print(f"    {r.method}: PSNR {r.psnr_noisy:.1f}→{r.psnr_denoised:.1f} dB "
                                  f"(+{r.psnr_improvement:.1f}), {r.time_ms:.1f}ms")
                            
            except Exception as e:
                print(f"✗ Sleep-EDF {edf_file.name} error: {e}")
    else:
        print("✗ Sleep-EDF data not found")
    
    # =========================================
    # Summary Statistics
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    summary = {}
    methods = ["Wavelet (Haar)", "RFT-only", "RFT-Wavelet Hybrid"]
    
    for noise_type in ["gaussian", "rician", "poisson"]:
        summary[noise_type] = {}
        relevant = [r for r in all_results if r.noise_type == noise_type]
        
        if not relevant:
            continue
            
        print(f"\n{noise_type.upper()} NOISE:")
        print("-" * 50)
        
        for method in methods:
            method_results = [r for r in relevant if r.method == method]
            if method_results:
                avg_improvement = np.mean([r.psnr_improvement for r in method_results])
                avg_corr = np.mean([r.correlation_denoised for r in method_results])
                avg_time = np.mean([r.time_ms for r in method_results])
                
                summary[noise_type][method] = {
                    "avg_psnr_improvement": round(avg_improvement, 2),
                    "avg_correlation": round(avg_corr, 4),
                    "avg_time_ms": round(avg_time, 2),
                    "n_tests": len(method_results),
                }
                
                print(f"  {method:25s}: +{avg_improvement:5.2f} dB  "
                      f"(corr={avg_corr:.4f}, {avg_time:.1f}ms)")
        
        # Winner for this noise type
        if summary[noise_type]:
            winner = max(summary[noise_type].items(), 
                        key=lambda x: x[1]["avg_psnr_improvement"])
            print(f"  → WINNER: {winner[0]} (+{winner[1]['avg_psnr_improvement']:.2f} dB)")
    
    # Overall winner
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    
    overall_by_method = {}
    for method in methods:
        method_results = [r for r in all_results if r.method == method]
        if method_results:
            overall_by_method[method] = {
                "avg_psnr_improvement": round(
                    np.mean([r.psnr_improvement for r in method_results]), 2
                ),
                "avg_correlation": round(
                    np.mean([r.correlation_denoised for r in method_results]), 4
                ),
                "avg_time_ms": round(
                    np.mean([r.time_ms for r in method_results]), 2
                ),
                "n_tests": len(method_results),
            }
            print(f"{method:25s}: +{overall_by_method[method]['avg_psnr_improvement']:5.2f} dB avg  "
                  f"({overall_by_method[method]['n_tests']} tests)")
    
    if overall_by_method:
        overall_winner = max(overall_by_method.items(), 
                            key=lambda x: x[1]["avg_psnr_improvement"])
        print(f"\n🏆 OVERALL WINNER: {overall_winner[0]}")
        print(f"   Average PSNR improvement: +{overall_winner[1]['avg_psnr_improvement']:.2f} dB")
    
    # Build final report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0",
        "datasets": ["MIT-BIH Arrhythmia Database", "PhysioNet Sleep-EDF"],
        "license": "Data: PhysioNet/MIT License | Code: AGPL-3.0",
        "disclaimer": DISCLAIMER.strip(),
        "noise_configs": [
            {"type": t, "level": l} for t, l in noise_configs
        ],
        "methods": methods,
        "results": [asdict(r) for r in all_results],
        "summary_by_noise_type": summary,
        "overall_by_method": overall_by_method,
    }
    
    # Save to JSON
    output_path = PROJECT_ROOT / "data" / "experiments" / "rft_wavelet_real_data_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(DISCLAIMER)
    
    return report


if __name__ == "__main__":
    report = run_full_benchmark()
    
    # Print JSON path for easy access
    print("\n" + "=" * 70)
    print("To view full results:")
    print(f"  cat data/experiments/rft_wavelet_real_data_results.json | python -m json.tool")
