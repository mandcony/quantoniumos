#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# MEDICAL RESEARCH LICENSE:
# FREE for hospitals, medical researchers, academics, and healthcare
# institutions for testing, validation, and research purposes.
# Commercial medical device use: See LICENSE-CLAIMS-NC.md
#
"""
Medical Applications Test Suite Runner
======================================

Runs all medical/biomedical application tests and generates
a comprehensive report.

Usage:
    python tests/medical/run_medical_benchmarks.py [OPTIONS]
    
Options:
    --quick     Run quick tests only (skip slow benchmarks)
    --report    Generate markdown report
    --imaging   Run imaging tests only
    --biosignal Run biosignal tests only
    --genomics  Run genomics tests only
    --security  Run security tests only
    --edge      Run edge device tests only
"""

import sys
import os
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def run_imaging_benchmark(quick: bool = False) -> Dict[str, Any]:
    """Run imaging reconstruction benchmark."""
    print("\n" + "=" * 70)
    print("MEDICAL IMAGING BENCHMARK")
    print("=" * 70)
    
    from tests.medical.test_imaging_reconstruction import (
        shepp_logan_phantom,
        add_rician_noise,
        add_poisson_noise,
        rft_denoise_2d,
        dct_denoise_2d,
        wavelet_denoise_2d,
        psnr,
        ssim_simple
    )
    
    results = []
    phantom_size = 128 if quick else 256
    phantom = shepp_logan_phantom(phantom_size)
    
    noise_tests = [
        ('Rician_0.05', lambda p: add_rician_noise(p, 0.05)),
        ('Rician_0.10', lambda p: add_rician_noise(p, 0.10)),
        ('Poisson_500', lambda p: add_poisson_noise(p, 500)),
    ]
    
    if not quick:
        noise_tests.extend([
            ('Rician_0.15', lambda p: add_rician_noise(p, 0.15)),
            ('Poisson_100', lambda p: add_poisson_noise(p, 100)),
        ])
    
    methods = [
        ('RFT', rft_denoise_2d),
        ('DCT', dct_denoise_2d),
        ('Wavelet', wavelet_denoise_2d),
    ]
    
    for noise_name, noise_func in noise_tests:
        noisy = noise_func(phantom)
        psnr_noisy = psnr(phantom, noisy)
        
        for method_name, method in methods:
            t0 = time.perf_counter()
            denoised = method(noisy, threshold_ratio=0.05)
            elapsed = (time.perf_counter() - t0) * 1000
            
            results.append({
                'test': 'imaging',
                'noise': noise_name,
                'method': method_name,
                'psnr_before': psnr_noisy,
                'psnr_after': psnr(phantom, denoised),
                'ssim': ssim_simple(phantom, denoised),
                'time_ms': elapsed
            })
    
    return {'imaging': results}


def run_biosignal_benchmark(quick: bool = False) -> Dict[str, Any]:
    """Run biosignal compression benchmark."""
    print("\n" + "=" * 70)
    print("BIOSIGNAL COMPRESSION BENCHMARK")
    print("=" * 70)
    
    from tests.medical.test_biosignal_compression import (
        generate_ecg_signal,
        generate_eeg_signal,
        rft_compress_signal,
        fft_compress_signal,
        snr,
        prd,
        correlation_coefficient
    )
    
    results = []
    
    duration = 10.0 if quick else 30.0
    
    # ECG tests
    ecg, _ = generate_ecg_signal(duration_sec=duration, sample_rate=360)
    keep_ratios = [0.3, 0.5] if quick else [0.2, 0.3, 0.5, 0.7]
    
    for keep_ratio in keep_ratios:
        for method_name, method in [('RFT', rft_compress_signal), ('FFT', fft_compress_signal)]:
            t0 = time.perf_counter()
            recon, stats = method(ecg, keep_ratio=keep_ratio)
            elapsed = (time.perf_counter() - t0) * 1000
            
            results.append({
                'test': 'biosignal',
                'signal': 'ECG',
                'method': method_name,
                'keep_ratio': keep_ratio,
                'compression_ratio': stats['compression_ratio'],
                'snr_db': snr(ecg, recon),
                'prd_percent': prd(ecg, recon),
                'correlation': correlation_coefficient(ecg, recon),
                'time_ms': elapsed
            })
    
    # EEG tests
    eeg, _ = generate_eeg_signal(duration_sec=duration, sample_rate=256)
    
    for keep_ratio in [0.3, 0.5]:
        for method_name, method in [('RFT', rft_compress_signal), ('FFT', fft_compress_signal)]:
            t0 = time.perf_counter()
            recon, stats = method(eeg, keep_ratio=keep_ratio)
            elapsed = (time.perf_counter() - t0) * 1000
            
            results.append({
                'test': 'biosignal',
                'signal': 'EEG',
                'method': method_name,
                'keep_ratio': keep_ratio,
                'compression_ratio': stats['compression_ratio'],
                'snr_db': snr(eeg, recon),
                'prd_percent': prd(eeg, recon),
                'correlation': correlation_coefficient(eeg, recon),
                'time_ms': elapsed
            })
    
    return {'biosignal': results}


def run_genomics_benchmark(quick: bool = False) -> Dict[str, Any]:
    """Run genomics transform benchmark."""
    print("\n" + "=" * 70)
    print("GENOMICS TRANSFORM BENCHMARK")
    print("=" * 70)
    
    from tests.medical.test_genomics_transforms import (
        generate_random_dna,
        compute_kmer_spectrum,
        transform_kmer_spectrum,
        generate_contact_map,
        compress_contact_map_rft,
        contact_map_accuracy,
        compress_sequence_rft,
        gzip_compress_sequence
    )
    
    results = []
    
    # K-mer analysis
    lengths = [10000] if quick else [10000, 50000]
    
    for length in lengths:
        seq = generate_random_dna(length)
        spectrum = compute_kmer_spectrum(seq, k=4)
        
        for transform in ['rft', 'fft', 'dct']:
            t0 = time.perf_counter()
            coeffs = transform_kmer_spectrum(spectrum, transform)
            elapsed = (time.perf_counter() - t0) * 1000
            
            import numpy as np
            mags = np.abs(coeffs)
            top_10_energy = np.sum(np.sort(mags)[-10:]**2) / np.sum(mags**2)
            
            results.append({
                'test': 'genomics',
                'type': 'kmer_spectrum',
                'method': transform.upper(),
                'input_size': length,
                'top_10_energy': top_10_energy,
                'time_ms': elapsed
            })
    
    # Contact map compression
    sizes = [64] if quick else [64, 128]
    
    for size in sizes:
        cmap = generate_contact_map(size)
        
        t0 = time.perf_counter()
        recon, stats = compress_contact_map_rft(cmap, keep_ratio=0.5)
        elapsed = (time.perf_counter() - t0) * 1000
        
        metrics = contact_map_accuracy(cmap, recon)
        
        results.append({
            'test': 'genomics',
            'type': 'contact_map',
            'method': 'RFT',
            'size': f'{size}x{size}',
            'compression_ratio': stats['compression_ratio'],
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'time_ms': elapsed
        })
    
    return {'genomics': results}


def run_security_benchmark(quick: bool = False) -> Dict[str, Any]:
    """Run security and privacy benchmark."""
    print("\n" + "=" * 70)
    print("MEDICAL DATA SECURITY BENCHMARK")
    print("=" * 70)
    
    from tests.medical.test_medical_security import (
        rft_waveform_hash,
        compute_avalanche_effect,
        check_collision_resistance,
        simulate_federated_update,
        federated_aggregate_mean,
        federated_aggregate_rft_filter,
        aggregation_error
    )
    
    import numpy as np
    
    results = []
    
    # Avalanche effect
    waveform = np.random.randn(256)
    n_trials = 30 if quick else 100
    avalanche = compute_avalanche_effect(
        lambda x: rft_waveform_hash(x),
        waveform,
        n_trials=n_trials
    )
    
    results.append({
        'test': 'security',
        'type': 'avalanche_effect',
        'value': avalanche,
        'ideal': 0.5,
        'passed': 0.35 < avalanche < 0.65
    })
    
    # Collision resistance
    n_samples = 200 if quick else 500
    collision_stats = check_collision_resistance(
        lambda x: rft_waveform_hash(x),
        n_samples=n_samples,
        signal_length=256
    )
    
    results.append({
        'test': 'security',
        'type': 'collision_resistance',
        'samples': n_samples,
        'collisions': collision_stats['collisions'],
        'passed': collision_stats['collision_rate'] == 0
    })
    
    # Byzantine resilience
    for byzantine_frac in [0.0, 0.2, 0.3]:
        true_grad, client_grads = simulate_federated_update(
            n_clients=20,
            byzantine_fraction=byzantine_frac,
            noise_std=0.01
        )
        
        mean_error = aggregation_error(true_grad, federated_aggregate_mean(client_grads))
        rft_error = aggregation_error(true_grad, federated_aggregate_rft_filter(client_grads))
        
        results.append({
            'test': 'security',
            'type': 'byzantine_resilience',
            'byzantine_fraction': byzantine_frac,
            'mean_error': mean_error,
            'rft_error': rft_error,
            'improvement': (mean_error - rft_error) / mean_error if mean_error > 0 else 0
        })
    
    return {'security': results}


def run_edge_benchmark(quick: bool = False) -> Dict[str, Any]:
    """Run edge device benchmark."""
    print("\n" + "=" * 70)
    print("EDGE/WEARABLE DEVICE BENCHMARK")
    print("=" * 70)
    
    from tests.medical.test_edge_wearable import (
        DEVICE_PROFILES,
        check_device_memory_fit,
        benchmark_rft_latency,
        simulate_embedded_latency,
        estimate_battery_impact
    )
    
    results = []
    
    # Memory fit test
    for device_name, device in DEVICE_PROFILES.items():
        for size in [128, 256, 512]:
            fits, footprint = check_device_memory_fit(size, device, 'float32')
            
            results.append({
                'test': 'edge',
                'type': 'memory_fit',
                'device': device.name,
                'buffer_size': size,
                'memory_kb': footprint['total_kb'],
                'utilization': footprint['utilization'],
                'fits': fits
            })
    
    # Latency test
    sizes = [128, 256] if quick else [64, 128, 256, 512]
    host_cpu_mhz = 3000
    
    for size in sizes:
        stats = benchmark_rft_latency(size, n_iterations=20 if quick else 50)
        
        for device_name, device in DEVICE_PROFILES.items():
            estimated = simulate_embedded_latency(
                stats['roundtrip_mean_ms'],
                host_cpu_mhz,
                device.cpu_mhz
            )
            
            results.append({
                'test': 'edge',
                'type': 'latency',
                'device': device.name,
                'buffer_size': size,
                'estimated_ms': estimated,
                'target_ms': device.target_latency_ms,
                'meets_target': estimated < device.target_latency_ms
            })
    
    return {'edge': results}


def generate_report(all_results: Dict[str, List], output_path: str = None):
    """Generate markdown report from benchmark results."""
    
    lines = [
        "# QuantoniumOS Medical Benchmark Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n---\n",
    ]
    
    # Imaging results
    if 'imaging' in all_results:
        lines.append("## Medical Imaging Results\n")
        lines.append("| Noise Type | Method | PSNR Before | PSNR After | SSIM | Time (ms) |")
        lines.append("|------------|--------|-------------|------------|------|-----------|")
        
        for r in all_results['imaging']:
            lines.append(f"| {r['noise']} | {r['method']} | {r['psnr_before']:.2f} dB | "
                        f"{r['psnr_after']:.2f} dB | {r['ssim']:.3f} | {r['time_ms']:.1f} |")
        lines.append("")
    
    # Biosignal results
    if 'biosignal' in all_results:
        lines.append("## Biosignal Compression Results\n")
        lines.append("| Signal | Method | Keep Ratio | CR | SNR (dB) | PRD (%) | Corr |")
        lines.append("|--------|--------|------------|-----|----------|---------|------|")
        
        for r in all_results['biosignal']:
            lines.append(f"| {r['signal']} | {r['method']} | {r['keep_ratio']:.1f} | "
                        f"{r['compression_ratio']:.2f}x | {r['snr_db']:.2f} | "
                        f"{r['prd_percent']:.2f} | {r['correlation']:.4f} |")
        lines.append("")
    
    # Genomics results
    if 'genomics' in all_results:
        lines.append("## Genomics Transform Results\n")
        
        kmer_results = [r for r in all_results['genomics'] if r['type'] == 'kmer_spectrum']
        if kmer_results:
            lines.append("### K-mer Spectrum Analysis\n")
            lines.append("| Method | Input Size | Top-10 Energy | Time (ms) |")
            lines.append("|--------|------------|---------------|-----------|")
            for r in kmer_results:
                lines.append(f"| {r['method']} | {r['input_size']} | {r['top_10_energy']:.3f} | {r['time_ms']:.2f} |")
            lines.append("")
        
        cmap_results = [r for r in all_results['genomics'] if r['type'] == 'contact_map']
        if cmap_results:
            lines.append("### Contact Map Compression\n")
            lines.append("| Size | CR | Accuracy | F1 Score | Time (ms) |")
            lines.append("|------|-----|----------|----------|-----------|")
            for r in cmap_results:
                lines.append(f"| {r['size']} | {r['compression_ratio']:.2f}x | "
                            f"{r['accuracy']:.3f} | {r['f1_score']:.3f} | {r['time_ms']:.1f} |")
            lines.append("")
    
    # Security results
    if 'security' in all_results:
        lines.append("## Security and Privacy Results\n")
        
        for r in all_results['security']:
            if r['type'] == 'avalanche_effect':
                status = "✓" if r['passed'] else "✗"
                lines.append(f"- **Avalanche Effect**: {r['value']:.3f} (ideal: 0.5) {status}")
            elif r['type'] == 'collision_resistance':
                status = "✓" if r['passed'] else "✗"
                lines.append(f"- **Collision Resistance**: {r['collisions']} collisions in {r['samples']} samples {status}")
            elif r['type'] == 'byzantine_resilience':
                lines.append(f"- **Byzantine {r['byzantine_fraction']:.0%}**: "
                            f"RFT improvement = {r['improvement']:.1%}")
        lines.append("")
    
    # Edge results
    if 'edge' in all_results:
        lines.append("## Edge Device Results\n")
        
        # Group by device
        devices = {}
        for r in all_results['edge']:
            if r['device'] not in devices:
                devices[r['device']] = []
            devices[r['device']].append(r)
        
        for device, tests in devices.items():
            lines.append(f"### {device}\n")
            
            latency_tests = [t for t in tests if t['type'] == 'latency']
            if latency_tests:
                lines.append("| Buffer Size | Estimated (ms) | Target (ms) | Status |")
                lines.append("|-------------|----------------|-------------|--------|")
                for t in latency_tests:
                    status = "✓" if t['meets_target'] else "✗"
                    lines.append(f"| {t['buffer_size']} | {t['estimated_ms']:.1f} | {t['target_ms']:.1f} | {status} |")
                lines.append("")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\n✓ Report saved to: {output_path}")
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Medical Applications Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--report', action='store_true', help='Generate markdown report')
    parser.add_argument('--imaging', action='store_true', help='Run imaging tests only')
    parser.add_argument('--biosignal', action='store_true', help='Run biosignal tests only')
    parser.add_argument('--genomics', action='store_true', help='Run genomics tests only')
    parser.add_argument('--security', action='store_true', help='Run security tests only')
    parser.add_argument('--edge', action='store_true', help='Run edge tests only')
    
    args = parser.parse_args()
    
    # If no specific test selected, run all
    run_all = not any([args.imaging, args.biosignal, args.genomics, args.security, args.edge])
    
    print("=" * 70)
    print("QUANTONIUMOS MEDICAL APPLICATIONS BENCHMARK SUITE")
    print("=" * 70)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    try:
        if run_all or args.imaging:
            results = run_imaging_benchmark(args.quick)
            all_results.update(results)
        
        if run_all or args.biosignal:
            results = run_biosignal_benchmark(args.quick)
            all_results.update(results)
        
        if run_all or args.genomics:
            results = run_genomics_benchmark(args.quick)
            all_results.update(results)
        
        if run_all or args.security:
            results = run_security_benchmark(args.quick)
            all_results.update(results)
        
        if run_all or args.edge:
            results = run_edge_benchmark(args.quick)
            all_results.update(results)
        
    except ImportError as e:
        print(f"\n⚠ Import error: {e}")
        print("Some tests may require additional dependencies.")
        return 1
    
    # Generate report
    if args.report:
        report_path = os.path.join(
            os.path.dirname(__file__),
            f"MEDICAL_BENCHMARK_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        generate_report(all_results, report_path)
    else:
        # Print summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        for category, results in all_results.items():
            print(f"\n{category.upper()}: {len(results)} tests completed")
    
    print("\n✓ Medical benchmark suite complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
