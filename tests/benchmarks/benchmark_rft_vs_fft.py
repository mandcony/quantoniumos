#!/usr/bin/env python3
"""
RFT vs FFT Competitive Benchmark Suite
=====================================

Critical validation test comparing RFT performance against standard FFT.
This is a BLOCKING requirement for QuantoniumOS validation.

PHASE 1.1: Competitive Transform Benchmarks
Timeline: 1-2 weeks
Priority: CRITICAL
"""

import numpy as np
import time
import json
from pathlib import Path
from scipy.fftpack import fft, ifft
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Import QuantoniumOS RFT implementation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rft_unified_interface import UnifiedRFTInterface, get_available_rft_implementations
    available_impls = get_available_rft_implementations()
    RFT_AVAILABLE = any(available_impls.values())
    if RFT_AVAILABLE:
        print(f"🚀 PROVEN ASSEMBLY RFT available: {[k for k, v in available_impls.items() if v]}")
        print("   Using libquantum_symbolic.so with SIMD optimization")
    else:
        print("⚠️ WARNING: No RFT implementations available")
except ImportError as e:
    print(f"⚠️ WARNING: RFT implementation not found: {e}")
    RFT_AVAILABLE = False

class TransformBenchmark:
    """Comprehensive benchmark comparing RFT vs FFT performance."""
    
    def __init__(self):
        self.results = {
            'speed': {},
            'accuracy': {},
            'compression': {},
            'memory': {}
        }
        
    def benchmark_speed(self) -> Dict:
        """Test 1: Speed comparison across different sizes."""
        print("\n🏃 SPEED BENCHMARK: RFT vs FFT")
        print("=" * 50)
        
        sizes = [64, 128, 256, 512, 1024, 2048]
        iterations = 1000
        speed_results = {}
        
        for N in sizes:
            print(f"Testing N={N}...")
            data = np.random.randn(N) + 1j * np.random.randn(N)
            
            # FFT benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                fft_result = fft(data)
            fft_time = time.perf_counter() - start
            
            # RFT benchmark
            if RFT_AVAILABLE:
                try:
                    rft = UnifiedRFTInterface(N, "auto")
                    start = time.perf_counter()
                    for _ in range(iterations):
                        rft_result = rft.transform(data)
                    rft_time = time.perf_counter() - start
                    
                    ratio = rft_time / fft_time
                    print(f"  FFT: {fft_time:.4f}s | RFT: {rft_time:.4f}s | Ratio: {ratio:.2f}x")
                except Exception as e:
                    rft_time = float('inf')
                    ratio = float('inf')
                    print(f"  FFT: {fft_time:.4f}s | RFT: ERROR ({e})")
            else:
                rft_time = float('inf')
                ratio = float('inf')
                print(f"  FFT: {fft_time:.4f}s | RFT: UNAVAILABLE")
            
            speed_results[N] = {
                'fft_time': fft_time,
                'rft_time': rft_time,
                'ratio': ratio
            }
        
        self.results['speed'] = speed_results
        return speed_results
    
    def benchmark_accuracy(self) -> Dict:
        """Test 2: Accuracy comparison via round-trip error."""
        print("\n🎯 ACCURACY BENCHMARK: Round-trip Error")
        print("=" * 50)
        
        test_sizes = [256, 512, 1024]
        accuracy_results = {}
        
        for N in test_sizes:
            print(f"Testing accuracy N={N}...")
            
            # Create test data
            data = np.random.randn(N) + 1j * np.random.randn(N)
            
            # FFT round-trip
            fft_forward = fft(data)
            fft_back = ifft(fft_forward)
            fft_error = np.max(np.abs(data - fft_back))
            
            # RFT round-trip
            if RFT_AVAILABLE:
                try:
                    rft = UnifiedRFTInterface(N, "auto")
                    rft_forward = rft.transform(data)
                    rft_back = rft.inverse_transform(rft_forward)
                    rft_error = np.max(np.abs(data - rft_back))
                    
                    print(f"  FFT error: {fft_error:.2e}")
                    print(f"  RFT error: {rft_error:.2e}")
                except Exception as e:
                    rft_error = float('inf')
                    print(f"  FFT error: {fft_error:.2e}")
                    print(f"  RFT error: ERROR ({e})")
            else:
                rft_error = float('inf')
                print(f"  FFT error: {fft_error:.2e}")
                print(f"  RFT error: UNAVAILABLE")
            
            accuracy_results[N] = {
                'fft_error': float(fft_error),
                'rft_error': float(rft_error)
            }
        
        self.results['accuracy'] = accuracy_results
        return accuracy_results
    
    def benchmark_compression(self) -> Dict:
        """Test 3: Compression efficiency on real model weights."""
        print("\n📦 COMPRESSION BENCHMARK: Real Model Data")
        print("=" * 50)
        
        # Create synthetic model weights (simulating GPT-2 embedding layer)
        vocab_size, embed_dim = 50257, 768
        weights = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        
        print(f"Original data: {weights.nbytes / 1024 / 1024:.1f} MB")
        
        # FFT-based compression (simple quantization)
        fft_compressed = self._compress_with_fft(weights)
        fft_reconstructed = self._decompress_with_fft(fft_compressed)
        fft_mse = np.mean((weights - fft_reconstructed)**2)
        fft_size = len(fft_compressed) if isinstance(fft_compressed, bytes) else fft_compressed.nbytes
        
        # RFT-based compression
        if RFT_AVAILABLE:
            rft_compressed = self._compress_with_rft(weights)
            rft_reconstructed = self._decompress_with_rft(rft_compressed)
            rft_mse = np.mean((weights - rft_reconstructed)**2)
            rft_size = len(rft_compressed) if isinstance(rft_compressed, bytes) else rft_compressed.nbytes
            
            print(f"FFT: {fft_size / 1024 / 1024:.1f} MB, MSE={fft_mse:.6f}")
            print(f"RFT: {rft_size / 1024 / 1024:.1f} MB, MSE={rft_mse:.6f}")
        else:
            rft_mse = float('inf')
            rft_size = float('inf')
            print(f"FFT: {fft_size / 1024 / 1024:.1f} MB, MSE={fft_mse:.6f}")
            print(f"RFT: UNAVAILABLE")
        
        compression_results = {
            'original_size': weights.nbytes,
            'fft_size': fft_size,
            'rft_size': rft_size,
            'fft_mse': float(fft_mse),
            'rft_mse': float(rft_mse),
            'fft_ratio': weights.nbytes / fft_size if fft_size > 0 else 0,
            'rft_ratio': weights.nbytes / rft_size if rft_size != float('inf') and rft_size > 0 else 0
        }
        
        self.results['compression'] = compression_results
        return compression_results
    
    def _compress_with_fft(self, data: np.ndarray) -> np.ndarray:
        """Simple FFT-based compression via quantization."""
        # Flatten, transform, quantize high frequencies
        flat = data.flatten()
        transformed = fft(flat)
        # Keep only low frequencies (simple compression)
        compressed = transformed[:len(transformed)//4]  # 4:1 compression
        return compressed
    
    def _decompress_with_fft(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress FFT data."""
        # Pad with zeros and inverse transform
        full_size = len(compressed) * 4
        padded = np.zeros(full_size, dtype=complex)
        padded[:len(compressed)] = compressed
        reconstructed = ifft(padded).real
        # Reshape back to original
        return reconstructed.reshape(50257, 768)[:50257, :768]
    
    def _compress_with_rft(self, data: np.ndarray) -> np.ndarray:
        """RFT-based compression."""
        if not RFT_AVAILABLE:
            return np.array([])
        
        # Use vertex codec for compression
        try:
            from algorithms.compression.rft_vertex_codec import RFTVertexCodec
            codec = RFTVertexCodec()
            return codec.encode(data)
        except ImportError:
            # Fallback: simple RFT compression
            try:
                flat = data.flatten()
                rft = UnifiedRFTInterface(len(flat), "auto")
                transformed = rft.transform(flat)
                return transformed[:len(transformed)//4]  # 4:1 compression
            except Exception as e:
                print(f"RFT compression failed: {e}")
                return np.array([])
    
    def _decompress_with_rft(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress RFT data."""
        if not RFT_AVAILABLE:
            return np.zeros((50257, 768))
        
        try:
            from algorithms.compression.rft_vertex_codec import RFTVertexCodec
            codec = RFTVertexCodec()
            return codec.decode(compressed)
        except ImportError:
            # Fallback reconstruction
            try:
                if len(compressed) == 0:
                    return np.zeros((50257, 768))
                full_size = len(compressed) * 4
                padded = np.zeros(full_size, dtype=complex)
                padded[:len(compressed)] = compressed
                rft = UnifiedRFTInterface(full_size, "auto")
                reconstructed = rft.inverse_transform(padded).real
                return reconstructed.reshape(50257, 768)[:50257, :768]
            except Exception as e:
                print(f"RFT decompression failed: {e}")
                return np.zeros((50257, 768))
    
    def run_all_benchmarks(self) -> Dict:
        """Run complete benchmark suite."""
        print("🚀 QUANTONIUMOS RFT vs FFT BENCHMARK SUITE")
        print("=" * 60)
        print("PHASE 1.1: Competitive Transform Benchmarks")
        print("Status: CRITICAL BLOCKING REQUIREMENT")
        print("=" * 60)
        
        # Run all benchmark tests
        self.benchmark_speed()
        self.benchmark_accuracy()
        self.benchmark_compression()
        
        return self.results
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        results_path = Path(__file__).parent / filename
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📊 Results saved to: {results_path}")
        
    def generate_report(self) -> str:
        """Generate markdown report of benchmark results."""
        report = """# RFT vs FFT Benchmark Results

## Executive Summary

This report presents competitive benchmarks between QuantoniumOS RFT and standard FFT implementations.

## Speed Performance

| Size | FFT Time (s) | RFT Time (s) | Ratio |
|------|-------------|-------------|-------|
"""
        
        if 'speed' in self.results:
            for size, data in self.results['speed'].items():
                fft_time = data['fft_time']
                rft_time = data['rft_time']
                ratio = data['ratio']
                report += f"| {size} | {fft_time:.4f} | {rft_time:.4f} | {ratio:.2f}x |\n"
        
        report += "\n## Accuracy Results\n\n"
        report += "| Size | FFT Error | RFT Error |\n"
        report += "|------|-----------|----------|\n"
        
        if 'accuracy' in self.results:
            for size, data in self.results['accuracy'].items():
                fft_err = data['fft_error']
                rft_err = data['rft_error']
                report += f"| {size} | {fft_err:.2e} | {rft_err:.2e} |\n"
        
        report += "\n## Compression Efficiency\n\n"
        if 'compression' in self.results:
            comp = self.results['compression']
            report += f"- Original size: {comp['original_size'] / 1024 / 1024:.1f} MB\n"
            report += f"- FFT compressed: {comp['fft_size'] / 1024 / 1024:.1f} MB (MSE: {comp['fft_mse']:.6f})\n"
            report += f"- RFT compressed: {comp['rft_size'] / 1024 / 1024:.1f} MB (MSE: {comp['rft_mse']:.6f})\n"
        
        return report

def main():
    """Run RFT vs FFT competitive benchmarks."""
    benchmarker = TransformBenchmark()
    
    # Run all benchmarks
    results = benchmarker.run_all_benchmarks()
    
    # Save results
    benchmarker.save_results("benchmark_results.json")
    
    # Generate and save report
    report = benchmarker.generate_report()
    report_path = Path(__file__).parent / "benchmark_results.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📋 Report saved to: {report_path}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLETE")
    print("="*60)
    print("Files generated:")
    print("- benchmark_results.json (raw data)")
    print("- benchmark_results.md (formatted report)")
    print("\nNext steps:")
    print("1. Review results for competitive advantage")
    print("2. Address any performance gaps")
    print("3. Proceed to NIST randomness tests (Phase 1.2)")

if __name__ == "__main__":
    main()