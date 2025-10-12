import numpy as np
import time
from scipy.fft import fft
import sys
import os

# Add project root to path to allow importing project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    # First, try to import the high-performance C/ASM implementation
    from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY
    print("✅ Successfully imported C/ASM UnitaryRFT for benchmarking.")
    RFT_IMPLEMENTATION = "Hybrid C/ASM via Python"
except ImportError:
    print("⚠️ C/ASM UnitaryRFT not found. Falling back to Python implementation.")
    try:
        # Fallback to the pure Python implementation
        from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
        print("✅ Successfully imported Python CanonicalTrueRFT for benchmarking.")
        RFT_IMPLEMENTATION = "Python"
    except ImportError as e:
        print(f"❌ Failed to import any RFT implementation: {e}")
        sys.exit(1)

def run_benchmark(sizes):
    """
    Compares the performance of RFT and FFT for various input sizes.
    """
    print("=" * 80)
    print(f"🚀 RFT vs. FFT Performance Benchmark")
    print(f"   - RFT Implementation: {RFT_IMPLEMENTATION}")
    if RFT_IMPLEMENTATION == "Hybrid C/ASM via Python":
        print("   - Testing the hybrid model: Python wrapper calling compiled C/ASM kernels.")
    else:
        print("   - WARNING: Testing the pure Python model. This is not a fair performance comparison.")
    print("=" * 80)
    print(f"{'Input Size':<15} {'RFT Time (ms)':<20} {'FFT Time (ms)':<20} {'Winner':<20}")
    print("-" * 80)

    results = {"rft": [], "fft": []}

    for size in sizes:
        # Create random complex data
        data = np.random.rand(size) + 1j * np.random.rand(size)
        
        # --- RFT Benchmark ---
        rft_time = 0
        try:
            if RFT_IMPLEMENTATION == "Hybrid C/ASM via Python":
                # Benchmark C/ASM RFT
                rft_engine = UnitaryRFT(size, RFT_FLAG_UNITARY)
                start_time = time.perf_counter()
                rft_engine.forward(data)
                end_time = time.perf_counter()
                rft_time = (end_time - start_time) * 1000  # Convert to ms
            else:
                # Benchmark Python RFT
                rft_engine = CanonicalTrueRFT(size)
                start_time = time.perf_counter()
                rft_engine.forward_transform(data)
                end_time = time.perf_counter()
                rft_time = (end_time - start_time) * 1000  # Convert to ms
        except Exception as e:
            print(f"Error during RFT benchmark at size {size}: {e}")
            rft_time = float('inf') # Mark as failed

        results["rft"].append(rft_time)

        # --- FFT Benchmark ---
        start_time = time.perf_counter()
        fft(data)
        end_time = time.perf_counter()
        fft_time = (end_time - start_time) * 1000  # Convert to ms
        results["fft"].append(fft_time)

        # --- Print Results ---
        winner = "FFT" if fft_time < rft_time else "RFT"
        if rft_time == float('inf'):
            winner = "FFT (RFT Error)"
        
        print(f"{size:<15} {rft_time:<20.4f} {fft_time:<20.4f} {winner:<10}")

    return results

def main():
    # Define a range of input sizes for the benchmark
    # Using powers of 2, which is optimal for FFT
    benchmark_sizes = [2**i for i in range(8, 16)] # From 256 to 32768
    
    run_benchmark(benchmark_sizes)
    
    print("\n" + "="*60)
    print("Benchmark Complete.")
    print("This test compares the execution time of the forward transform.")
    if RFT_IMPLEMENTATION == "Python":
        print("Note: RFT was tested using the pure Python implementation, which is expected to be slower.")
        print("For a fair comparison, the C/ASM kernels must be compiled and accessible.")
    else:
        print("Note: RFT was tested using the high-performance C/ASM implementation.")
    print("="*60)

if __name__ == "__main__":
    main()
