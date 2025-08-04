# Cryptographic Algorithm Benchmark Suite
# Compare QuantoniumOS algorithms against standard implementations

import time
import os
import json
import hashlib
import secrets
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse

# Try to import common crypto libraries
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    has_cryptography = True
except ImportError:
    has_cryptography = False

try:
    from Crypto.Cipher import AES as PyCryptoAES
    has_pycrypto = True
except ImportError:
    has_pycrypto = False

# Import QuantoniumOS libraries
try:
    from quantoniumos import resonance_encrypt, resonance_decrypt, geometric_wave_hash
    has_quantoniumos = True
except ImportError:
    print("Warning: QuantoniumOS libraries not found. Only standard algorithms will be tested.")
    has_quantoniumos = False

# Configuration
ITERATIONS = 100  # Number of iterations for each test
DATA_SIZES = [64, 256, 1024, 4096, 16384]  # Data sizes in bytes
OUTPUT_DIR = "benchmark_results"

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def generate_test_data(size):
    """Generate random test data of the specified size"""
    return secrets.token_bytes(size)

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function execution time"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return (end_time - start_time) * 1000, result  # Return time in milliseconds

# Standard encryption algorithms
def aes_encrypt_cryptography(data, key):
    """AES encryption using cryptography library"""
    if not has_cryptography:
        return None
    
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Pad data to 16-byte blocks
    padded_data = data + b'\x00' * (16 - len(data) % 16) if len(data) % 16 else data
    
    ct = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ct

def aes_encrypt_pycrypto(data, key):
    """AES encryption using PyCrypto library"""
    if not has_pycrypto:
        return None
    
    iv = os.urandom(16)
    cipher = PyCryptoAES.new(key, PyCryptoAES.MODE_CBC, iv)
    
    # Pad data to 16-byte blocks
    padded_data = data + b'\x00' * (16 - len(data) % 16) if len(data) % 16 else data
    
    ct = cipher.encrypt(padded_data)
    return iv + ct

# Hashing algorithms
def sha256_hash(data):
    """SHA-256 hash"""
    return hashlib.sha256(data).digest()

def sha3_256_hash(data):
    """SHA3-256 hash"""
    return hashlib.sha3_256(data).digest()

def run_encryption_benchmarks():
    """Run encryption benchmarks"""
    results = []
    key = os.urandom(32)  # 256-bit key for all algorithms
    
    for size in DATA_SIZES:
        data = generate_test_data(size)
        
        # Standard AES encryption
        if has_cryptography:
            aes_cryptography_times = []
            for _ in range(ITERATIONS):
                time_ms, _ = benchmark_function(aes_encrypt_cryptography, data, key)
                aes_cryptography_times.append(time_ms)
            results.append({
                'algorithm': 'AES-256 (cryptography)',
                'data_size': size,
                'avg_time_ms': sum(aes_cryptography_times) / len(aes_cryptography_times),
                'min_time_ms': min(aes_cryptography_times),
                'max_time_ms': max(aes_cryptography_times)
            })
        
        if has_pycrypto:
            aes_pycrypto_times = []
            for _ in range(ITERATIONS):
                time_ms, _ = benchmark_function(aes_encrypt_pycrypto, data, key)
                aes_pycrypto_times.append(time_ms)
            results.append({
                'algorithm': 'AES-256 (PyCrypto)',
                'data_size': size,
                'avg_time_ms': sum(aes_pycrypto_times) / len(aes_pycrypto_times),
                'min_time_ms': min(aes_pycrypto_times),
                'max_time_ms': max(aes_pycrypto_times)
            })
        
        # QuantoniumOS Resonance encryption
        if has_quantoniumos:
            quantonium_times = []
            for _ in range(ITERATIONS):
                time_ms, _ = benchmark_function(resonance_encrypt, data, key.hex())
                quantonium_times.append(time_ms)
            results.append({
                'algorithm': 'QuantoniumOS Resonance',
                'data_size': size,
                'avg_time_ms': sum(quantonium_times) / len(quantonium_times),
                'min_time_ms': min(quantonium_times),
                'max_time_ms': max(quantonium_times)
            })
    
    return results

def run_hash_benchmarks():
    """Run hash benchmarks"""
    results = []
    
    for size in DATA_SIZES:
        data = generate_test_data(size)
        
        # Standard SHA-256
        sha256_times = []
        for _ in range(ITERATIONS):
            time_ms, _ = benchmark_function(sha256_hash, data)
            sha256_times.append(time_ms)
        results.append({
            'algorithm': 'SHA-256',
            'data_size': size,
            'avg_time_ms': sum(sha256_times) / len(sha256_times),
            'min_time_ms': min(sha256_times),
            'max_time_ms': max(sha256_times)
        })
        
        # SHA3-256
        sha3_times = []
        for _ in range(ITERATIONS):
            time_ms, _ = benchmark_function(sha3_256_hash, data)
            sha3_times.append(time_ms)
        results.append({
            'algorithm': 'SHA3-256',
            'data_size': size,
            'avg_time_ms': sum(sha3_times) / len(sha3_times),
            'min_time_ms': min(sha3_times),
            'max_time_ms': max(sha3_times)
        })
        
        # QuantoniumOS Geometric Wave Hash
        if has_quantoniumos:
            geo_wave_times = []
            for _ in range(ITERATIONS):
                time_ms, _ = benchmark_function(geometric_wave_hash, data)
                geo_wave_times.append(time_ms)
            results.append({
                'algorithm': 'QuantoniumOS Geometric Wave',
                'data_size': size,
                'avg_time_ms': sum(geo_wave_times) / len(geo_wave_times),
                'min_time_ms': min(geo_wave_times),
                'max_time_ms': max(geo_wave_times)
            })
    
    return results

def run_avalanche_benchmarks():
    """Run avalanche effect benchmarks"""
    results = []
    key = os.urandom(32)
    
    # Only test a single data size for avalanche effect
    size = 1024
    data = generate_test_data(size)
    
    # Create a second data set with a single bit change
    data_modified = bytearray(data)
    data_modified[0] ^= 0x01  # Flip the first bit
    data_modified = bytes(data_modified)
    
    # For each algorithm, compare outputs with original and modified input
    if has_quantoniumos:
        # Benchmark QuantoniumOS avalanche effect
        original_output = resonance_encrypt(data, key.hex())
        modified_output = resonance_encrypt(data_modified, key.hex())
        
        # Skip signature and token (first 40 bytes)
        original_output = original_output[40:]
        modified_output = modified_output[40:]
        
        # Count number of bits that differ
        bit_changes = 0
        for b1, b2 in zip(original_output, modified_output):
            xor = b1 ^ b2
            # Count bits in XOR result
            bit_changes += bin(xor).count('1')
        
        total_bits = len(original_output) * 8
        avalanche_percentage = (bit_changes / total_bits) * 100
        
        results.append({
            'algorithm': 'QuantoniumOS Resonance',
            'data_size': size,
            'bit_changes': bit_changes,
            'total_bits': total_bits,
            'avalanche_percentage': avalanche_percentage
        })
    
    if has_cryptography:
        # Benchmark AES avalanche effect
        key_hex = key.hex()
        original_output = aes_encrypt_cryptography(data, key)
        modified_output = aes_encrypt_cryptography(data_modified, key)
        
        # Skip IV (first 16 bytes)
        original_output = original_output[16:]
        modified_output = modified_output[16:]
        
        # Count number of bits that differ
        bit_changes = 0
        for b1, b2 in zip(original_output, modified_output):
            xor = b1 ^ b2
            bit_changes += bin(xor).count('1')
        
        total_bits = len(original_output) * 8
        avalanche_percentage = (bit_changes / total_bits) * 100
        
        results.append({
            'algorithm': 'AES-256',
            'data_size': size,
            'bit_changes': bit_changes,
            'total_bits': total_bits,
            'avalanche_percentage': avalanche_percentage
        })
    
    return results

def save_results(results, filename):
    """Save results to a JSON file"""
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        json.dump(results, f, indent=2)

def save_csv(results, filename, result_type):
    """Save results to a CSV file"""
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        if result_type == "encryption" or result_type == "hashing":
            f.write("algorithm,data_size,avg_time_ms,min_time_ms,max_time_ms\n")
            for r in results:
                f.write(f"{r['algorithm']},{r['data_size']},{r['avg_time_ms']:.4f},{r['min_time_ms']:.4f},{r['max_time_ms']:.4f}\n")
        elif result_type == "avalanche":
            f.write("algorithm,data_size,bit_changes,total_bits,avalanche_percentage\n")
            for r in results:
                f.write(f"{r['algorithm']},{r['data_size']},{r['bit_changes']},{r['total_bits']},{r['avalanche_percentage']:.2f}\n")

def generate_plots(encryption_results, hash_results, avalanche_results, timestamp):
    """Generate plots for visualization"""
    # Prepare data for encryption benchmark plot
    algorithms = set(r['algorithm'] for r in encryption_results)
    data_sizes = sorted(set(r['data_size'] for r in encryption_results))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in algorithms:
        algo_results = [r for r in encryption_results if r['algorithm'] == algo]
        algo_results.sort(key=lambda x: x['data_size'])
        x = [r['data_size'] for r in algo_results]
        y = [r['avg_time_ms'] for r in algo_results]
        ax.plot(x, y, marker='o', label=algo)
    
    ax.set_xlabel('Data Size (bytes)')
    ax.set_ylabel('Average Time (ms)')
    ax.set_title('Encryption Performance Comparison')
    ax.legend()
    ax.grid(True)
    ax.set_xscale('log', base=2)
    plt.savefig(os.path.join(OUTPUT_DIR, f"encryption_benchmark_{timestamp}.png"))
    
    # Prepare data for hash benchmark plot
    algorithms = set(r['algorithm'] for r in hash_results)
    data_sizes = sorted(set(r['data_size'] for r in hash_results))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in algorithms:
        algo_results = [r for r in hash_results if r['algorithm'] == algo]
        algo_results.sort(key=lambda x: x['data_size'])
        x = [r['data_size'] for r in algo_results]
        y = [r['avg_time_ms'] for r in algo_results]
        ax.plot(x, y, marker='o', label=algo)
    
    ax.set_xlabel('Data Size (bytes)')
    ax.set_ylabel('Average Time (ms)')
    ax.set_title('Hash Function Performance Comparison')
    ax.legend()
    ax.grid(True)
    ax.set_xscale('log', base=2)
    plt.savefig(os.path.join(OUTPUT_DIR, f"hashing_benchmark_{timestamp}.png"))
    
    # Prepare data for avalanche effect bar chart
    if avalanche_results:
        algorithms = [r['algorithm'] for r in avalanche_results]
        percentages = [r['avalanche_percentage'] for r in avalanche_results]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(algorithms, percentages)
        ax.set_ylabel('Avalanche Effect (%)')
        ax.set_title('Avalanche Effect Comparison')
        ax.grid(True, axis='y')
        # Add ideal 50% reference line
        ax.axhline(y=50, color='r', linestyle='--', label='Ideal (50%)')
        ax.legend()
        
        # Add percentage labels on top of bars
        for i, v in enumerate(percentages):
            ax.text(i, v + 1, f"{v:.2f}%", ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"avalanche_benchmark_{timestamp}.png"))

def print_results(encryption_results, hash_results, avalanche_results):
    """Print results to the console in a readable format"""
    print("\n===== ENCRYPTION BENCHMARKS =====")
    encryption_table = []
    for r in sorted(encryption_results, key=lambda x: (x['data_size'], x['algorithm'])):
        encryption_table.append([
            r['algorithm'], 
            f"{r['data_size']} bytes", 
            f"{r['avg_time_ms']:.2f} ms",
            f"{r['min_time_ms']:.2f} ms",
            f"{r['max_time_ms']:.2f} ms"
        ])
    print(tabulate(encryption_table, headers=["Algorithm", "Data Size", "Avg Time", "Min Time", "Max Time"]))
    
    print("\n===== HASH BENCHMARKS =====")
    hash_table = []
    for r in sorted(hash_results, key=lambda x: (x['data_size'], x['algorithm'])):
        hash_table.append([
            r['algorithm'], 
            f"{r['data_size']} bytes", 
            f"{r['avg_time_ms']:.2f} ms",
            f"{r['min_time_ms']:.2f} ms",
            f"{r['max_time_ms']:.2f} ms"
        ])
    print(tabulate(hash_table, headers=["Algorithm", "Data Size", "Avg Time", "Min Time", "Max Time"]))
    
    print("\n===== AVALANCHE EFFECT =====")
    avalanche_table = []
    for r in avalanche_results:
        avalanche_table.append([
            r['algorithm'],
            f"{r['data_size']} bytes",
            f"{r['bit_changes']} / {r['total_bits']}",
            f"{r['avalanche_percentage']:.2f}%"
        ])
    print(tabulate(avalanche_table, headers=["Algorithm", "Data Size", "Bit Changes", "Avalanche Effect"]))

def generate_summary(encryption_results, hash_results, avalanche_results, timestamp):
    """Generate a summary of the benchmark results"""
    summary = {
        "timestamp": timestamp,
        "iterations": ITERATIONS,
        "data_sizes": DATA_SIZES,
        "encryption": {
            "algorithms": list(set(r['algorithm'] for r in encryption_results)),
            "fastest_algorithm": None,
            "slowest_algorithm": None,
        },
        "hashing": {
            "algorithms": list(set(r['algorithm'] for r in hash_results)),
            "fastest_algorithm": None,
            "slowest_algorithm": None,
        },
        "avalanche": {
            "results": avalanche_results
        }
    }
    
    # Find fastest and slowest algorithms for medium size data (1024 bytes)
    medium_size = 1024
    encryption_medium = [r for r in encryption_results if r['data_size'] == medium_size]
    if encryption_medium:
        fastest = min(encryption_medium, key=lambda x: x['avg_time_ms'])
        slowest = max(encryption_medium, key=lambda x: x['avg_time_ms'])
        summary["encryption"]["fastest_algorithm"] = {
            "name": fastest['algorithm'],
            "avg_time_ms": fastest['avg_time_ms']
        }
        summary["encryption"]["slowest_algorithm"] = {
            "name": slowest['algorithm'],
            "avg_time_ms": slowest['avg_time_ms']
        }
        
        # Calculate speed ratios
        if "QuantoniumOS" in fastest['algorithm']:
            summary["encryption"]["quantonium_vs_standard"] = "faster"
            summary["encryption"]["speed_ratio"] = slowest['avg_time_ms'] / fastest['avg_time_ms']
        elif any("QuantoniumOS" in algo['algorithm'] for algo in encryption_medium):
            summary["encryption"]["quantonium_vs_standard"] = "slower"
            quantonium = next(algo for algo in encryption_medium if "QuantoniumOS" in algo['algorithm'])
            standard = fastest
            summary["encryption"]["speed_ratio"] = quantonium['avg_time_ms'] / standard['avg_time_ms']
    
    # Find fastest and slowest hash algorithms for medium size data (1024 bytes)
    hash_medium = [r for r in hash_results if r['data_size'] == medium_size]
    if hash_medium:
        fastest = min(hash_medium, key=lambda x: x['avg_time_ms'])
        slowest = max(hash_medium, key=lambda x: x['avg_time_ms'])
        summary["hashing"]["fastest_algorithm"] = {
            "name": fastest['algorithm'],
            "avg_time_ms": fastest['avg_time_ms']
        }
        summary["hashing"]["slowest_algorithm"] = {
            "name": slowest['algorithm'],
            "avg_time_ms": slowest['avg_time_ms']
        }
        
        # Calculate speed ratios
        if "QuantoniumOS" in fastest['algorithm']:
            summary["hashing"]["quantonium_vs_standard"] = "faster"
            summary["hashing"]["speed_ratio"] = slowest['avg_time_ms'] / fastest['avg_time_ms']
        elif any("QuantoniumOS" in algo['algorithm'] for algo in hash_medium):
            summary["hashing"]["quantonium_vs_standard"] = "slower"
            quantonium = next(algo for algo in hash_medium if "QuantoniumOS" in algo['algorithm'])
            standard = fastest
            summary["hashing"]["speed_ratio"] = quantonium['avg_time_ms'] / standard['avg_time_ms']
    
    # Avalanche effect analysis
    if avalanche_results:
        quantonium_avalanche = next((r for r in avalanche_results if "QuantoniumOS" in r['algorithm']), None)
        standard_avalanche = next((r for r in avalanche_results if "QuantoniumOS" not in r['algorithm']), None)
        
        if quantonium_avalanche and standard_avalanche:
            summary["avalanche"]["comparison"] = {
                "quantonium": quantonium_avalanche['avalanche_percentage'],
                "standard": standard_avalanche['avalanche_percentage'],
                "difference": abs(quantonium_avalanche['avalanche_percentage'] - standard_avalanche['avalanche_percentage'])
            }
            
            # Determine which is closer to ideal 50%
            quantonium_distance = abs(quantonium_avalanche['avalanche_percentage'] - 50)
            standard_distance = abs(standard_avalanche['avalanche_percentage'] - 50)
            
            if quantonium_distance < standard_distance:
                summary["avalanche"]["closer_to_ideal"] = "QuantoniumOS"
            elif standard_distance < quantonium_distance:
                summary["avalanche"]["closer_to_ideal"] = standard_avalanche['algorithm']
            else:
                summary["avalanche"]["closer_to_ideal"] = "Equal"
    
    return summary

def run_benchmarks():
    """Run all benchmarks and generate reports"""
    ensure_output_dir()
    
    print("Running cryptographic benchmarks...")
    print(f"Iterations per test: {ITERATIONS}")
    print(f"Data sizes: {DATA_SIZES} bytes")
    
    # Generate timestamp for file naming
    timestamp = int(time.time())
    
    print("\nRunning encryption benchmarks...")
    encryption_results = run_encryption_benchmarks()
    
    print("Running hash benchmarks...")
    hash_results = run_hash_benchmarks()
    
    print("Running avalanche effect benchmarks...")
    avalanche_results = run_avalanche_benchmarks()
    
    # Print results to console
    print_results(encryption_results, hash_results, avalanche_results)
    
    # Save results to files
    save_results(encryption_results, f"encryption_benchmark_{timestamp}.json")
    save_results(hash_results, f"hashing_benchmark_{timestamp}.json")
    save_results(avalanche_results, f"avalanche_benchmark_{timestamp}.json")
    
    save_csv(encryption_results, f"encryption_benchmark_{timestamp}.csv", "encryption")
    save_csv(hash_results, f"hashing_benchmark_{timestamp}.csv", "hashing")
    save_csv(avalanche_results, f"avalanche_benchmark_{timestamp}.csv", "avalanche")
    
    # Generate summary
    summary = generate_summary(encryption_results, hash_results, avalanche_results, timestamp)
    save_results(summary, f"benchmark_summary_{timestamp}.json")
    
    # Generate plots
    generate_plots(encryption_results, hash_results, avalanche_results, timestamp)
    
    print(f"\nBenchmark complete. Results saved to {OUTPUT_DIR}/")
    print(f"Summary file: benchmark_summary_{timestamp}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cryptographic benchmarks')
    parser.add_argument('--iterations', type=int, default=ITERATIONS, help='Number of iterations per test')
    parser.add_argument('--sizes', nargs='+', type=int, default=DATA_SIZES, help='Data sizes to test (in bytes)')
    
    args = parser.parse_args()
    
    ITERATIONS = args.iterations
    DATA_SIZES = args.sizes
    
    run_benchmarks()
