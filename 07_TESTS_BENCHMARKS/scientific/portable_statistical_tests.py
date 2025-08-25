"""
Statistical test suite for QuantoniumOS encryption using built-in Python tools
"""

import hashlib
import json
import math
import os
import time
from datetime import datetime
from typing import Dict

from optimized_resonance_encrypt import optimized_resonance_encrypt


class StatisticalTester:
    def __init__(self, sample_size_mb: int = 10):
        self.sample_size = sample_size_mb * 1024 * 1024 * 8  # Convert MB to bits
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("test_results", "statistical", self.timestamp)
        os.makedirs(self.results_dir, exist_ok=True)

    def generate_test_data(self) -> bytes:
        """Generate test data using our encryption"""
        print(f"Generating {self.sample_size // 8 // 1024 // 1024}MB test data...")

        key = hashlib.sha256(str(time.time()).encode()).hexdigest()
        chunk_size = 1024 * 1024  # 1MB chunks
        data = bytearray()

        while len(data) < (self.sample_size // 8):
            plaintext = os.urandom(chunk_size)
            ciphertext = optimized_resonance_encrypt(plaintext.hex(), key)
            data.extend(ciphertext[40:])  # Skip signature and token

            if len(data) % (10 * 1024 * 1024) == 0:  # Every 10MB
                print(f"Generated {len(data) // 1024 // 1024}MB...")

        return bytes(data[: self.sample_size // 8])

    def monobit_test(self, data: bytes) -> Dict:
        """Frequency (Monobit) Test"""
        ones = sum(bin(b).count("1") for b in data)
        n = len(data) * 8
        s_obs = abs(ones - (n / 2)) / math.sqrt(n / 4)
        p_value = math.erfc(s_obs / math.sqrt(2))
        return {
            "name": "Frequency (Monobit)",
            "p_value": p_value,
            "pass": p_value >= 0.01,
            "statistics": {"ones": ones, "zeros": n - ones, "total_bits": n},
        }

    def block_frequency_test(self, data: bytes, block_size: int = 128) -> Dict:
        """Block Frequency Test"""
        n = len(data) * 8
        num_blocks = n // block_size
        if num_blocks == 0:
            return {"error": "Sample size too small for block frequency test"}

        proportions = []
        for i in range(num_blocks):
            block_start = i * block_size // 8
            block_end = (i + 1) * block_size // 8
            block = data[block_start:block_end]
            ones = sum(bin(b).count("1") for b in block)
            proportions.append(ones / block_size)

        chi_sq = 4 * block_size * sum((p - 0.5) ** 2 for p in proportions)
        p_value = math.erfc(math.sqrt(chi_sq / 2))

        return {
            "name": "Block Frequency",
            "p_value": p_value,
            "pass": p_value >= 0.01,
            "statistics": {
                "block_size": block_size,
                "num_blocks": num_blocks,
                "chi_square": chi_sq,
            },
        }

    def runs_test(self, data: bytes) -> Dict:
        """Runs Test"""
        bits = "".join(format(b, "08b") for b in data)
        n = len(bits)

        ones = bits.count("1")
        pi = ones / n

        if abs(pi - 0.5) >= (2 / math.sqrt(n)):
            return {
                "name": "Runs",
                "p_value": 0.0,
                "pass": False,
                "error": "Prerequisite frequency test failed",
            }

        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i - 1]:
                runs += 1

        p_value = math.erfc(
            abs(runs - (2 * n * pi * (1 - pi))) / (2 * math.sqrt(2 * n) * pi * (1 - pi))
        )

        return {
            "name": "Runs",
            "p_value": p_value,
            "pass": p_value >= 0.01,
            "statistics": {"runs": runs, "pi": pi, "total_bits": n},
        }

    def longest_run_ones_test(self, data: bytes) -> Dict:
        """Longest Run of Ones Test"""
        bits = "".join(format(b, "08b") for b in data)
        n = len(bits)

        # Find longest run of ones
        current_run = 0
        longest_run = 0
        for bit in bits:
            if bit == "1":
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0

        # Calculate test statistic
        k = int(math.log2(n))
        v = longest_run

        # Theoretical probabilities for longest runs
        pi = [0.2148, 0.3672, 0.2305, 0.1875]

        # Map v to categories
        if v <= k - 2:
            category = 0
        elif v == k - 1:
            category = 1
        elif v == k:
            category = 2
        else:
            category = 3

        # Chi-square test
        chi_sq = ((category - (n * pi[category])) ** 2) / (n * pi[category])
        p_value = math.exp(-chi_sq / 2)

        return {
            "name": "Longest Run of Ones",
            "p_value": p_value,
            "pass": p_value >= 0.01,
            "statistics": {"longest_run": longest_run, "chi_square": chi_sq},
        }

    def run_all_tests(self, data: bytes) -> Dict[str, Dict]:
        """Run all statistical tests"""
        results = {}

        # Run tests
        results["monobit"] = self.monobit_test(data)
        results["block_frequency"] = self.block_frequency_test(data)
        results["runs"] = self.runs_test(data)
        results["longest_run"] = self.longest_run_ones_test(data)

        return results

    def run_full_battery(self) -> Dict:
        """Run complete test battery and save results"""
        start_time = time.time()

        # Generate test data
        data = self.generate_test_data()

        # Run all tests
        print("\nRunning statistical tests...")
        results = self.run_all_tests(data)

        # Compile results
        final_results = {
            "timestamp": self.timestamp,
            "sample_size_mb": self.sample_size // 8 // 1024 // 1024,
            "duration_seconds": time.time() - start_time,
            "tests": results,
        }

        # Save results
        results_file = os.path.join(self.results_dir, "statistical_tests.json")
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)

        return final_results


def print_results_summary(results: Dict):
    """Print formatted summary of test results"""
    print("\nStatistical Test Results")
    print("=" * 50)

    tests = results["tests"]
    passed = sum(1 for t in tests.values() if t["pass"])

    for name, test in tests.items():
        status = "✓ PASS" if test["pass"] else "✗ FAIL"
        p_value = test["p_value"]
        print(f"{test['name']:25} {status:8} p={p_value:.4f}")

    print("\nSummary:")
    print(f"Tests Passed : {passed}/{len(tests)}")
    print(f"Sample Size : {results['sample_size_mb']} MB")
    print(f"Duration : {results['duration_seconds']:.1f} seconds")


if __name__ == "__main__":
    tester = StatisticalTester(sample_size_mb=10)
    results = tester.run_full_battery()
    print_results_summary(results)
