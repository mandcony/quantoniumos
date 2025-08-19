""""""
NIST Statistical Test Suite runner for QuantoniumOS encryption
Generates large test vectors and runs complete NIST SP 800-22 battery
""""""

import os
import time
import hashlib
import subprocess
from datetime import datetime
from typing import List, Dict, Any
import json
from optimized_resonance_encrypt import optimized_resonance_encrypt

class NISTTester:
    def __init__(self, sample_size_mb: int = 10):
        self.sample_size = sample_size_mb * 1024 * 1024 * 8  # Convert MB to bits
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("test_results", "nist_sts", self.timestamp)
        os.makedirs(self.results_dir, exist_ok=True)

    def generate_test_data(self) -> str:
        """"""Generate large test vector using our encryption""""""
        print(f"Generating {self.sample_size // 8 // 1024 // 1024}MB test data...")

        output_file = os.path.join(self.results_dir, "test_vector.bin")
        key = hashlib.sha256(str(time.time()).encode()).hexdigest()

        # Generate data in chunks to manage memory
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(output_file, "wb") as f:
            bytes_written = 0
            while bytes_written < (self.sample_size // 8):
                # Generate random plaintext
                plaintext = os.urandom(chunk_size)
                # Encrypt it
                ciphertext = optimized_resonance_encrypt(plaintext.hex(), key)
                # Write only the ciphertext portion (skip signature and token)
                f.write(ciphertext[40:])
                bytes_written += len(ciphertext) - 40

                if bytes_written % (10 * 1024 * 1024) == 0:  # Every 10MB
                    print(f"Generated {bytes_written // 1024 // 1024}MB...")

        return output_file

    def run_nist_tests(self, data_file: str) -> Dict[str, Any]:
        """"""Run full NIST SP 800-22 test suite""""""
        print("\nRunning NIST Statistical Test Suite...")

        # NIST tests to run and their parameters
        tests = [
            ("Frequency", []),
            ("BlockFrequency", ["128"]),
            ("Runs", []),
            ("LongestRun", []),
            ("Rank", []),
            ("FFT", []),
            ("NonOverlappingTemplate", ["9"]),
            ("OverlappingTemplate", ["9"]),
            ("Universal", []),
            ("LinearComplexity", ["500"]),
            ("Serial", ["16"]),
            ("ApproximateEntropy", ["10"]),
            ("CumulativeSums", []),
            ("RandomExcursions", []),
            ("RandomExcursionsVariant", [])
        ]

        results = {}
        for test_name, params in tests:
            print(f"Running {test_name} test...")
            cmd = ["assess", str(self.sample_size), data_file, test_name] + params
            try:
                output = subprocess.check_output(cmd, text=True)
                p_value = self.extract_p_value(output)
                results[test_name] = {
                    "p_value": p_value,
                    "pass": p_value >= 0.01,
                    "raw_output": output
                }
            except subprocess.CalledProcessError as e:
                results[test_name] = {
                    "error": str(e),
                    "pass": False,
                    "raw_output": e.output if hasattr(e, 'output') else None
                }

        return results

    def run_dieharder_tests(self, data_file: str) -> Dict[str, Any]:
        """"""Run Dieharder test suite""""""
        print("\nRunning Dieharder test suite...")

        results = {}
        cmd = ["dieharder", "-a", "-f", data_file]
        try:
            output = subprocess.check_output(cmd, text=True)
            tests = self.parse_dieharder_output(output)
            results = {
                "tests": tests,
                "raw_output": output
            }
        except subprocess.CalledProcessError as e:
            results = {
                "error": str(e),
                "raw_output": e.output if hasattr(e, 'output') else None
            }

        return results

    def extract_p_value(self, output: str) -> float:
        """"""Extract p-value from NIST test output""""""
        try:
            p_value_line = [l for l in output.split('\n') if 'P-value' in l][0]
            return float(p_value_line.split('=')[1].strip())
        except:
            return 0.0

    def parse_dieharder_output(self, output: str) -> List[Dict[str, Any]]:
        """"""Parse Dieharder test results""""""
        tests = []
        for line in output.split('\n'):
            if '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) >= 4 and parts[0].strip() and 'PASSED' in line:
                tests.append({
                    "name": parts[0].strip(),
                    "p_value": float(parts[2].strip()),
                    "result": "PASS"
                })
            elif len(parts) >= 4 and parts[0].strip() and 'FAILED' in line:
                tests.append({
                    "name": parts[0].strip(),
                    "p_value": float(parts[2].strip()),
                    "result": "FAIL"
                })
        return tests

    def run_full_battery(self):
        """"""Run complete test battery and save results""""""
        start_time = time.time()

        # Generate test data
        data_file = self.generate_test_data()

        # Run both test suites
        nist_results = self.run_nist_tests(data_file)
        dieharder_results = self.run_dieharder_tests(data_file)

        # Compile results
        results = {
            "timestamp": self.timestamp,
            "sample_size_mb": self.sample_size // 8 // 1024 // 1024,
            "duration_seconds": time.time() - start_time,
            "nist_tests": nist_results,
            "dieharder_tests": dieharder_results
        }

        # Save detailed results
        results_file = os.path.join(self.results_dir, "statistical_tests.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save raw data file location
        with open(os.path.join(self.results_dir, "test_info.txt"), 'w') as f:
            f.write(f"Test vector file: {data_file}\n")
            f.write(f"Results file: {results_file}\n")
            f.write(f"Sample size: {self.sample_size // 8 // 1024 // 1024}MB\n")
            f.write(f"Duration: {results['duration_seconds']:.2f} seconds\n")

        return results

def summarize_results(results: Dict[str, Any]):
    """"""Print summary of test results""""""
    print("\nTest Results Summary")
    print("=" * 50)

    print("\nNIST SP 800-22 Tests:")
    print("-" * 30)
    nist_passed = 0
    total_nist = len(results["nist_tests"])
    for test_name, data in results["nist_tests"].items():
        if data.get("pass", False):
            nist_passed += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        p_value = data.get("p_value", "N/A")
        print(f"{test_name:25} {status:8} p={p_value:.4f}")

    print("\nDieharder Tests:")
    print("-" * 30)
    dh_passed = 0
    dh_tests = results["dieharder_tests"].get("tests", [])
    for test in dh_tests:
        if test["result"] == "PASS":
            dh_passed += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{test['name']:25} {status:8} p={test['p_value']:.4f}")

    print("\nOverall Summary:")
    print("-" * 30)
    print(f"NIST Tests Passed : {nist_passed}/{total_nist}")
    print(f"Dieharder Tests Passed: {dh_passed}/{len(dh_tests)}")
    print(f"Total Duration : {results['duration_seconds']:.1f} seconds")
    print(f"Sample Size : {results['sample_size_mb']} MB")

if __name__ == "__main__":
    # Run with larger sample size for thorough testing
    tester = NISTTester(sample_size_mb=10)  # 10MB minimum for reliable results
    results = tester.run_full_battery()
    summarize_results(results)
