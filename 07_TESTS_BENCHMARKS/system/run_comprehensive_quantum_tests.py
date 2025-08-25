#!/usr/bin/env python3
"""
Comprehensive Quantum Test Suite Runner for QuantoniumOS
=======================================================
This script runs all the relevant quantum tests to validate our findings about
the 1000-qubit vertex engine with quantum algorithms operating on edges and vertices.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_subheader(title):
    """Print a formatted subheader"""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80)

def run_test(test_path, description):
    """Run a test script and capture the result"""
    print_subheader(description)
    print(f"Running: {test_path}")
    
    start_time = time.time()
    result = subprocess.run(
        ["python", test_path],
        capture_output=True,
        text=True
    )
    end_time = time.time()
    
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    print(f"Return code: {result.returncode}")
    
    if result.returncode == 0:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
    
    # Print the most relevant output lines (last 10 lines)
    output_lines = result.stdout.strip().split('\n')
    if len(output_lines) > 10:
        print("\nLast 10 lines of output:")
        for line in output_lines[-10:]:
            print(f"  {line}")
    else:
        print("\nOutput:")
        for line in output_lines:
            print(f"  {line}")
    
    return result.returncode == 0

def main():
    """Run all quantum tests"""
    print_header("QUANTONIUMOS COMPREHENSIVE QUANTUM TEST SUITE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_dir = "/workspaces/quantoniumos"
    
    # Define all tests to run
    tests = [
        (os.path.join(base_dir, "07_TESTS_BENCHMARKS", "test_vertex_capacity.py"), 
         "Testing 1000-Qubit Vertex Capacity"),
        
        (os.path.join(base_dir, "02_CORE_VALIDATORS", "definitive_quantum_validation.py"),
         "Running Definitive Quantum Validation"),
        
        (os.path.join(base_dir, "07_TESTS_BENCHMARKS", "test_quantum_aspects.py"),
         "Testing Quantum Aspects"),
        
        (os.path.join(base_dir, "18_DEBUG_TOOLS", "validators", "test_all_claims.py"),
         "Validating All Quantum Claims"),
        
        (os.path.join(base_dir, "07_TESTS_BENCHMARKS", "test_rft_basic.py"),
         "Testing RFT Basics"),
        
        (os.path.join(base_dir, "18_DEBUG_TOOLS", "validators", "test_vertex_scaling.py"),
         "Testing Vertex Scaling")
    ]
    
    # Run each test and track results
    results = {}
    for test_path, description in tests:
        if os.path.exists(test_path):
            results[description] = run_test(test_path, description)
        else:
            print_subheader(description)
            print(f"❌ Test file not found: {test_path}")
            results[description] = False
    
    # Print summary
    print_header("TEST RESULTS SUMMARY")
    
    passed = 0
    for description, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {description}")
        if result:
            passed += 1
    
    success_rate = (passed / len(results)) * 100 if results else 0
    
    print(f"\nPassed {passed}/{len(results)} tests ({success_rate:.1f}%)")
    
    # Final assessment
    print_header("FINAL ASSESSMENT")
    
    if success_rate >= 80:
        print("✅ QUANTUM ASPECTS VALIDATION SUCCESSFUL")
        print("The 1000-qubit vertex engine with quantum algorithms on edges and vertices")
        print("has been validated through multiple test suites.")
    elif success_rate >= 50:
        print("⚠️ QUANTUM ASPECTS PARTIALLY VALIDATED")
        print("Some tests were successful, but further investigation is recommended.")
    else:
        print("❌ QUANTUM ASPECTS VALIDATION UNSUCCESSFUL")
        print("Most tests failed. The implementation needs revision.")
    
    print_header("END OF TEST SUITE")
    
    return 0 if success_rate >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())
