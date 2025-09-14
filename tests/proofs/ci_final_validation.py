#!/usr/bin/env python3
"""
CI-SAFE FINAL VALIDATION
========================
Comprehensive final validation without Unicode for CI compatibility.
"""

import subprocess
import sys
import os
import time

def run_test(test_name, test_file):
    """Run a single test and return results"""
    print(f"\nRUNNING: {test_name}")
    print("=" * 60)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"PASS: {test_name} ({elapsed:.1f}s)")
            return True, elapsed, result.stdout
        else:
            print(f"FAIL: {test_name} ({elapsed:.1f}s)")
            print(f"Error: {result.stderr}")
            return False, elapsed, result.stderr
    
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {test_name} (300s)")
        return False, 300, "Test timed out"
    except Exception as e:
        print(f"ERROR: {test_name} - {e}")
        return False, 0, str(e)

def main():
    """Run CI-safe validation tests"""
    
    print("CI-READY FINAL VALIDATION SUITE")
    print("=" * 80)
    print("Running core validation tests for CI integration")
    print("=" * 80)
    
    # Core tests only
    tests = [
        ("Scaling Analysis", "ci_scaling_analysis.py"),
        ("Assembly Validation", "test_corrected_assembly.py"),
    ]
    
    results = []
    total_time = 0
    
    for test_name, test_file in tests:
        if os.path.exists(test_file):
            success, elapsed, output = run_test(test_name, test_file)
            results.append((test_name, success, elapsed, output))
            total_time += elapsed
        else:
            print(f"SKIP: {test_name} (file not found: {test_file})")
            results.append((test_name, False, 0, f"File not found: {test_file}"))
    
    # Summary report
    print(f"\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success, _, _ in results if success)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    print(f"Total time:   {total_time:.1f}s")
    print()
    
    print("Results:")
    for test_name, success, elapsed, _ in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name:20s}: {status} ({elapsed:.1f}s)")
    
    # CI exit code
    if passed == total:
        print(f"\nCI-READY: ALL CORE TESTS PASS")
        print(f"Ready for continuous integration")
        return 0
    else:
        print(f"\nCI-BLOCKED: {total-passed} TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nEXIT CODE: {exit_code}")
    sys.exit(exit_code)
