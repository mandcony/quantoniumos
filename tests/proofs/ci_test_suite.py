#!/usr/bin/env python3
"""
CI-READY TEST SUITE
===================
Comprehensive CI-ready test suite for bulletproof validation.
"""

import subprocess
import sys
import os
import time

def run_test(test_name, test_file):
    """Run a single test and return results"""
    print(f"\n🧪 RUNNING: {test_name}")
    print("=" * 60)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name}: PASS ({elapsed:.1f}s)")
            return True, elapsed, result.stdout
        else:
            print(f"❌ {test_name}: FAIL ({elapsed:.1f}s)")
            print(f"Error: {result.stderr}")
            return False, elapsed, result.stderr
    
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name}: TIMEOUT (300s)")
        return False, 300, "Test timed out"
    except Exception as e:
        print(f"💥 {test_name}: ERROR - {e}")
        return False, 0, str(e)

def main():
    """Run comprehensive CI test suite"""
    
    print("🚀 CI-READY COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Running all validation tests for bulletproof CI integration")
    print("=" * 80)
    
    # Test suite configuration
    tests = [
        ("Scaling Analysis", "scaling_analysis.py"),
        ("Parity Validation", "parity_test.py"), 
        ("Build Info Check", "build_info_test.py"),
        ("Assembly Engine Validation", "test_corrected_assembly.py"),
    ]
    
    results = []
    total_time = 0
    
    for test_name, test_file in tests:
        if os.path.exists(test_file):
            success, elapsed, output = run_test(test_name, test_file)
            results.append((test_name, success, elapsed, output))
            total_time += elapsed
        else:
            print(f"⚠️ {test_name}: SKIPPED (file not found: {test_file})")
            results.append((test_name, False, 0, f"File not found: {test_file}"))
    
    # Summary report
    print(f"\n" + "=" * 80)
    print("CI TEST SUITE SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success, _, _ in results if success)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    print(f"Total time:   {total_time:.1f}s")
    print()
    
    print("Individual Results:")
    for test_name, success, elapsed, _ in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:25s}: {status} ({elapsed:.1f}s)")
    
    # CI exit code
    if passed == total:
        print(f"\n🎉 CI-READY: ALL TESTS PASS")
        print(f"✅ Ready for continuous integration")
        print(f"✅ Bulletproof validation complete")
        return 0
    else:
        print(f"\n⚠️ CI-BLOCKED: {total-passed} TESTS FAILED")
        print(f"❌ Fix failing tests before CI integration")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
