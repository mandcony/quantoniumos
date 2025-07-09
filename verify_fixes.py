#!/usr/bin/env python3
"""
Verification script to confirm all 7 originally failing tests now pass individually.
"""

import subprocess
import sys
import time

# List of the 7 originally failing tests
FIXED_TESTS = [
    "tests/test_waveform_hash.py::test_wave_hash_length_and_extract",
    "tests/test_security.py::TestSecurityHeaders::test_csp_default_src", 
    "tests/test_security.py::TestSecurityHeaders::test_csp_frame_ancestors",
    "tests/test_security.py::TestSecurityHeaders::test_csp_img_src",
    "tests/test_stream.py::SSEContractTest::test_stream_endpoint_headers",
    "tests/test_stream.py::SSEContractTest::test_stream_receives_events", 
    "tests/test_rft_stability_and_performance.py::TestRFTStabilityAndPerformance::test_rft_numerical_stability"
]

def run_test(test_name):
    """Run a single test and return whether it passed."""
    print(f"Testing: {test_name}")
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_name, "-v", "--tb=no"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"✅ PASSED: {test_name}")
            return True
        else:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {result.stdout}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {test_name}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {test_name} - {e}")
        return False

def main():
    print("🔍 Verifying all 7 originally failing tests now pass...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test in FIXED_TESTS:
        if run_test(test):
            passed += 1
        else:
            failed += 1
        print()  # Add spacing
        time.sleep(1)  # Brief pause between tests
    
    print("=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed out of {len(FIXED_TESTS)} tests")
    
    if failed == 0:
        print("🎉 ALL TESTS FIXED! All 7 originally failing tests now pass.")
        return 0
    else:
        print(f"⚠️  {failed} tests still failing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
