#!/usr/bin/env python3
"""
Debug script to find why scientific validation is failing.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_scientific_test_suite import (ScientificRFTTestSuite,
                                                 TestConfiguration)


def debug_validation_failure():
    """Debug why all tests are showing 0% success rate."""

    print("DEBUG: Analyzing validation failure")
    print("=" * 50)

    # Create test suite with minimal config
    config = TestConfiguration(
        dimension_range=[8],  # Just test one small dimension
        precision_tolerance=1e-6,  # Looser tolerance
        num_trials=1,  # Just one trial
        statistical_significance=0.1,  # Looser significance
    )

    suite = ScientificRFTTestSuite(config)

    # Test each domain individually to see what's wrong
    domains = [
        ("MATHEMATICAL", "test_asymptotic_complexity"),
        ("SIGNAL_PROCESSING", "test_compression_benchmarks"),
        ("CRYPTOGRAPHY", "test_entropy_analysis"),
        ("QUANTUM", "test_large_scale_entanglement"),
        ("INFORMATION_THEORY", "test_channel_capacity_analysis"),
    ]

    for domain_name, test_method in domains:
        print(f"\nTesting {domain_name}:")
        try:
            method = getattr(suite, test_method)
            result = method()

            print(f"  Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"  Keys: {list(result.keys())}")
                if "test_passed" in result:
                    print(f"  Test passed: {result['test_passed']}")
                if "success" in result:
                    print(f"  Success: {result['success']}")
                if "validation_passed" in result:
                    print(f"  Validation passed: {result['validation_passed']}")

                # Look for any boolean success indicators
                success_indicators = [
                    k for k, v in result.items() if isinstance(v, bool) and v
                ]
                failure_indicators = [
                    k for k, v in result.items() if isinstance(v, bool) and not v
                ]
                print(f"  Success indicators: {success_indicators}")
                print(f"  Failure indicators: {failure_indicators}")
            else:
                print(f"  Result: {result}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    print("\nDEBUG: Checking how success is determined...")

    # Check the validation summary logic
    try:
        results = suite.run_comprehensive_scientific_validation()
        print(f"Full results type: {type(results)}")
        if isinstance(results, dict):
            print(f"Full results keys: {list(results.keys())}")

            # Look for the summary calculation
            for domain in [
                "MATHEMATICAL",
                "SIGNAL_PROCESSING",
                "CRYPTOGRAPHY",
                "QUANTUM",
                "INFORMATION_THEORY",
            ]:
                if domain in results:
                    domain_result = results[domain]
                    print(f"{domain}: {type(domain_result)} - {domain_result}")

    except Exception as e:
        print(f"ERROR in full validation: {e}")


if __name__ == "__main__":
    debug_validation_failure()
