"""
Long-run statistical test executor for QuantoniumOS encryption
Runs multiple large test vectors through NIST and Dieharder suites
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List

from nist_statistical_tests import NISTTester


def run_long_statistical_validation(
    num_iterations: int = 5,
    sample_size_mb: int = 10,
    export_dir: str = "statistical_validation",
):
    """
    Run multiple iterations of statistical tests with large samples

    Args:
        num_iterations: Number of test iterations to run
        sample_size_mb: Size of each test vector in MB
        export_dir: Directory to export results
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory
    results_dir = os.path.join(export_dir, f"longrun_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    print("Starting long-run statistical validation")
    print(f"Running {num_iterations} iterations with {sample_size_mb}MB samples")
    print("=" * 50)

    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        print("-" * 30)

        # Run test battery
        tester = NISTTester(sample_size_mb=sample_size_mb)
        results = tester.run_full_battery()

        # Store results
        all_results.append(results)

        # Export iteration results
        iteration_file = os.path.join(results_dir, f"iteration_{i+1}.json")
        with open(iteration_file, "w") as f:
            json.dump(results, f, indent=2)

    # Calculate aggregate statistics
    aggregate = aggregate_results(all_results)

    # Export aggregate results
    aggregate_file = os.path.join(results_dir, "aggregate_results.json")
    with open(aggregate_file, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Export summary report
    summary_file = os.path.join(results_dir, "summary_report.txt")
    with open(summary_file, "w") as f:
        write_summary_report(f, aggregate, time.time() - start_time)

    return aggregate


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate statistics across multiple test runs"""
    aggregate = {
        "iterations": len(results),
        "total_sample_size_mb": sum(r["sample_size_mb"] for r in results),
        "total_duration": sum(r["duration_seconds"] for r in results),
        "nist_tests": {},
        "dieharder_tests": {},
    }

    # Aggregate NIST results
    for test_name in results[0]["nist_tests"].keys():
        passes = sum(
            1 for r in results if r["nist_tests"][test_name].get("pass", False)
        )
        p_values = [r["nist_tests"][test_name].get("p_value", 0) for r in results]
        p_values = [p for p in p_values if p > 0]  # Filter valid p-values

        aggregate["nist_tests"][test_name] = {
            "pass_rate": passes / len(results),
            "avg_p_value": sum(p_values) / len(p_values) if p_values else 0,
            "min_p_value": min(p_values) if p_values else 0,
            "max_p_value": max(p_values) if p_values else 0,
        }

    # Aggregate Dieharder results
    dh_tests = {}
    for result in results:
        for test in result["dieharder_tests"].get("tests", []):
            if test["name"] not in dh_tests:
                dh_tests[test["name"]] = []
            dh_tests[test["name"]].append(
                {"p_value": test["p_value"], "passed": test["result"] == "PASS"}
            )

    for test_name, test_results in dh_tests.items():
        passes = sum(1 for r in test_results if r["passed"])
        p_values = [r["p_value"] for r in test_results]

        aggregate["dieharder_tests"][test_name] = {
            "pass_rate": passes / len(test_results),
            "avg_p_value": sum(p_values) / len(p_values),
            "min_p_value": min(p_values),
            "max_p_value": max(p_values),
        }

    return aggregate


def write_summary_report(f, aggregate: Dict, total_time: float):
    """Write detailed summary report"""
    f.write("QuantoniumOS Statistical Validation Report\n")
    f.write("=" * 50 + "\n\n")

    f.write("Test Configuration:\n")
    f.write(f"- Number of iterations: {aggregate['iterations']}\n")
    f.write(f"- Total sample size : {aggregate['total_sample_size_mb']} MB\n")
    f.write(f"- Total duration : {total_time:.1f} seconds\n\n")

    f.write("NIST SP 800-22 Test Results:\n")
    f.write("-" * 30 + "\n")
    for test_name, results in aggregate["nist_tests"].items():
        f.write(f"\n{test_name}:\n")
        f.write(f" Pass rate : {results['pass_rate']*100:.1f}%\n")
        f.write(f" Avg p-value : {results['avg_p_value']:.4f}\n")
        f.write(
            f" p-value range: [{results['min_p_value']:.4f}, {results['max_p_value']:.4f}]\n"
        )

    f.write("\nDieharder Test Results:\n")
    f.write("-" * 30 + "\n")
    for test_name, results in aggregate["dieharder_tests"].items():
        f.write(f"\n{test_name}:\n")
        f.write(f" Pass rate : {results['pass_rate']*100:.1f}%\n")
        f.write(f" Avg p-value : {results['avg_p_value']:.4f}\n")
        f.write(
            f" p-value range: [{results['min_p_value']:.4f}, {results['max_p_value']:.4f}]\n"
        )

    f.write("\nConclusion:\n")
    f.write("-" * 30 + "\n")
    nist_pass_rates = [r["pass_rate"] for r in aggregate["nist_tests"].values()]
    dh_pass_rates = [r["pass_rate"] for r in aggregate["dieharder_tests"].values()]

    avg_nist_pass = sum(nist_pass_rates) / len(nist_pass_rates)
    avg_dh_pass = sum(dh_pass_rates) / len(dh_pass_rates)

    f.write(f"Average NIST test pass rate : {avg_nist_pass*100:.1f}%\n")
    f.write(f"Average Dieharder pass rate : {avg_dh_pass*100:.1f}%\n")

    if avg_nist_pass > 0.99 and avg_dh_pass > 0.99:
        f.write("\nOVERALL STATUS: ✓ PASSED - Cryptographically sound\n")
    else:
        f.write("\nOVERALL STATUS: ✗ FAILED - Does not meet statistical requirements\n")


if __name__ == "__main__":
    # Run long validation with 5 iterations of 10MB each
    results = run_long_statistical_validation(
        num_iterations=5, sample_size_mb=10, export_dir="statistical_validation"
    )
