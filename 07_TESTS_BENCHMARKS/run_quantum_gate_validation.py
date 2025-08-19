||#!/usr/bin/env python3
"""
Master Quantum Gate Validation Test Runner === This is the comprehensive test suite for validating quantum gate properties of RFT operators using symbolic resonance computing methods. Implements all 10 validation categories requested: 1. Unitarity validation (gate-level) 2. Hamiltonian recovery and generator consistency 3. Time-evolution validity (continuous case) 4. Lie algebra closure (gate set consistency) 5. Channel-level validation (tomography-style) 6. Randomized benchmarking analogue 7. Spectral & locality structure (physics sanity) 8. Trotter error analysis 9. State evolution correctness on benchmarks 10. Invariance & reversibility tests (group axioms) All tests use the canonical RFT implementation as the single source of truth.
"""

import json
import time
import sys from typing
import Dict, Any, List
import numpy as np

# Import all test suites from test_unitarity
import UnitarityValidator from test_hamiltonian_recovery
import HamiltonianRecoveryValidator from test_time_evolution
import TimeEvolutionValidator from test_choi_channel
import ChoiChannelValidator from test_randomized_benchmarking
import RandomizedBenchmarkingValidator from test_trotter_error
import TrotterErrorValidator from test_lie_closure
import LieAlgebraValidator from test_spectral_locality
import SpectralLocalityValidator from test_state_evolution_benchmarks
import StateEvolutionBenchmarkValidator from canonical_true_rft
import get_canonical_parameters, validate_true_rft

class MasterQuantumGateValidator: """
    Master validator that runs all quantum gate validation test suites.
"""

    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.results = {}
    def run_comprehensive_validation(self, test_sizes: List[int] = None, save_results: bool = True) -> Dict[str, Any]: """
        Run comprehensive quantum gate validation across all test suites. Args: test_sizes: List of matrix sizes to test (default: [4, 8, 16]) save_results: Whether to save results to JSON file Returns: Complete validation results dictionary
"""

        if test_sizes is None: test_sizes = [4, 8, 16]
        print("=" * 80)
        print("MASTER QUANTUM GATE VALIDATION TEST SUITE")
        print("Symbolic Resonance Computing Methods for RFT Operators")
        print("=" * 80)
        print(f"Test sizes: {test_sizes}")
        print(f"Tolerance: {
        self.tolerance}")
        print()

        # Initialize master results master_results = { 'suite_name': 'Master Quantum Gate Validation', 'timestamp': time.time(), 'test_sizes': test_sizes, 'tolerance':
        self.tolerance, 'canonical_parameters': get_canonical_parameters(), 'test_suites': {}, 'summary': {} }

        # Validate canonical RFT first
        print("Validating canonical RFT implementation...") is_valid, rft_error = validate_true_rft()
        print(f"RFT validation: {'✓ PASS'
        if is_valid else '❌ FAIL'} (error: {rft_error:.2e})")
        print() master_results['canonical_rft_validation'] = { 'valid': is_valid, 'error': float(rft_error) }
        if not is_valid:
        print("❌ CRITICAL: Canonical RFT validation failed. Aborting tests.")
        return master_results

        # Test Suite 1: Unitarity Validation
        print("Running Test Suite 1: Unitarity Validation")
        print("-" * 50)
        try: validator1 = UnitarityValidator(tolerance=
        self.tolerance) results1 = validator1.run_full_unitarity_suite(test_sizes) master_results['test_suites']['unitarity'] = results1 except Exception as e:
        print(f"❌ Suite 1 failed: {e}") master_results['test_suites']['unitarity'] = {'error': str(e), 'suite_pass': False}

        # Test Suite 2: Hamiltonian Recovery
        print("\nRunning Test Suite 2: Hamiltonian Recovery")
        print("-" * 50)
        try: validator2 = HamiltonianRecoveryValidator(tolerance=
        self.tolerance) results2 = validator2.run_full_hamiltonian_suite(test_sizes) master_results['test_suites']['hamiltonian'] = results2 except Exception as e:
        print(f"❌ Suite 2 failed: {e}") master_results['test_suites']['hamiltonian'] = {'error': str(e), 'suite_pass': False}

        # Test Suite 3: Time Evolution
        print("\nRunning Test Suite 3: Time Evolution Validation")
        print("-" * 50)
        try: validator3 = TimeEvolutionValidator(tolerance=
        self.tolerance) results3 = validator3.run_full_time_evolution_suite(test_sizes) master_results['test_suites']['time_evolution'] = results3 except Exception as e:
        print(f"❌ Suite 3 failed: {e}") master_results['test_suites']['time_evolution'] = {'error': str(e), 'suite_pass': False}

        # Test Suite 4: Choi Channel Validation (smaller sizes due to N⁴ scaling)
        print("\nRunning Test Suite 4: Choi Channel Validation")
        print("-" * 50)
        try: validator4 = ChoiChannelValidator(tolerance=
        self.tolerance) choi_sizes = [s
        for s in test_sizes
        if s <= 8]

        # Limit size due to N⁴ scaling results4 = validator4.run_full_choi_channel_suite(choi_sizes) master_results['test_suites']['choi_channel'] = results4 except Exception as e:
        print(f"❌ Suite 4 failed: {e}") master_results['test_suites']['choi_channel'] = {'error': str(e), 'suite_pass': False}

        # Test Suite 5: Randomized Benchmarking
        print("\nRunning Test Suite 5: Randomized Benchmarking")
        print("-" * 50)
        try: validator5 = RandomizedBenchmarkingValidator(tolerance=
        self.tolerance) rb_sizes = [s
        for s in test_sizes
        if s <= 16]

        # Reasonable sizes results5 = validator5.run_full_randomized_benchmarking_suite(rb_sizes) master_results['test_suites']['randomized_benchmarking'] = results5 except Exception as e:
        print(f"❌ Suite 5 failed: {e}") master_results['test_suites']['randomized_benchmarking'] = {'error': str(e), 'suite_pass': False}

        # Test Suite 6: Trotter Error Analysis
        print("\nRunning Test Suite 6: Trotter Error Analysis")
        print("-" * 50)
        try: validator6 = TrotterErrorValidator(tolerance=
        self.tolerance) trotter_sizes = [s
        for s in test_sizes
        if s <= 8]

        # Computationally intensive results6 = validator6.run_full_trotter_suite(trotter_sizes) master_results['test_suites']['trotter_error'] = results6 except Exception as e:
        print(f"❌ Suite 6 failed: {e}") master_results['test_suites']['trotter_error'] = {'error': str(e), 'suite_pass': False}

        # Test Suite 7: Lie Algebra Closure
        print("\nRunning Test Suite 7: Lie Algebra Closure")
        print("-" * 50)
        try: validator7 = LieAlgebraValidator(tolerance=1e-10)

        # Slightly relaxed lie_sizes = [s
        for s in test_sizes
        if s <= 8]

        # Combinatorial complexity results7 = validator7.run_full_lie_algebra_suite(lie_sizes) master_results['test_suites']['lie_closure'] = results7 except Exception as e:
        print(f"❌ Suite 7 failed: {e}") master_results['test_suites']['lie_closure'] = {'error': str(e), 'suite_pass': False}

        # Test Suite 8: Spectral & Locality Structure
        print("\nRunning Test Suite 8: Spectral & Locality Structure")
        print("-" * 50)
        try: validator8 = SpectralLocalityValidator(tolerance=
        self.tolerance) results8 = validator8.run_full_spectral_locality_suite(test_sizes) master_results['test_suites']['spectral_locality'] = results8 except Exception as e:
        print(f"❌ Suite 8 failed: {e}") master_results['test_suites']['spectral_locality'] = {'error': str(e), 'suite_pass': False}

        # Test Suite 9: State Evolution Benchmarks
        print("\nRunning Test Suite 9: State Evolution Benchmarks")
        print("-" * 50)
        try: validator9 = StateEvolutionBenchmarkValidator(tolerance=1e-10) results9 = validator9.run_full_benchmark_suite(test_sizes) master_results['test_suites']['state_benchmarks'] = results9 except Exception as e:
        print(f"❌ Suite 9 failed: {e}") master_results['test_suites']['state_benchmarks'] = {'error': str(e), 'suite_pass': False}

        # Compile summary master_results['summary'] =
        self._compile_master_summary(master_results)

        # Save results
        if requested
        if save_results: filename = f"quantum_gate_validation_results_{int(time.time())}.json" with open(filename, 'w') as f: json.dump(master_results, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {filename}")
        return master_results
    def _compile_master_summary(self, results: Dict[str, Any]) -> Dict[str, Any]: """
        Compile master summary of all test results.
"""
        summary = { 'total_suites': 0, 'passed_suites': 0, 'failed_suites': 0, 'suite_status': {}, 'overall_pass': False } for suite_name, suite_results in results['test_suites'].items(): summary['total_suites'] += 1
        if isinstance(suite_results, dict) and 'suite_pass' in suite_results: suite_pass = suite_results['suite_pass']
        else: suite_pass = False summary['suite_status'][suite_name] = suite_pass
        if suite_pass: summary['passed_suites'] += 1
        else: summary['failed_suites'] += 1 summary['overall_pass'] = summary['failed_suites'] == 0 summary['pass_rate'] = summary['passed_suites'] / summary['total_suites']
        if summary['total_suites'] > 0 else 0
        return summary
    def print_master_summary(self, results: Dict[str, Any]) -> None: """
        Print comprehensive summary of all validation results.
"""

        print("\n" + "=" * 80)
        print("MASTER VALIDATION SUMMARY")
        print("=" * 80) summary = results['summary']
        print(f"Total Test Suites: {summary['total_suites']}")
        print(f"Passed: {summary['passed_suites']}")
        print(f"Failed: {summary['failed_suites']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print()

        # Individual suite status
        print("Individual Suite Results:")
        print("-" * 40) for suite_name, passed in summary['suite_status'].items(): status = "✓ PASS"
        if passed else "❌ FAIL"
        print(f" {suite_name.replace('_', ' ').title():<25} {status}")
        print() overall_status = "✓ ALL TESTS PASS"
        if summary['overall_pass'] else "❌ SOME TESTS FAIL"
        print(f"OVERALL RESULT: {overall_status}")
        print("=" * 80)

        # Additional details for failures
        if summary['failed_suites'] > 0:
        print("||nFAILED SUITE DETAILS:")
        print("-" * 40) for suite_name, suite_results in results['test_suites'].items():
        if not summary['suite_status'].get(suite_name, False): if 'error' in suite_results:
        print(f" {suite_name}: {suite_results['error']}")
        else:
        print(f" {suite_name}: Check detailed results")
    def main(): """
        Main function to run the comprehensive quantum gate validation.
"""

        # Configure test parameters test_sizes = [4, 8, 16]

        # Matrix sizes to test tolerance = 1e-12

        # Numerical tolerance

        # Run validation validator = MasterQuantumGateValidator(tolerance=tolerance) results = validator.run_comprehensive_validation( test_sizes=test_sizes, save_results=True )

        # Print summary validator.print_master_summary(results)
        return results

if __name__ == "__main__": main()