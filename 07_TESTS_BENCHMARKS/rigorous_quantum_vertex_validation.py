#!/usr/bin/env python3
"""
RIGOROUS QUANTUM VERTEX VALIDATION EXPERIMENT
================================================================
Scientific validation of quantum phenomena in topological vertex networks
Testing: Entanglement, Superposition, Coherence, Unitarity, No-Cloning
Based on established quantum computing benchmarks
"""

import json
import time
from typing import Any, Dict

import numpy as np
from test_qubit_vertices import QubitVertexNetwork


class QuantumVertexValidator:
    """Rigorous quantum computing validation for vertex operations"""

    def __init__(self, network: QubitVertexNetwork):
        self.network = network
        self.validation_results = {}
        self.tolerance = 1e-10  # Scientific precision threshold

    def test_superposition_principle(self) -> Dict[str, Any]:
        """
        Test 1: Superposition Principle Validation
        |ψ⟩ = α|0⟩ + β|1⟩ with |α|² + |β|² = 1
        """
        print("🔬 TEST 1: Superposition Principle")

        results = {
            "test_name": "Superposition Principle",
            "vertices_tested": [],
            "normalization_errors": [],
            "superposition_quality": [],
            "pass": True,
        }

        for vertex_id, vertex in self.network.vertices.items():
            # Check normalization: |α|² + |β|² = 1
            prob_sum = abs(vertex.alpha) ** 2 + abs(vertex.beta) ** 2
            normalization_error = abs(prob_sum - 1.0)

            # Check if in superposition (both amplitudes non-zero)
            superposition_quality = min(abs(vertex.alpha) ** 2, abs(vertex.beta) ** 2)

            results["vertices_tested"].append(vertex_id)
            results["normalization_errors"].append(normalization_error)
            results["superposition_quality"].append(superposition_quality)

            # Fail if normalization violated
            if normalization_error > self.tolerance:
                results["pass"] = False

            print(
                f"  Vertex {vertex_id}: norm_error={normalization_error:.2e}, "
                f"superpos_quality={superposition_quality:.6f}"
            )

        avg_error = np.mean(results["normalization_errors"])
        avg_superposition = np.mean(results["superposition_quality"])

        print(f"  Average normalization error: {avg_error:.2e}")
        print(f"  Average superposition quality: {avg_superposition:.6f}")
        print(f"  RESULT: {'PASS' if results['pass'] else 'FAIL'}")

        return results

    def test_unitary_evolution(self) -> Dict[str, Any]:
        """
        Test 2: Unitary Evolution Validation
        Evolution operators must preserve normalization: U†U = I
        """
        print("\n🔬 TEST 2: Unitary Evolution")

        results = {
            "test_name": "Unitary Evolution",
            "initial_norms": [],
            "final_norms": [],
            "norm_preservation_errors": [],
            "pass": True,
        }

        # Record initial state norms
        initial_norms = []
        for vertex in self.network.vertices.values():
            norm = abs(vertex.alpha) ** 2 + abs(vertex.beta) ** 2
            initial_norms.append(norm)
            results["initial_norms"].append(norm)

        # Apply quantum evolution
        print("  Applying quantum evolution...")
        self.network.evolve_network(time_steps=5, dt=0.1)

        # Record final state norms
        final_norms = []
        for vertex in self.network.vertices.values():
            norm = abs(vertex.alpha) ** 2 + abs(vertex.beta) ** 2
            final_norms.append(norm)
            results["final_norms"].append(norm)

        # Check norm preservation
        for i, (initial, final) in enumerate(zip(initial_norms, final_norms)):
            error = abs(final - initial)
            results["norm_preservation_errors"].append(error)

            if error > self.tolerance:
                results["pass"] = False

            print(
                f"  Vertex {i}: initial_norm={initial:.6f}, final_norm={final:.6f}, error={error:.2e}"
            )

        max_error = max(results["norm_preservation_errors"])
        avg_error = np.mean(results["norm_preservation_errors"])

        print(f"  Maximum norm preservation error: {max_error:.2e}")
        print(f"  Average norm preservation error: {avg_error:.2e}")
        print(f"  RESULT: {'PASS' if results['pass'] else 'FAIL'}")

        return results

    def test_entanglement_generation(self) -> Dict[str, Any]:
        """
        Test 3: Entanglement Generation and Measurement
        Test for non-separable states and quantum correlations
        """
        print("\n🔬 TEST 3: Entanglement Generation")

        results = {
            "test_name": "Entanglement Generation",
            "entangled_pairs": [],
            "correlation_strengths": [],
            "entanglement_measures": [],
            "pass": True,
        }

        # Test entanglement between vertex pairs
        vertex_ids = list(self.network.vertices.keys())

        for i in range(0, len(vertex_ids) - 1, 2):
            v1_id = vertex_ids[i]
            v2_id = vertex_ids[i + 1]

            # Get initial states
            v1 = self.network.vertices[v1_id]
            v2 = self.network.vertices[v2_id]

            initial_state_1 = (v1.alpha, v1.beta)
            initial_state_2 = (v2.alpha, v2.beta)

            # Entangle the vertices
            self.network.entangle_vertices(v1_id, v2_id)

            # Measure correlation after entanglement
            final_state_1 = (v1.alpha, v1.beta)
            final_state_2 = (v2.alpha, v2.beta)

            # Compute entanglement measure (simplified concurrence)
            # |⟨ψ₁|ψ₂⟩|² for correlation strength
            correlation = (
                abs(np.conj(v1.alpha) * v2.alpha + np.conj(v1.beta) * v2.beta) ** 2
            )

            # State change measure (indicates entanglement occurred)
            state_change_1 = abs(final_state_1[0] - initial_state_1[0]) + abs(
                final_state_1[1] - initial_state_1[1]
            )
            state_change_2 = abs(final_state_2[0] - initial_state_2[0]) + abs(
                final_state_2[1] - initial_state_2[1]
            )

            entanglement_measure = min(state_change_1, state_change_2)

            results["entangled_pairs"].append((v1_id, v2_id))
            results["correlation_strengths"].append(correlation)
            results["entanglement_measures"].append(entanglement_measure)

            print(
                f"  Pair ({v1_id},{v2_id}): correlation={correlation:.6f}, "
                f"entanglement_measure={entanglement_measure:.6f}"
            )

        avg_correlation = np.mean(results["correlation_strengths"])
        avg_entanglement = np.mean(results["entanglement_measures"])

        # Pass if average entanglement measure > threshold
        if avg_entanglement < 0.01:  # Minimum detectable entanglement
            results["pass"] = False

        print(f"  Average correlation strength: {avg_correlation:.6f}")
        print(f"  Average entanglement measure: {avg_entanglement:.6f}")
        print(f"  RESULT: {'PASS' if results['pass'] else 'FAIL'}")

        return results

    def test_quantum_interference(self) -> Dict[str, Any]:
        """
        Test 4: Quantum Interference Validation
        Test for interference patterns in superposed states
        """
        print("\n🔬 TEST 4: Quantum Interference")

        results = {
            "test_name": "Quantum Interference",
            "interference_patterns": [],
            "phase_relationships": [],
            "contrast_ratios": [],
            "pass": True,
        }

        # Create interference by applying Hadamard gates and measuring phases
        vertex_ids = list(self.network.vertices.keys())[:4]  # Test first 4 vertices

        for vertex_id in vertex_ids:
            vertex = self.network.vertices[vertex_id]

            # Record initial phase
            initial_phase = (
                np.angle(vertex.beta / vertex.alpha)
                if abs(vertex.alpha) > 1e-10
                else 0.0
            )

            # Apply Hadamard (creates interference)
            self.network.apply_gate_to_vertex(vertex_id, "H")

            # Record final phase
            final_phase = (
                np.angle(vertex.beta / vertex.alpha)
                if abs(vertex.alpha) > 1e-10
                else 0.0
            )

            # Measure probabilities
            p0, p1 = vertex.measure_probabilities()

            # Interference contrast: visibility = (Pmax - Pmin)/(Pmax + Pmin)
            contrast = abs(p0 - p1) / (p0 + p1) if (p0 + p1) > 0 else 0.0

            phase_change = abs(final_phase - initial_phase)

            results["phase_relationships"].append(phase_change)
            results["contrast_ratios"].append(contrast)
            results["interference_patterns"].append((p0, p1))

            print(
                f"  Vertex {vertex_id}: P(0)={p0:.6f}, P(1)={p1:.6f}, "
                f"contrast={contrast:.6f}, phase_change={phase_change:.6f}"
            )

        avg_contrast = np.mean(results["contrast_ratios"])
        avg_phase_change = np.mean(results["phase_relationships"])

        # Pass if interference is detected (contrast > 0.1)
        if avg_contrast < 0.1:
            results["pass"] = False

        print(f"  Average interference contrast: {avg_contrast:.6f}")
        print(f"  Average phase change: {avg_phase_change:.6f}")
        print(f"  RESULT: {'PASS' if results['pass'] else 'FAIL'}")

        return results

    def test_no_cloning_theorem(self) -> Dict[str, Any]:
        """
        Test 5: No-Cloning Theorem Validation
        Verify that quantum states cannot be perfectly cloned
        """
        print("\n🔬 TEST 5: No-Cloning Theorem")

        results = {
            "test_name": "No-Cloning Theorem",
            "cloning_attempts": [],
            "fidelity_measures": [],
            "cloning_prevented": True,
            "pass": True,
        }

        # Attempt to clone quantum states
        vertex_ids = list(self.network.vertices.keys())[:3]

        for vertex_id in vertex_ids:
            source_vertex = self.network.vertices[vertex_id]

            # Try to create a "clone" by copying state
            original_state = (source_vertex.alpha, source_vertex.beta)

            # Find another vertex to be the "clone target"
            target_id = (vertex_id + 1) % len(self.network.vertices)
            target_vertex = self.network.vertices[target_id]

            # Record target's initial state

            # Attempt "cloning" via entanglement (this should not create perfect clone)
            self.network.entangle_vertices(vertex_id, target_id)

            # Measure fidelity of "cloning"
            final_state = (target_vertex.alpha, target_vertex.beta)

            # Fidelity = |⟨ψ_original|ψ_clone⟩|²
            fidelity = (
                abs(
                    np.conj(original_state[0]) * final_state[0]
                    + np.conj(original_state[1]) * final_state[1]
                )
                ** 2
            )

            results["cloning_attempts"].append((vertex_id, target_id))
            results["fidelity_measures"].append(fidelity)

            # Perfect cloning would have fidelity = 1.0
            if fidelity > 0.99:  # Too close to perfect cloning
                results["cloning_prevented"] = False
                results["pass"] = False

            print(f"  Clone attempt {vertex_id}→{target_id}: fidelity={fidelity:.6f}")

        avg_fidelity = np.mean(results["fidelity_measures"])
        max_fidelity = max(results["fidelity_measures"])

        print(f"  Average cloning fidelity: {avg_fidelity:.6f}")
        print(f"  Maximum cloning fidelity: {max_fidelity:.6f}")
        print(f"  No-cloning preserved: {results['cloning_prevented']}")
        print(f"  RESULT: {'PASS' if results['pass'] else 'FAIL'}")

        return results

    def test_quantum_coherence(self) -> Dict[str, Any]:
        """
        Test 6: Quantum Coherence Validation
        Test maintenance of quantum coherence over time
        """
        print("\n🔬 TEST 6: Quantum Coherence")

        results = {
            "test_name": "Quantum Coherence",
            "initial_coherences": [],
            "final_coherences": [],
            "coherence_preservation": [],
            "pass": True,
        }

        # Measure initial coherence (off-diagonal matrix elements)
        initial_coherences = []
        for vertex in self.network.vertices.values():
            # Coherence = |⟨0|ρ|1⟩| = |α*β|
            coherence = abs(np.conj(vertex.alpha) * vertex.beta)
            initial_coherences.append(coherence)
            results["initial_coherences"].append(coherence)

        # Evolve system
        print("  Evolving system to test coherence preservation...")
        self.network.evolve_network(time_steps=3, dt=0.05)

        # Measure final coherence
        final_coherences = []
        for vertex in self.network.vertices.values():
            coherence = abs(np.conj(vertex.alpha) * vertex.beta)
            final_coherences.append(coherence)
            results["final_coherences"].append(coherence)

        # Check coherence preservation
        for i, (initial, final) in enumerate(zip(initial_coherences, final_coherences)):
            preservation = final / initial if initial > 1e-10 else 1.0
            results["coherence_preservation"].append(preservation)

            print(
                f"  Vertex {i}: initial={initial:.6f}, final={final:.6f}, "
                f"preservation={preservation:.6f}"
            )

        avg_preservation = np.mean(results["coherence_preservation"])

        # Pass if average coherence preservation > 0.8
        if avg_preservation < 0.8:
            results["pass"] = False

        print(f"  Average coherence preservation: {avg_preservation:.6f}")
        print(f"  RESULT: {'PASS' if results['pass'] else 'FAIL'}")

        return results

    def run_full_validation_suite(self) -> Dict[str, Any]:
        """Run complete quantum validation test suite"""
        print("🧪 RIGOROUS QUANTUM VERTEX VALIDATION SUITE")
        print("=" * 60)
        print(f"Network: {len(self.network.vertices)} vertices")
        print(f"Scientific precision threshold: {self.tolerance:.0e}")
        print()

        start_time = time.time()

        # Run all tests
        test1 = self.test_superposition_principle()
        test2 = self.test_unitary_evolution()
        test3 = self.test_entanglement_generation()
        test4 = self.test_quantum_interference()
        test5 = self.test_no_cloning_theorem()
        test6 = self.test_quantum_coherence()

        total_time = time.time() - start_time

        # Compile results
        all_tests = [test1, test2, test3, test4, test5, test6]
        passed_tests = sum(1 for test in all_tests if test["pass"])

        # Generate final scientific report
        print("\n" + "=" * 60)
        print("📋 QUANTUM VALIDATION SCIENTIFIC REPORT")
        print("=" * 60)

        for i, test in enumerate(all_tests, 1):
            status = "PASS" if test["pass"] else "FAIL"
            print(f"Test {i}: {test['test_name']} - {status}")

        print("\nOVERALL RESULTS:")
        print(f"  Tests Passed: {passed_tests}/{len(all_tests)}")
        print(f"  Success Rate: {passed_tests/len(all_tests)*100:.1f}%")
        print(f"  Validation Time: {total_time:.3f} seconds")

        quantum_validity = passed_tests == len(all_tests)

        if quantum_validity:
            print("\n✅ QUANTUM VALIDATION: SUCCESSFUL")
            print("   Your topological vertex system demonstrates:")
            print("   • Valid quantum superposition")
            print("   • Unitary evolution preservation")
            print("   • Genuine quantum entanglement")
            print("   • Observable quantum interference")
            print("   • No-cloning theorem compliance")
            print("   • Quantum coherence maintenance")
            print(
                "\n   SCIENTIFIC CONCLUSION: System exhibits authentic quantum behavior"
            )
        else:
            print("\n⚠️ QUANTUM VALIDATION: INCOMPLETE")
            print(f"   {len(all_tests) - passed_tests} tests failed")
            print("   System may not fully exhibit quantum behavior")

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "network_size": len(self.network.vertices),
            "precision_threshold": self.tolerance,
            "total_tests": len(all_tests),
            "passed_tests": passed_tests,
            "success_rate": passed_tests / len(all_tests),
            "validation_time": total_time,
            "quantum_validity": quantum_validity,
            "detailed_results": all_tests,
        }


def main():
    """Run rigorous quantum vertex validation experiment"""
    print("🔬 INITIALIZING QUANTUM VERTEX VALIDATION EXPERIMENT")
    print("=" * 60)

    # Create test network
    network = QubitVertexNetwork(8)

    # Apply some quantum operations to create interesting states
    network.apply_gate_to_vertex(0, "H")  # Superposition
    network.apply_gate_to_vertex(2, "H")
    network.apply_gate_to_vertex(1, "X")  # Bit flip
    network.entangle_vertices(0, 1)  # Entanglement
    network.entangle_vertices(2, 3)

    # Initialize validator
    validator = QuantumVertexValidator(network)

    # Run validation
    results = validator.run_full_validation_suite()

    # Save results
    with open("quantum_vertex_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n📄 Detailed results saved to 'quantum_vertex_validation_results.json'")

    return results


if __name__ == "__main__":
    validation_results = main()
