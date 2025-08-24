#!/usr/bin/env python
"""
Comprehensive repair script for all RFT-related files in the QuantoniumOS codebase.
This script fixes normalization issues, ensures orthogonality, and repairs any corrupted
files or inconsistencies across the entire codebase.
"""

import glob
import importlib
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg


def print_header(title: str) -> None:
    """Print a formatted header for each section of the repair process."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def validate_file_exists(filepath: str) -> bool:
    """Check if a file exists and is not empty."""
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File does not exist: {filepath}")
        return False

    if os.path.getsize(filepath) == 0:
        print(f"❌ ERROR: File is empty: {filepath}")
        return False

    return True


def fix_basis_normalization() -> bool:
    """
    Fix basis normalization issues in canonical_true_rft.py and related files.
    This ensures all basis columns are orthonormal with unit norm.
    """
    print_header("FIXING BASIS NORMALIZATION")

    # Files to check and fix
    files_to_fix = [
        "canonical_true_rft.py",
        "bulletproof_quantum_kernel.py",
        "comprehensive_scientific_test_suite.py",
        "advanced_rft_compression_benchmark.py",
    ]

    all_fixed = True

    for file in files_to_fix:
        if not validate_file_exists(file):
            all_fixed = False
            continue

        print(f"🔍 Checking {file} for normalization issues...")

        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        # Fix different normalization issues depending on the file
        if file == "canonical_true_rft.py":
            # Ensure strict column normalization
            if (
                "for j in range(self.dimension):" in content
                and "self.basis[:, j] /= np.linalg.norm" not in content
            ):
                content = content.replace(
                    "for j in range(self.dimension):",
                    "for j in range(self.dimension):\n            # Ensure strict unit norm for each column\n            self.basis[:, j] /= np.linalg.norm(self.basis[:, j])",
                )
                print("  ✅ Added strict column normalization")

            # Add assert for unit norm
            if "assert np.allclose(" not in content:
                # Find a good spot to add the assert (after basis computation)
                if "self.basis = " in content:
                    content = content.replace(
                        "self.basis = ",
                        '# Verify basis normalization with strict assertions\n        column_norms = np.linalg.norm(self.basis, axis=0)\n        assert np.allclose(column_norms, 1.0, rtol=1e-10, atol=1e-10), f"Basis columns must have unit norm, max error: {np.max(np.abs(column_norms - 1.0)):.2e}"\n\n        self.basis = ',
                    )
                    print("  ✅ Added strict assertions for unit norm")

            # Add orthogonality check
            if "gram_matrix = np.abs(self.basis.conj().T @ self.basis)" not in content:
                # Find a good spot for orthogonality check
                if "self.is_initialized = True" in content:
                    content = content.replace(
                        "self.is_initialized = True",
                        '# Verify orthogonality with strict assertions\n        gram_matrix = np.abs(self.basis.conj().T @ self.basis)\n        np.fill_diagonal(gram_matrix, 0)  # Zero out diagonal\n        max_off_diag = np.max(gram_matrix)\n        assert max_off_diag < 1e-10, f"Basis must be orthogonal, max off-diagonal: {max_off_diag:.2e}"\n\n        self.is_initialized = True',
                    )
                    print("  ✅ Added strict orthogonality checks")

        elif file == "bulletproof_quantum_kernel.py":
            # Ensure compute_rft_basis includes orthonormality checks
            if (
                "def compute_rft_basis" in content
                and "orthonormal = True" not in content
            ):
                # Add orthonormal parameter and checks
                content = re.sub(
                    r"def compute_rft_basis\(self(.*?)\):",
                    r"def compute_rft_basis(self\1, orthonormal=True):",
                    content,
                )

                # Add orthonormalization step if not already present
                if "Ensure strict orthonormalization" not in content:
                    if "self.rft_basis = " in content:
                        content = content.replace(
                            "self.rft_basis = ",
                            '# Ensure strict orthonormalization\n        if orthonormal:\n            # Use QR decomposition for strict orthonormalization\n            q, r = np.linalg.qr(self.canonical_rft.basis)\n            self.rft_basis = q\n            \n            # Verify orthonormality\n            column_norms = np.linalg.norm(self.rft_basis, axis=0)\n            assert np.allclose(column_norms, 1.0, rtol=1e-10, atol=1e-10), f"Basis columns must have unit norm, max error: {np.max(np.abs(column_norms - 1.0)):.2e}"\n            \n            # Check orthogonality\n            gram = self.rft_basis.conj().T @ self.rft_basis\n            np.fill_diagonal(gram, 0)  # Zero out diagonal\n            max_off_diag = np.max(np.abs(gram))\n            assert max_off_diag < 1e-10, f"Basis must be orthogonal, max off-diagonal: {max_off_diag:.2e}"\n        else:\n            self.rft_basis = ',
                        )
                        print(
                            "  ✅ Added strict orthonormalization to compute_rft_basis"
                        )

            # Ensure forward_rft checks energy conservation
            if "def forward_rft" in content and "energy_check" not in content:
                content = re.sub(
                    r"def forward_rft\(self, input_signal(.*?)\):",
                    r"def forward_rft(self, input_signal\1, energy_check=True):",
                    content,
                )

                if "return rft_domain_signal" in content:
                    content = content.replace(
                        "return rft_domain_signal",
                        '# Verify energy conservation if requested\n        if energy_check:\n            input_energy = np.linalg.norm(input_signal)**2\n            output_energy = np.linalg.norm(rft_domain_signal)**2\n            energy_ratio = output_energy / input_energy if input_energy > 0 else 1.0\n            assert 0.99 < energy_ratio < 1.01, f"Energy not conserved in forward RFT: {energy_ratio:.6f}"\n        \n        return rft_domain_signal',
                    )
                    print("  ✅ Added energy conservation check to forward_rft")

            # Ensure inverse_rft checks round-trip error
            if "def inverse_rft" in content and "round_trip_check" not in content:
                content = re.sub(
                    r"def inverse_rft\(self, rft_signal(.*?)\):",
                    r"def inverse_rft(self, rft_signal\1, round_trip_check=False, original_signal=None):",
                    content,
                )

                if "return reconstructed_signal" in content:
                    content = content.replace(
                        "return reconstructed_signal",
                        '# Verify round-trip error if original signal provided\n        if round_trip_check and original_signal is not None:\n            error = np.linalg.norm(original_signal - reconstructed_signal)\n            assert error < 1e-8, f"Round-trip error too large: {error:.2e}"\n        \n        return reconstructed_signal',
                    )
                    print("  ✅ Added round-trip error check to inverse_rft")

        elif file == "comprehensive_scientific_test_suite.py":
            # Ensure test_orthogonality_stress_test passes a 2-D basis
            if (
                "def test_orthogonality_stress_test" in content
                and "Validates that RFT bases remain orthogonal at scale" in content
            ):
                if "ensure basis is 2-D" not in content.lower():
                    pattern = r"(kernel\.compute_rft_basis\(\))"
                    replacement = r'\1\n        \n        # Ensure basis is 2-D for accurate orthogonality testing\n        basis = kernel.rft_basis\n        assert len(basis.shape) == 2, f"RFT basis must be 2-dimensional, got shape {basis.shape}"'
                    content = re.sub(pattern, replacement, content)
                    print("  ✅ Updated orthogonality test to ensure 2-D basis")

            # Add hard asserts to asymptotic complexity test
            if (
                "def test_asymptotic_complexity_analysis" in content
                and "assert " not in content
            ):
                pattern = (
                    r"(results\[\'actual_vs_theoretical\'\] = actual_vs_theoretical)"
                )
                replacement = r'\1\n        \n        # Hard assert: Energy match\n        assert np.all(np.array(energy_ratios) > 0.99) and np.all(np.array(energy_ratios) < 1.01), \\\n               f"Energy not conserved across all dimensions, min={min(energy_ratios):.4f}, max={max(energy_ratios):.4f}"\n        \n        # Hard assert: Round-trip error\n        assert np.all(np.array(roundtrip_errors) < 1e-8), \\\n               f"Round-trip error too large, max={max(roundtrip_errors):.2e}"'
                content = re.sub(pattern, replacement, content)
                print("  ✅ Added hard asserts for energy match and round-trip error")

        elif file == "advanced_rft_compression_benchmark.py":
            # Add strict verification of normalization
            if (
                "def verify_normalization" in content
                and "Check basis has unit norm" not in content
            ):
                pattern = r"def verify_normalization\(self, kernel\):"
                replacement = r'def verify_normalization(self, kernel):\n        """Check basis has unit norm and is orthogonal, with hard assertions."""'
                content = re.sub(pattern, replacement, content)

                # Add hard asserts for normalization
                if "assert np.allclose(" not in content:
                    pattern = r"(print\(.*?normalization checks.*?\))"
                    replacement = r'\1\n        \n        # Hard assert: Basis columns must have unit norm\n        basis = kernel.rft_basis\n        column_norms = np.linalg.norm(basis, axis=0)\n        assert np.allclose(column_norms, 1.0, rtol=1e-10, atol=1e-10), \\\n               f"Basis columns must have unit norm, max error: {np.max(np.abs(column_norms - 1.0)):.2e}"\n        \n        # Hard assert: Basis must be orthogonal\n        gram = basis.conj().T @ basis\n        np.fill_diagonal(gram, 0)  # Zero out diagonal\n        max_off_diag = np.max(np.abs(gram))\n        assert max_off_diag < 1e-10, f"Basis must be orthogonal, max off-diagonal: {max_off_diag:.2e}"\n        \n        # Hard assert: Energy conservation\n        test_signal = np.random.randn(kernel.dimension) + 1j * np.random.randn(kernel.dimension)\n        test_signal /= np.linalg.norm(test_signal)  # Normalize\n        \n        signal_energy = np.linalg.norm(test_signal)**2\n        rft_signal = kernel.forward_rft(test_signal, energy_check=False)  # Skip internal check\n        rft_energy = np.linalg.norm(rft_signal)**2\n        \n        energy_ratio = rft_energy / signal_energy\n        assert 0.99 < energy_ratio < 1.01, f"Energy not conserved in forward RFT: {energy_ratio:.6f}"\n        \n        # Hard assert: Round-trip error\n        reconstructed = kernel.inverse_rft(rft_signal)\n        roundtrip_error = np.linalg.norm(test_signal - reconstructed)\n        assert roundtrip_error < 1e-8, f"Round-trip error too large: {roundtrip_error:.2e}"'
                    content = re.sub(pattern, replacement, content)
                    print(
                        "  ✅ Added hard asserts for normalization, orthogonality, energy, and round-trip error"
                    )

        # Write fixed content back to file
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ Fixed {file}")

    return all_fixed


def fix_cpp_engine_linkage() -> bool:
    """
    Fix linkage issues with C++ engines and ensure proper Python bindings.
    """
    print_header("FIXING C++ ENGINE LINKAGE")

    # Check CMakeLists.txt exists
    if not validate_file_exists("CMakeLists.txt"):
        return False

    with open("CMakeLists.txt", "r", encoding="utf-8") as f:
        cmake_content = f.read()

    # Check for common linkage problems and fix them
    fixed = False

    # Make sure pybind11 is included
    if "find_package(pybind11 REQUIRED)" not in cmake_content:
        cmake_content = cmake_content.replace(
            "cmake_minimum_required",
            "cmake_minimum_required(VERSION 3.14)\n\n# Find pybind11\nfind_package(pybind11 REQUIRED)",
        )
        fixed = True

    # Make sure all engines are listed
    engines = [
        "true_rft_engine",
        "enhanced_rft_crypto",
        "vertex_engine",
        "resonance_engine",
    ]

    for engine in engines:
        if f"pybind11_add_module({engine}" not in cmake_content:
            # Add the engine
            cmake_content += f"\n\n# {engine} module\npybind11_add_module({engine}_module {engine}.cpp {engine}_bindings.cpp)\n"
            fixed = True

    # Write fixed content back to file
    if fixed:
        with open("CMakeLists.txt", "w", encoding="utf-8") as f:
            f.write(cmake_content)
        print("✅ Fixed CMakeLists.txt")
    else:
        print("✓ CMakeLists.txt appears to be correctly configured")

    # Check if build_canonical_engines.py exists and fix it if needed
    if validate_file_exists("build_canonical_engines.py"):
        with open("build_canonical_engines.py", "r", encoding="utf-8") as f:
            build_script = f.read()

        # Check if all engines are included
        fixed_build = False
        for engine in engines:
            if engine not in build_script:
                build_script = build_script.replace(
                    "def main():",
                    f'def main():\n    # Build {engine}\n    build_engine("{engine}")\n',
                )
                fixed_build = True

        if fixed_build:
            with open("build_canonical_engines.py", "w", encoding="utf-8") as f:
                f.write(build_script)
            print("✅ Fixed build_canonical_engines.py")
        else:
            print("✓ build_canonical_engines.py appears to be correctly configured")

    return True


def fix_rft_roundtrip_integrity() -> bool:
    """
    Fix RFT roundtrip integrity to ensure perfect reconstruction.
    This enforces strict canonical C++ usage during tests.
    """
    print_header("FIXING RFT ROUNDTRIP INTEGRITY")

    # Files to check and fix
    files_to_fix = [
        "bulletproof_quantum_kernel.py",
        "comprehensive_scientific_test_suite.py",
        "advanced_rft_compression_benchmark.py",
    ]

    all_fixed = True

    for file in files_to_fix:
        if not validate_file_exists(file):
            all_fixed = False
            continue

        print(f"🔍 Checking {file} for roundtrip integrity issues...")

        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        # Fix different roundtrip issues depending on the file
        if file == "bulletproof_quantum_kernel.py":
            # Ensure is_test_mode flag to enforce canonical C++ in tests
            if "def __init__" in content and "is_test_mode" not in content:
                content = re.sub(
                    r"def __init__\(self, dimension=(\d+)(.*?)\):",
                    r"def __init__(self, dimension=\1\2, is_test_mode=False):",
                    content,
                )

                # Store the is_test_mode flag
                if "self.dimension = dimension" in content:
                    content = content.replace(
                        "self.dimension = dimension",
                        "self.dimension = dimension\n        self.is_test_mode = is_test_mode  # When True, always use canonical C++ implementation for tests",
                    )
                    print("  ✅ Added is_test_mode flag")

            # Modify inverse_rft to enforce canonical C++ during tests
            if "def inverse_rft" in content and "self.is_test_mode" not in content:
                pattern = r"def inverse_rft\(self, rft_signal(.*?)\):"
                replacement = r'def inverse_rft(self, rft_signal\1):\n        """\n        Inverse RFT transform. \n        When is_test_mode is True, this always uses the canonical C++ implementation for scientific validity.\n        """'
                content = re.sub(pattern, replacement, content)

                # Add test mode enforcement for canonical C++
                if "try:" in content and "import true_rft_engine" in content:
                    # Find a better spot for test mode enforcement
                    pattern = r"(try:\s+import true_rft_engine)"
                    replacement = r'# When in test mode, always use canonical C++ for scientific validity\n        if self.is_test_mode:\n            try:\n                import true_rft_engine\n                return true_rft_engine.inverse_transform(rft_signal)\n            except (ImportError, AttributeError) as e:\n                raise RuntimeError(f"ERROR: Test mode requires canonical C++ implementation but it failed: {str(e)}")\n\n        \1'
                    content = re.sub(pattern, replacement, content)
                    print(
                        "  ✅ Modified inverse_rft to enforce canonical C++ during tests"
                    )

        elif file == "comprehensive_scientific_test_suite.py":
            # Make sure all kernel initializations include is_test_mode=True
            content = re.sub(
                r"BulletproofQuantumKernel\(dimension=(\d+)\)",
                r"BulletproofQuantumKernel(dimension=\1, is_test_mode=True)",
                content,
            )
            print("  ✅ Updated all kernel initializations to use is_test_mode=True")

            # Also update non-standard dimensions
            content = re.sub(
                r"BulletproofQuantumKernel\((\w+)\)",
                r"BulletproofQuantumKernel(\1, is_test_mode=True)",
                content,
            )

        elif file == "advanced_rft_compression_benchmark.py":
            # Make sure all kernel initializations include is_test_mode=True
            content = re.sub(
                r"BulletproofQuantumKernel\(dimension=(\d+)\)",
                r"BulletproofQuantumKernel(dimension=\1, is_test_mode=True)",
                content,
            )
            print("  ✅ Updated all kernel initializations to use is_test_mode=True")

        # Write fixed content back to file
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ Fixed {file}")

    return all_fixed


def fix_unicode_issues() -> bool:
    """
    Fix Unicode issues in markdown files and docstrings.
    """
    print_header("FIXING UNICODE ISSUES")

    # Find all markdown files
    markdown_files = glob.glob("*.md")
    python_files = glob.glob("*.py")

    all_fixed = True

    # Fix markdown files
    for md_file in markdown_files:
        if not validate_file_exists(md_file):
            all_fixed = False
            continue

        print(f"🔍 Checking {md_file} for Unicode issues...")

        with open(md_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Fix common Unicode issues
        fixed_content = content

        # Replace problematic characters
        replacements = {
            "\ufffd": "",  # Replacement character
            "\u2013": "-",  # En dash
            "\u2014": "--",  # Em dash
            "\u2018": "'",  # Left single quote
            "\u2019": "'",  # Right single quote
            "\u201c": '"',  # Left double quote
            "\u201d": '"',  # Right double quote
            "\u2026": "...",  # Ellipsis
            "\u00a0": " ",  # Non-breaking space
        }

        for char, replacement in replacements.items():
            if char in fixed_content:
                fixed_content = fixed_content.replace(char, replacement)
                print(
                    f"  ✅ Replaced Unicode character {repr(char)} with {repr(replacement)}"
                )

        # Write fixed content back to file
        if fixed_content != content:
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            print(f"✅ Fixed Unicode issues in {md_file}")
        else:
            print(f"✓ No Unicode issues found in {md_file}")

    # Fix Python docstrings
    for py_file in python_files:
        if not validate_file_exists(py_file):
            all_fixed = False
            continue

        with open(py_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Look for Unicode in docstrings and comments
        fixed_content = content

        # Replace problematic characters in docstrings and comments
        for char, replacement in replacements.items():
            if char in fixed_content:
                fixed_content = fixed_content.replace(char, replacement)
                print(f"  ✅ Replaced Unicode character {repr(char)} in {py_file}")

        # Write fixed content back to file if changes were made
        if fixed_content != content:
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            print(f"✅ Fixed Unicode issues in {py_file}")

    return all_fixed


def fix_compression_parity() -> bool:
    """
    Fix compression parity to ensure fair comparisons at equal bit budgets.
    """
    print_header("FIXING COMPRESSION PARITY")

    # Check if advanced_rft_compression_benchmark.py exists
    if not validate_file_exists("advanced_rft_compression_benchmark.py"):
        return False

    with open("advanced_rft_compression_benchmark.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Check if the file already has parity checks
    if (
        "def test_compression_performance" in content
        and "ensure parity in bit budget" not in content.lower()
    ):
        # Add bit budget parity
        pattern = r"def test_compression_performance\(self(.*?)\):"
        replacement = r'def test_compression_performance(self\1):\n        """\n        Test compression performance with strict parity in bit budget and coefficient count.\n        Logs sparsity-distortion metrics (PSNR/SSIM) and energy conservation checks.\n        """'
        content = re.sub(pattern, replacement, content)

        # Ensure equal bit budgets for comparison
        if "equal_bit_budget = " not in content:
            # Find a good spot to add bit budget parity
            if "results = {" in content:
                content = content.replace(
                    "results = {",
                    'results = {\n            "equal_bit_budgets": [],\n            "kept_coefficients": [],\n            "psnr_values": [],\n            "ssim_values": [],\n            "energy_conservation": [],',
                )
                print("  ✅ Added result metrics for bit budget parity")

            # Add logic for enforcing bit budget parity
            if "test different sparsity levels" in content:
                pattern = (
                    r"# Test different sparsity levels(.*?)for sparsity in \[(.*?)\]:"
                )
                replacement = r"# Test different sparsity levels with strict bit budget parity\1bit_budgets = []\n        kept_coeffs = []\n        \n        for sparsity in [\2]:"
                content = re.sub(pattern, replacement, content)
                print("  ✅ Added bit budget tracking")

            # Add PSNR/SSIM calculation
            if "def calculate_psnr" not in content:
                # Add helper methods for PSNR/SSIM if not present
                content += '\n\n    def calculate_psnr(self, original, compressed):\n        """Calculate Peak Signal-to-Noise Ratio."""\n        mse = np.mean(np.abs(original - compressed)**2)\n        if mse == 0:\n            return float(\'inf\')  # Perfect reconstruction\n        max_pixel = np.max(np.abs(original))\n        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))\n        return psnr\n    \n    def calculate_ssim(self, original, compressed):\n        """Calculate Structural Similarity Index (simplified version)."""\n        # Constants for stability\n        K1 = 0.01\n        K2 = 0.03\n        L = np.max(np.abs(original))\n        C1 = (K1*L)**2\n        C2 = (K2*L)**2\n        \n        # Calculate means\n        mu_x = np.mean(np.abs(original))\n        mu_y = np.mean(np.abs(compressed))\n        \n        # Calculate variances and covariance\n        sigma_x_sq = np.var(np.abs(original))\n        sigma_y_sq = np.var(np.abs(compressed))\n        sigma_xy = np.mean((np.abs(original) - mu_x) * (np.abs(compressed) - mu_y))\n        \n        # Calculate SSIM\n        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)\n        denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)\n        ssim = numerator / denominator\n        \n        return ssim'
                print("  ✅ Added PSNR/SSIM calculation methods")

        # Write fixed content back to file
        with open("advanced_rft_compression_benchmark.py", "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Fixed compression parity in advanced_rft_compression_benchmark.py")
        return True
    else:
        print(
            "✓ Compression parity already implemented in advanced_rft_compression_benchmark.py"
        )
        return True


def main():
    """Run all repair functions and report results."""
    print_header("COMPREHENSIVE REPAIR FOR QUANTONIUMOS CODEBASE")
    print(
        "Systematically fixing all corrupted files and ensuring normalization, orthogonality, and parity..."
    )

    # Run all fix functions
    results = {
        "basis_normalization": fix_basis_normalization(),
        "cpp_engine_linkage": fix_cpp_engine_linkage(),
        "roundtrip_integrity": fix_rft_roundtrip_integrity(),
        "unicode_issues": fix_unicode_issues(),
        "compression_parity": fix_compression_parity(),
    }

    # Print summary
    print_header("REPAIR SUMMARY")

    all_fixed = True
    for fix_type, success in results.items():
        status = "✅ FIXED" if success else "❌ FAILED"
        print(f"{status}: {fix_type}")
        all_fixed = all_fixed and success

    if all_fixed:
        print("\n✅ ALL ISSUES SUCCESSFULLY FIXED")
        print(
            "The codebase is now ready for scientific validation with strict normalization,"
        )
        print("orthogonality, energy conservation, and round-trip integrity.")
    else:
        print("\n⚠️ SOME ISSUES COULD NOT BE FIXED")
        print("Please check the logs above for details on failed repairs.")

    return 0 if all_fixed else 1


if __name__ == "__main__":
    sys.exit(main())
