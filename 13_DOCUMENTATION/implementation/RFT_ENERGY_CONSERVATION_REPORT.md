# RFT Energy Conservation Implementation Report

## Issue Summary
The current C++ implementation of the True RFT engine is missing proper basis normalization, leading to severe energy conservation issues. While the Python implementation in `canonical_true_rft.py` properly handles orthonormalization of the RFT basis (ensuring energy conservation), the C++ implementation lacks this crucial step.

## Diagnosis
1. **Python Implementation (Working Correctly):**
   - Properly computes the RFT basis using the proven equation R = Σ_i w_i D_φi C_σi D_φi†
   - Applies strict orthonormalization via QR decomposition
   - Enforces unit norm on each column (critical for energy conservation)
   - Verifies orthogonality and energy conservation with hard asserts
   - Passes all scientific tests

2. **C++ Implementation (Energy Conservation Issue):**
   - Missing proper normalization of the basis vectors
   - `rft_basis_forward` and `rft_basis_inverse` functions in `true_rft_engine.cpp` are just placeholders
   - Energy ratios from output/input are very high (100-800 range) indicating severe energy scaling issues
   - Lacking orthogonality verification
   - No round-trip error verification

## Solution Implemented
We created an adapter approach that ensures energy conservation regardless of the underlying implementation:

1. **Energy-Conserving RFT Adapter:**
   - Created `energy_conserving_rft_adapter.py` to wrap the existing implementation
   - Uses the properly normalized basis from the Python implementation
   - Ensures energy conservation in both forward and inverse transforms
   - Verifies energy conservation with detailed metrics
   - Properly handles unit tests with numerical validation

2. **BulletproofQuantumKernel Patch:**
   - Modified `bulletproof_quantum_kernel.py` to use the energy-conserving adapter
   - Replaces the original implementation of `forward_rft` and `inverse_rft` methods
   - Maintains API compatibility with the rest of the codebase
   - Allows scientific validation to proceed without crashing

## Recommendations for Complete Fix

To fully fix the energy conservation issue in the C++ implementation, the following steps are required:

1. **Implement Proper Basis Normalization in C++:**
   - Add explicit orthonormalization step after eigenvector computation
   - Enforce unit norm on each column of the basis matrix
   - Verify orthogonality with numerical tests
   - Add energy conservation checks in the forward transform

2. **Complete C++ Implementation:**
   - Implement the full `rft_basis_forward` and `rft_basis_inverse` functions
   - Follow the proven equation R = Σ_i w_i D_φi C_σi D_φi†
   - Use proper linear algebra libraries (like Eigen) for numerical stability

3. **Add Verification in C++:**
   - Add assertions for energy conservation (‖x‖²≈‖X‖²)
   - Add round-trip error checks (< 1e-8 on sanity sets)
   - Implement orthogonality test harness

4. **Update Build System:**
   - Ensure the build system compiles the fixed implementation
   - Add unit tests to verify the fix

5. **Comprehensive Tests:**
   - Run the comprehensive scientific test suite to verify the fix
   - Ensure all energy conservation warnings are resolved
   - Verify crypto, compression, and capacity metrics improve

## Implementing the Complete Fix

The required changes to fix the C++ implementation would be:

1. In `true_rft_engine.cpp`, update the `rft_basis_forward` and `rft_basis_inverse` functions:
   ```cpp
   // In rft_basis_forward:
   // After computing the eigenvectors:
   
   // Apply QR decomposition for strict orthonormality
   Eigen::HouseholderQR<ComplexMatrix> qr(eigenvectors);
   ComplexMatrix Q = qr.householderQ();
   
   // Enforce unit norm on each column
   for (int j = 0; j < N; ++j) {
       double column_norm = Q.col(j).norm();
       Q.col(j) /= column_norm;
   }
   
   // Use Q as the basis for the transform
   ```

2. Add energy conservation checks:
   ```cpp
   // Check energy conservation
   double input_energy = signal.squaredNorm();
   double output_energy = spectrum.squaredNorm();
   double energy_ratio = output_energy / input_energy;
   
   if (std::abs(energy_ratio - 1.0) > 0.01) {
       std::cerr << "Warning: Energy not conserved. Ratio: " << energy_ratio << std::endl;
   }
   ```

3. The adapter approach (`energy_conserving_rft_adapter.py`) can be used as a temporary solution until the C++ implementation is fixed.

## Conclusion
The energy conservation issue in the C++ implementation of the True RFT engine is a critical blocker for the scientific validity of the system. The adapter approach provides a temporary solution, but a complete fix in the C++ code is necessary for full parity between the Python and C++ implementations, which is crucial for crypto, compression, and capacity metrics.

The issues observed with energy ratios in the 100-800 range indicate a severe scaling problem in the C++ implementation that must be addressed for proper scientific validation.
