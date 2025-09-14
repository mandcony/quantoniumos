#!/usr/bin/env python3
from tests.tests.rft_scientific_validation import create_unitary_rft_engine, create_random_vector, max_abs_error, mean_abs_error

print("Testing unitarity with single engine...")

size = 64
rft = create_unitary_rft_engine(size)
max_errors = []
mean_errors = []

for i in range(5):  # Test with fewer repetitions
    x = create_random_vector(size, complex_valued=True, seed=i)
    X = rft.forward(x)
    x_recovered = rft.inverse(X)
    max_err = max_abs_error(x, x_recovered)
    mean_err = mean_abs_error(x, x_recovered)
    max_errors.append(max_err)
    mean_errors.append(mean_err)
    print(f'Test {i+1}: max={max_err:.2e}, mean={mean_err:.2e}')

max_error = max(max_errors)
mean_error = sum(mean_errors) / len(mean_errors)
print(f'Overall: max={max_error:.2e}, mean={mean_error:.2e}')
print(f'Passes thresholds: max<{5e-12}: {max_error < 5e-12}, mean<{1e-12}: {mean_error < 1e-12}')