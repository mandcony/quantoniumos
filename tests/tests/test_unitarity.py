#!/usr/bin/env python3
from tests.tests.rft_scientific_validation import create_unitary_rft_engine, create_random_vector, max_abs_error
import numpy as np

print("Testing unitarity with native RFT...")

errors = []
for i in range(10):
    rft = create_unitary_rft_engine(64)
    x = create_random_vector(64, seed=i)
    X = rft.forward(x)
    x_recovered = rft.inverse(X)
    error = max_abs_error(x, x_recovered)
    errors.append(error)
    print(f'Test {i+1}: {error:.2e}')

print(f'Max error: {max(errors):.2e}')
print(f'Mean error: {np.mean(errors):.2e}')
print(f'All < 1e-12: {all(e < 1e-12 for e in errors)}')
print(f'All < 1.5e-12: {all(e < 1.5e-12 for e in errors)}')