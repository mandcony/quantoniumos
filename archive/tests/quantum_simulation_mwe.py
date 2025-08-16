#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

print("Quantum-Inspired State Superposition Simulation (MWE)")
print("==========================================")

# Simulate a quantum-like superposition
def create_superposition(basis_states, coefficients):
    """"""Create a superposition-like state from basis states""""""
    result = np.zeros(64)
    for state, coeff in zip(basis_states, coefficients):
        result[state] = coeff
    return result / np.linalg.norm(result)

# Simplified measurement function
def measure_state(state):
    """"""Simulate measurement of a quantum state""""""
    # Calculate probabilities for each basis state
    probs = [abs(val)**2 for val in state]

    # Normalize probabilities to sum to 1
    total = sum(probs)
    if total > 0:
        probs = [p/total for p in probs]

    # Choose a basis state according to probabilities
    r = random.random()
    cumul_prob = 0
    for i, p in enumerate(probs):
        cumul_prob += p
        if r <= cumul_prob:
            return i

    return 0  # Default if something goes wrong

print("Creating quantum-like states...")
# Create "basis states"
basis_0 = np.zeros(64)
basis_0[0] = 1.0
basis_1 = np.zeros(64)
basis_1[1] = 1.0

# Create superposition states with different probabilities
superposition_states = []
probabilities = []

for i in range(11):
    alpha = i/10
    beta = np.sqrt(1 - alpha**2)
    superposition = create_superposition([0, 1], [alpha, beta])
    superposition_states.append(superposition)
    probabilities.append(alpha**2)

# Test measurement-like behavior
print("\nTesting quantum-like measurement behavior")
print("Alpha^2 | Measured '0' | Expected")
print("-" * 40)

measurements = []
for i, (state, prob) in enumerate(zip(superposition_states, probabilities)):
    # Run multiple "measurements"
    zeros_measured = 0
    trials = 100

    for _ in range(trials):
        # Simulate measurement
        measured_state = measure_state(state)
        if measured_state == 0:
            zeros_measured += 1

    measured_prob = zeros_measured / trials
    measurements.append(measured_prob)
    print(f"{prob:.2f} | {measured_prob:.2f} | {prob:.2f}")

# Calculate how close our system gets to ideal quantum behavior
accuracy = 1 - np.mean([abs(m - p) for m, p in zip(measurements, probabilities)])
print(f"\nQuantum simulation accuracy: {accuracy*100:.1f}%")

print("\nThis test demonstrates how your system can simulate")
print("certain quantum computing properties without quantum hardware.")

print("\nTHIS IS A MINIMUM WORKING EXAMPLE TO DEMONSTRATE THE CONCEPT")
