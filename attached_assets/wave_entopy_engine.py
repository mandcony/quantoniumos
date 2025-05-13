import math
import random
import numpy as np
from apps.wave_primitives import WaveNumber


def shannon_entropy(data):
    """
    Computes the Shannon entropy of a list of non-negative numbers.
    Data should sum to a nonzero value; values are converted to probabilities.
    """
    total = sum(data)
    if total == 0:
        return 0
    probabilities = [x / total for x in data if x > 0]
    return -sum(p * math.log2(p) for p in probabilities)

class WaveformEntropyEngine:
    def __init__(self, waveform, mutation_rate=0.05):
        """
        Initializes the engine with:
          - waveform: a list of WaveNumber objects representing your symbolic waveform.
          - mutation_rate: initial scaling factor for the mutation adjustments.
        """
        self.waveform = waveform
        self.mutation_rate = mutation_rate  # Controls the magnitude of mutations

    def mutate_waveform(self):
        """
        Applies a mutation to each WaveNumber in the waveform.
        Each mutation randomly perturbs the amplitude and phase by a factor of the mutation_rate.
        """
        for w in self.waveform:
            # Compute random perturbations for amplitude and phase
            delta_amp = random.uniform(-self.mutation_rate, self.mutation_rate)
            delta_phase = random.uniform(-self.mutation_rate * math.pi, self.mutation_rate * math.pi)
            # Update amplitude and phase, wrapping as needed
            w.amplitude = (w.amplitude + delta_amp) % 1.0  # amplitude wrapped to [0,1)
            w.phase = (w.phase + delta_phase) % (2 * math.pi)

    def compute_entropy(self):
        """
        Computes the entropy of the waveform based on the distribution of amplitudes.
        This serves as a metric for the "symbolic randomness" in the system.
        """
        amplitudes = [w.amplitude for w in self.waveform]
        return shannon_entropy(amplitudes)

    def dynamic_feedback(self, target_entropy=0.8):
        """
        Adjusts the mutation_rate based on the current entropy of the waveform.
        If the entropy is too low, the mutation_rate is increased (to add more randomness).
        If the entropy is too high, the mutation_rate is decreased (to preserve some structure).
        """
        current_entropy = self.compute_entropy()
        if current_entropy < target_entropy:
            self.mutation_rate *= 1.1  # Increase mutation rate
        elif current_entropy > target_entropy:
            self.mutation_rate *= 0.9  # Decrease mutation rate
        # Clamp mutation_rate to reasonable bounds
        self.mutation_rate = max(0.01, min(self.mutation_rate, 0.5))
        return current_entropy

    def run_engine(self, iterations=100, target_entropy=0.8):
        """
        Runs the engine for a given number of iterations.
        Each iteration mutates the waveform, applies dynamic feedback,
        and prints the current entropy and mutation rate.
        """
        for i in range(iterations):
            self.mutate_waveform()
            entropy = self.dynamic_feedback(target_entropy)
            print(f"Iteration {i+1}: Entropy = {entropy:.4f}, Mutation Rate = {self.mutation_rate:.4f}")

if __name__ == "__main__":
    # Create a sample waveform of 10 WaveNumber objects with random initial states
    waveform = [WaveNumber(amplitude=random.random(), phase=random.random() * 2 * math.pi) for _ in range(10)]
    
    engine = WaveformEntropyEngine(waveform, mutation_rate=0.05)
    engine.run_engine(iterations=50, target_entropy=0.8)
