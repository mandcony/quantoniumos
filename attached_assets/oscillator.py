import math
import numpy as np
from geometric_container import GeometricContainer

class Oscillator(GeometricContainer):
    def __init__(self, frequency: float, amplitude: complex, phase: float):
        # Call the GeometricContainer initializer with a dummy id and a default vertex list.
        # Here we use a single vertex [0,0,0] as a placeholder.
        super().__init__(id="Oscillator", vertices=[[0, 0, 0]])
        self.frequency = frequency
        self.amplitude = amplitude  # Base amplitude for waveform generation
        self.phase = phase
        self.waveform = []  # To store generated samples

    def update(self, dt: float):
        t = len(self.waveform) * dt
        sample = self.amplitude * np.exp(1j * (2 * math.pi * self.frequency * t + self.phase))
        self.waveform.append(sample)

    def get_amplitude(self):
        return self.waveform[-1] if self.waveform else self.amplitude

def validate_oscillator(osc: Oscillator, duration: float, dt: float = 0.1) -> list:
    steps = int(duration / dt)
    for _ in range(steps):
        osc.update(dt)
    total_norm = math.sqrt(sum(abs(s) ** 2 for s in osc.waveform))
    target_norm = 1.5
    tolerance = 0.05
    if not (target_norm * (1 - tolerance) <= total_norm <= target_norm * (1 + tolerance)):
        scale = target_norm / total_norm
        osc.waveform = [s * scale for s in osc.waveform]
        print(f"Waveform normalized: Total norm was {total_norm:.2f}, scaled to {target_norm}")
    return osc.waveform

# For testing the oscillator module standalone:
if __name__ == "__main__":
    frequency = 1.0
    initial_amplitude = 1.0 + 0j
    phase = 0.0
    osc = Oscillator(frequency, initial_amplitude, phase)
    
    duration = 5.0
    dt = 0.1
    waveform = validate_oscillator(osc, duration, dt)
    
    print(f"Generated {len(waveform)} samples.")
    total_norm = math.sqrt(sum(abs(s) ** 2 for s in waveform))
    print(f"Total norm: {total_norm:.2f}")
