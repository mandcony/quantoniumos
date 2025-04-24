import math
import numpy as np
from apps.geometric_container import GeometricContainer

class Oscillator(GeometricContainer):
    def __init__(self, frequency: float, amplitude: complex, phase: float):
        super().__init__(amplitude)
        self.frequency = frequency
        self.phase = phase
        self.waveform = []

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