import math

class WaveNumber:
    def __init__(self, amplitude: float, phase: float = 0.0):
        self.amplitude = float(amplitude)
        self.phase = float(phase)

    def __repr__(self):
        return f"WaveNumber(amplitude={self.amplitude:.3f}, phase={self.phase:.3f})"

    def scale_amplitude(self, factor: float):
        self.amplitude *= factor

    def shift_phase(self, delta: float):
        self.phase += delta
        self.phase %= 2 * math.pi  # Normalize phase to [0, 2π)

    def to_float(self) -> float:
        return self.amplitude