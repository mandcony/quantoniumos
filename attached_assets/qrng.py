# qrng.py
import random
import math

from wave_primitives import WaveNumber  # Assuming wave_primitives.py exists

class WaveBasedQRNG:
    """
    A rudimentary wave-based pseudorandom number generator (PRNG).
    """

    def __init__(self, seed: str = None):
        if seed is not None:
            self.seed(seed)
        else:
            self.state = WaveNumber(amplitude=random.random(), phase=random.random() * 2 * math.pi)
        self.multiplier = WaveNumber(0.618034, 1.618034)  # Golden ratio
        self.increment = WaveNumber(0.318309, 0.318309)  # 1/pi

    def seed(self, seed_str: str):
        """Seed with string"""
        seed_val = sum(ord(c) for c in seed_str)
        self.state = WaveNumber(amplitude=seed_val % 1.0, phase=(seed_val * 0.618034) % (2 * math.pi))

    def random(self) -> float:
        """Generate random float between 0 and 1."""
        self.state.amplitude = (self.state.amplitude * self.multiplier.amplitude + self.increment.amplitude) % 1.0
        self.state.phase = (self.state.phase * self.multiplier.phase + self.increment.phase) % (2 * math.pi)
        return abs(math.sin(self.state.phase) * self.state.amplitude)

if __name__ == "__main__":
    qrng = WaveBasedQRNG(seed="initial seed")
    for _ in range(5):
        print(qrng.random())

    qrng2 = WaveBasedQRNG(seed="initial seed")
    for _ in range(5):
        print(qrng2.random())