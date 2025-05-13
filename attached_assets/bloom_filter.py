import math
import re
import numpy as np
from wave_primitives import WaveNumber

# === Symbolic Bloom Filter ===
class SymbolicBloomFilter:
    def __init__(self, size=512, hash_count=3):
        self.size = size
        self.bit_array = [False] * size
        self.hash_count = hash_count

    def _symbolic_hashes(self, wv: WaveNumber):
        amp_scaled = int(abs(wv.amplitude) * 1000)
        phase_scaled = int((wv.phase % (2 * math.pi)) * 1000)
        combined = amp_scaled ^ phase_scaled
        return [
            int(abs(math.sin(combined * (i + 1)) * 1e5)) % self.size
            for i in range(self.hash_count)
        ]

    def add(self, wv: WaveNumber):
        for idx in self._symbolic_hashes(wv):
            self.bit_array[idx] = True

    def test(self, wv: WaveNumber):
        return all(self.bit_array[idx] for idx in self._symbolic_hashes(wv))


# === Symbolic Signal Metadata Input (replace with your own structured zones) ===
symbolic_datasets = {
    "Scan01": {
        "signal_height": 137.0,
        "symbolic_zones": """
            Regions of interest: 15–22m, 43–45m, 55–58m
        """
    },
    "Scan02": {
        "signal_height": 136.4,
        "symbolic_zones": """
            Observed zones: 30–35m, 40–60m
        """
    },
}

def extract_symbolic_ranges(metadata):
    ranges = []
    pattern = r"(\d+(\.\d+)?)[–-](\d+(\.\d+)?)"
    for match in re.findall(pattern, metadata):
        start = float(match[0])
        end = float(match[2])
        ranges.append((start, end))
    return ranges


# === Symbolic Resonance Simulation ===
def symbolic_depth_scan(height):
    depths = np.linspace(0, height, 200)
    resonance = []
    for d in depths:
        base = np.sin(2 * np.pi * d / (height / 2)) * np.exp(-d / height)
        phase = np.cos(d / 20)
        geom = 1.0 + 0.1 * np.sin(d / 10)
        sig = np.exp(-d / 100)
        combined = base * phase * geom * sig * np.random.normal(1.0, 0.05)
        resonance.append((d, combined))
    return resonance


# === Main Scanner: Symbolic Probabilistic Pattern Test ===
def run_symbolic_pattern_scan(key):
    config = symbolic_datasets[key]
    height = config["signal_height"]
    metadata = config["symbolic_zones"]
    symbolic_ranges = extract_symbolic_ranges(metadata)

    bf = SymbolicBloomFilter()
    resonance_data = symbolic_depth_scan(height)

    for depth, amp in resonance_data:
        if abs(amp) > 0.6:
            phase = depth % (2 * math.pi)
            wv = WaveNumber(abs(amp), phase)
            bf.add(wv)

    print(f"\n🧪 [Symbolic Probabilistic Scan: {key}]")
    print(f"Defined Zones: {symbolic_ranges}")
    hits = []

    for start, end in symbolic_ranges:
        sample_points = np.linspace(start, end, 10)
        for d in sample_points:
            amp = 1.0  # synthetic test input
            phase = d % (2 * math.pi)
            wv = WaveNumber(amp, phase)
            if bf.test(wv):
                hits.append((round(d, 2), round(phase, 3)))

    if hits:
        print(f"✅ Symbolic pattern matches found at:")
        for d, p in hits:
            print(f" → Depth: {d}m | Phase: {p} rad")
    else:
        print("❌ No matches detected in symbolic ranges.")

# === CLI Entry ===
if __name__ == "__main__":
    run_symbolic_pattern_scan("Scan01")
    run_symbolic_pattern_scan("Scan02")
