# apps/bloom_filter.py (refined)

class BloomFilter(GeometricContainer):
    def __init__(self, size: int, num_hash_functions: int, amplitude: complex = complex(1.0, 0.0)):
        super().__init__(amplitude)
        self.size = size
        self.num_hash_functions = num_hash_functions
        self.bit_array = [0]*size

    def validate(self, items: list):
        print("Validating Bloom filter with resonance...")

        for item in items:
            # If item is (WaveNumber) or can be converted, do something:
            hashed_item = self._wavehash(item)
            for i in range(self.num_hash_functions):
                slot = (hash(f"{hashed_item}{i}") ^ hash(self.amplitude)) % self.size
                self.bit_array[slot] = 1

        fill_ratio = sum(self.bit_array) / self.size
        self.resonance = fill_ratio * abs(self.amplitude.real)
        return self.bit_array

    def _wavehash(self, item) -> str:
        """
        Example: if item is a WaveNumber, convert to (amplitude,phase) str. Otherwise just str().
        """
        if hasattr(item, 'amplitude') and hasattr(item, 'phase'):
            # Turn into a short string
            return f"W{item.amplitude:.2f}P{item.phase:.2f}"
        else:
            return str(item)
