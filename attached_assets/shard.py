import numpy as np
from bloom_filter import BloomFilter, hash1, hash2  # Fixed import

class Shard:
    def __init__(self, containers, filter_size=256, hash_functions=[hash1, hash2]):
        """Initializes the Shard with a Bloom filter for fast resonance checking."""
        self.containers = containers
        self.bloom = self._build_bloom(containers, filter_size, hash_functions)

    def _build_bloom(self, containers, filter_size, hash_functions):
        """Constructs a Bloom filter for quick resonance matching."""
        bloom = BloomFilter(filter_size, hash_functions)
        for container in containers:
            if container.resonant_frequencies:
                for freq in container.resonant_frequencies:
                    bloom.add(str(freq))  # Store frequencies in Bloom filter
        return bloom

    def search(self, target_freq, threshold=0.1):
        """Searches for containers matching the target frequency."""
        if not self.bloom.test(str(target_freq)):
            return []  # Fast rejection if Bloom filter says no match

        # Precise verification
        matched_containers = [
            container for container in self.containers
            if container.check_resonance(target_freq, threshold)
        ]
        return matched_containers

# TESTING SHARD SYSTEM
if __name__ == "__main__":
    from geometric_container import GeometricContainer  # Fixed import

    # Create test containers
    containers = [
        GeometricContainer(f"Container_{i}", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        for i in range(5)
    ]
    
    # Assign resonance frequencies
    for i, container in enumerate(containers):
        container.resonant_frequencies = [i * 0.1]  # Different resonance frequencies

    shard = Shard(containers)

    test_freq = 0.2
    results = shard.search(test_freq)

    print(f"Searching for frequency {test_freq}:")
    print(f"Found {len(results)} matching containers: {[c.id for c in results]}")
