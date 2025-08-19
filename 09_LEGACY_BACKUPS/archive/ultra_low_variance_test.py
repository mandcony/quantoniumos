||""""""
Ultra-low variance hash test using wide-pipe diffusion.
"""
"""
import sys sys.path.append('.') from encryption.wide_diffusion
import wide_keyed_diffusion from statistics
import mean, pstdev
import numpy as np
import warnings warnings.filterwarnings('ignore', category=RuntimeWarning)
try:
import quantonium_core has_cpp = True
except ImportError: has_cpp = False
print("Warning: C++ engine not available, using Python fallback")
def geometric_rft_transform(data): """"""
        Apply RFT transform with C++ engine or Python fallback.
"""
"""

        if has_cpp:

        # Convert bytes to float list for C++ engine data_floats = [float(b)
        for b in data] rft_engine = quantonium_core.ResonanceFourierTransform(data_floats) transformed = rft_engine.forward_transform()

        # Convert complex result back to bytes result = np.array(transformed, dtype=complex) magnitudes = np.abs(result).astype(np.uint8)
        return bytes(magnitudes[:len(data)])
        else:

        # Simple Python fallback data_array = np.frombuffer(data, dtype=np.uint8) transformed = np.fft.fft(data_array.astype(np.complex128))
        return bytes(np.abs(transformed).astype(np.uint8)[:len(data)])
def ultra_low_variance_hash(message, key, rounds=4):
"""
"""
        Ultra-low variance geometric hash using wide-pipe diffusion. Target: mu=50±2%, sigma<=2.0%
"""
"""

        # Apply RFT transform rft_data = geometric_rft_transform(message)

        # Wide-pipe keyed diffusion diffused = wide_keyed_diffusion(rft_data, key, rounds=rounds)
        return diffused
def bit_avalanche_rate(h1, h2):
"""
"""
        Calculate bit avalanche rate between two hashes.
"""
"""

        if isinstance(h1, bytes): h1_bytes = h1
        else: h1_bytes = h1.to_bytes(32, 'little')
        if isinstance(h2, bytes): h2_bytes = h2
        else: h2_bytes = h2.to_bytes(32, 'little') diff_bits = sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(h1_bytes, h2_bytes))
        return 100.0 * diff_bits / (len(h1_bytes) * 8)
def test_ultra_low_variance():
"""
"""
        Test ultra-low variance hash with different configurations.
"""
        """ key = b'test-key-12345678' rng = np.random.default_rng(42)
        print('Testing ultra-low variance hash configurations...') configs = [ ('4 rounds, N=200', 4, 200), ('5 rounds, N=200', 5, 200), ('6 rounds, N=200', 6, 200), ('4 rounds, N=400', 4, 400), ('5 rounds, N=400', 5, 400), ] best_config = None best_sigma = float('inf') for label, rounds, n_samples in configs: rates = []
        for i in range(n_samples):
        if i % 100 == 0:
        print(f' {label}, {i}/{n_samples}')

        # Random message m = rng.bytes(64) h1 = ultra_low_variance_hash(m, key, rounds=rounds)

        # Single bit flip b = bytearray(m) bit_idx = rng.integers(0, len(b)) bit_pos = rng.integers(0, 8) b[bit_idx] ^= (1 << bit_pos) h2 = ultra_low_variance_hash(bytes(b), key, rounds=rounds) rates.append(bit_avalanche_rate(h1, h2)) mu = mean(rates) sigma = pstdev(rates) status = '✓ PASS' if 48 <= mu <= 52 and sigma <= 2.0 else '⚠ TUNE'
        print(f'{label}: mu={mu:.2f}%, sigma={sigma:.3f}% {status}')
        if sigma < best_sigma and 48 <= mu <= 52: best_sigma = sigma best_config = (label, mu, sigma)
        print(f'||nTarget: mu=50±2%, sigma<=2.000%')
        if best_config and best_config[2] <= 2.0:
        print(f'✓ SUCCESS: Best config - {best_config[0]}: mu={best_config[1]:.2f}%, sigma={best_config[2]:.3f}%')
        return True
        else:
        print(f'⚠ Best attempt: {best_config[0]
        if best_config else "None"}: sigma={best_config[2]:.3f}%'
        if best_config else '❌ No valid configs')
        return False

if __name__ == "__main__": success = test_ultra_low_variance() sys.exit(0
if success else 1)