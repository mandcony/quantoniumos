#!/usr/bin/env python3
"""
Geometric Waveform Hashing Pipeline
Implements the structure-preserving hash pipeline described in the QuantoniumOS paper

Pipeline: x → Ψ(x) → Manifold Mapping → Topological Embedding → Digest
"""

import hashlib
import hmac
import struct
from typing import Union, Optional
import sys
import os
import secrets

# Add core path for RFT integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    from canonical_true_rft import CanonicalTrueRFT
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    CanonicalTrueRFT = None
    np = None


class GeometricWaveformHash:
    """
    Geometric waveform hashing with RFT-based manifold mapping.

    Implements the research pipeline with controllable diffusion properties
    suitable for cryptographic applications (research purposes only).
    """

    def __init__(
        self,
        size: int = 64,
        manifold_dim: int = 16,
        *,
        secret_salt: Optional[bytes] = None,
        signing_key: Optional[bytes] = None,
        quantization_scale: int = 1_000_000,
    ):
        """
        Initialize geometric hash pipeline.

        Args:
            size: RFT transform size
            manifold_dim: Dimension of manifold projection
            secret_salt: Deployment-specific secret salt for manifold/RFT derivation
            signing_key: Optional explicit key for the keyed digest
            quantization_scale: Scaling factor applied before quantization
        """

        self.size = size
        self.manifold_dim = manifold_dim
        self.quantization_scale = max(1_000, quantization_scale)

        self._secret_material = self._resolve_secret(secret_salt, "GEOMETRIC_HASH_SECRET")
        self._mac_key = self._resolve_secret(signing_key, "GEOMETRIC_HASH_SIGNING_KEY", allow_random=False)
        if self._mac_key is None:
            self._mac_key = hashlib.sha256(self._secret_material + b"GEOM_HASH_MAC").digest()

        # Initialize RFT if available
        if CanonicalTrueRFT is not None:
            basis_secret = hashlib.sha256(self._secret_material + b"GEOM_HASH_RFT_BASIS").digest()
            self.rft = CanonicalTrueRFT(size, basis_secret=basis_secret)
        else:
            self.rft = None
            print("Warning: NumPy not available, using classical hash fallback")

        # Precompute manifold projection matrix using deployment secret
        self.manifold_matrix = self._generate_manifold_matrix()

    def _resolve_secret(self, provided: Optional[bytes], env_var: str, *, allow_random: bool = True) -> Optional[bytes]:
        """Normalize secret inputs, preferring environment overrides."""

        if provided is None:
            env_val = os.getenv(env_var)
            if env_val:
                provided = env_val.encode('utf-8')

        if provided is None:
            if not allow_random:
                return None
            provided = secrets.token_bytes(32)

        if isinstance(provided, str):  # type: ignore[unreachable]
            provided = provided.encode('utf-8')

        if not isinstance(provided, (bytes, bytearray)):
            raise TypeError(f"Secret for {env_var} must be bytes-like")

        if len(provided) == 0:
            raise ValueError(f"Secret for {env_var} must not be empty")

        return bytes(provided)

    def _generate_manifold_matrix(self):
        """Generate secret-derived manifold projection matrix."""
        if np is None:
            return None

        seed_material = hashlib.sha256(self._secret_material + b"GEOM_HASH_MANIFOLD").digest()
        seed_value = int.from_bytes(seed_material[:16], 'big')
        rng = np.random.default_rng(seed_value)

        # Random projection matrix derived from secret seed
        matrix = rng.standard_normal((self.manifold_dim, self.size))

        # Normalize rows
        for i in range(self.manifold_dim):
            norm_val = np.linalg.norm(matrix[i])
            if norm_val > 0:
                matrix[i] /= norm_val

        return matrix

    def _bytes_to_signal(self, data: bytes) -> Union[np.ndarray, None]:
        """Convert bytes to complex signal for RFT processing."""
        if np is None:
            return None

        # Pad or truncate to size
        if len(data) < self.size:
            data = data + b'\x00' * (self.size - len(data))
        else:
            data = data[:self.size]

        # Convert to complex signal
        signal = np.zeros(self.size, dtype=complex)
        for i, byte_val in enumerate(data):
            # Map byte to complex unit circle
            phase = 2 * np.pi * byte_val / 256
            signal[i] = np.exp(1j * phase)

        return signal

    def _manifold_mapping(self, rft_coeffs: np.ndarray) -> np.ndarray:
        """Project RFT coefficients onto lower-dimensional manifold."""
        if self.manifold_matrix is None:
            return rft_coeffs[:self.manifold_dim]

        # Complex to real projection (amplitude and phase)
        real_coeffs = np.concatenate([
            np.real(rft_coeffs),
            np.imag(rft_coeffs)
        ])

        # Truncate to transform size
        real_coeffs = real_coeffs[:self.size]

        # Project onto manifold
        manifold_point = self.manifold_matrix @ real_coeffs
        return manifold_point

    def _topological_embedding(self, manifold_point: np.ndarray) -> bytes:
        """Create topological embedding preserving geometric structure."""
        # Quantize manifold coordinates with higher precision
        quantized = np.round(manifold_point * self.quantization_scale).astype(int)

        # Convert to bytes with controlled avalanche
        embedding = b""
        for val in quantized:
            # Ensure positive values and pack as 4-byte integers
            val_unsigned = val % (2**31)  # Keep in positive range
            embedding += struct.pack('>I', val_unsigned)

        return embedding

    def hash_classical_fallback(self, data: bytes) -> bytes:
        """Classical hash fallback when RFT is unavailable."""
        mac_data = b"GEOMETRIC_WAVEFORM_HASH_FALLBACK" + data
        return hmac.new(self._mac_key, mac_data, hashlib.sha256).digest()

    def hash_with_rft(self, data: bytes) -> bytes:
        """Full geometric waveform hashing with RFT pipeline."""
        # Step 1: Convert input to signal
        signal = self._bytes_to_signal(data)
        if signal is None:
            return self.hash_classical_fallback(data)

        # Step 2: Apply RFT transform
        rft_coeffs = self.rft.forward_transform(signal)

        # Step 3: Manifold mapping
        manifold_point = self._manifold_mapping(rft_coeffs)

        # Step 4: Topological embedding
        embedding = self._topological_embedding(manifold_point)

        # Step 5: Final keyed digest
        mac_data = b"GEOMETRIC_WAVEFORM_HASH_v2" + embedding + data
        return hmac.new(self._mac_key, mac_data, hashlib.sha256).digest()

    def hash(self, data: Union[bytes, str]) -> bytes:
        """
        Main hash interface supporting both classical and RFT modes.

        Args:
            data: Input data to hash

        Returns:
            32-byte hash digest
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        if self.rft is not None:
            return self.hash_with_rft(data)
        else:
            return self.hash_classical_fallback(data)

    def hex_digest(self, data: Union[bytes, str]) -> str:
        """Return hash as hexadecimal string."""
        return self.hash(data).hex()

    def get_diffusion_metrics(self) -> dict:
        """Analyze diffusion properties for research validation."""
        if self.rft is None:
            return {'error': 'RFT not available for metrics analysis'}

        # Test with sample inputs
        test_inputs = [
            b"test_message_1",
            b"test_message_2",  # 1-bit difference
            b"TEST_MESSAGE_1",  # Case difference
        ]

        hashes = [self.hash(inp) for inp in test_inputs]

        # Calculate bit differences
        def bit_diff(h1, h2):
            diff_bits = 0
            for b1, b2 in zip(h1, h2):
                diff_bits += bin(b1 ^ b2).count('1')
            return diff_bits

        # Avalanche analysis
        bit_diff_12 = bit_diff(hashes[0], hashes[1])
        bit_diff_13 = bit_diff(hashes[0], hashes[2])

        total_bits = len(hashes[0]) * 8
        avalanche_12 = bit_diff_12 / total_bits
        avalanche_13 = bit_diff_13 / total_bits

        return {
            'avalanche_small_change': avalanche_12,
            'avalanche_case_change': avalanche_13,
            'target_avalanche': 0.5,
            'total_output_bits': total_bits,
            'rft_enabled': True,
            'manifold_dimension': self.manifold_dim
        }


def validate_geometric_hashing() -> dict:
    """
    Validate geometric waveform hashing as described in the research paper.
    """
    print("Validating Geometric Waveform Hashing...")

    hasher = GeometricWaveformHash(size=64, manifold_dim=16)

    # Test basic functionality
    test_message = b"QuantoniumOS Geometric Hash Test Vector"
    hash_result = hasher.hash(test_message)
    hex_result = hasher.hex_digest(test_message)

    # Test deterministic behavior (depends on deployment secret)
    hash_repeat = hasher.hash(test_message)
    deterministic = (hash_result == hash_repeat)

    # Get diffusion metrics
    metrics = hasher.get_diffusion_metrics()

    # Test different inputs
    different_hash = hasher.hash(test_message + b"_modified")
    different_result = (hash_result != different_hash)

    results = {
        'basic_functionality': len(hash_result) == 32,
        'deterministic': deterministic,
        'different_inputs_differ': different_result,
        'hex_format_valid': len(hex_result) == 64,
        'diffusion_metrics': metrics,
        'sample_hash': hex_result[:16] + "...",  # First 16 hex chars
        'rft_integration': hasher.rft is not None
    }

    print(f"✓ Basic functionality: {'PASS' if results['basic_functionality'] else 'FAIL'}")
    print(f"✓ Deterministic: {'PASS' if deterministic else 'FAIL'}")
    print(f"✓ RFT integration: {'ENABLED' if results['rft_integration'] else 'FALLBACK'}")

    if 'avalanche_small_change' in metrics:
        print(f"✓ Avalanche (small change): {metrics['avalanche_small_change']:.3f}")

    return results


if __name__ == "__main__":
    # Run validation for research paper
    validation_results = validate_geometric_hashing()

    print("\n" + "="*50)
    print("GEOMETRIC HASH VALIDATION")
    print("="*50)

    # Demonstrate usage
    hasher = GeometricWaveformHash()

    test_messages = [
        "Hello QuantoniumOS",
        "Research Paper 2025",
        "RFT Geometric Hashing Pipeline"
    ]

    for msg in test_messages:
        hash_hex = hasher.hex_digest(msg)
        print(f"'{msg}' → {hash_hex[:32]}...")
