"""Round-trip tests for the symbolic RFT vertex codec."""
import numpy as np

from core import rft_vertex_codec as codec


def test_vertex_codec_roundtrip() -> None:
    rng = np.random.default_rng(2025)
    tensor = rng.standard_normal((16, 16)).astype(np.float32)

    container = codec.encode_tensor(tensor)
    restored = codec.decode_tensor(container)

    max_error = float(np.max(np.abs(restored.astype(np.float64) - tensor.astype(np.float64))))
    assert max_error < 1e-8


def test_roundtrip_helper_returns_success() -> None:
    tensor = np.arange(64, dtype=np.float32).reshape(8, 8)
    ok, max_err = codec.roundtrip_tensor(tensor)
    assert ok
    assert max_err < 1e-8


def test_vertex_codec_roundtrip_large_tensor() -> None:
    rng = np.random.default_rng(314159)
    tensor = rng.standard_normal(1_048_576).astype(np.float32)  # ~1 MB payload
    ok, max_err = codec.roundtrip_tensor(
        tensor,
        chunk_size=8192,
    )
    assert ok
    assert max_err < 1e-6
