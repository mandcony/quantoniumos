#!/usr/bin/env python3
"""Basic tests for the RFT Vertex Codec MVP."""
import numpy as np
from src.core.rft_vertex_codec import decode_tensor, encode_state_dict, encode_tensor, roundtrip_tensor

def test_roundtrip_small_vector():
    x = np.linspace(-1, 1, 37, dtype=np.float64)
    ok, max_err = roundtrip_tensor(x)
    assert ok, f"Roundtrip failed, max_err={max_err}"  # Expect near machine precision


def test_roundtrip_matrix():
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((8, 8), dtype=np.float64)
    container = encode_tensor(mat)
    recon = decode_tensor(container, verify_checksum=False)
    assert recon.shape == mat.shape
    diff = np.abs(recon - mat).max()
    assert diff < 1e-10, f"Matrix roundtrip error too large: {diff}"  


def test_stateful_multiple_chunks():
    # Force multiple chunks by reducing chunk size
    data = np.arange(5000, dtype=np.float64) / 1000.0
    container = encode_tensor(data, chunk_size=1024)
    assert len(container['chunks']) > 1
    recon = decode_tensor(container, verify_checksum=False)
    assert np.allclose(recon, data)


def test_integer_tensor_roundtrip():
    ints = np.arange(128, dtype=np.int32).reshape(32, 4)
    container = encode_tensor(ints, chunk_size=64)
    assert container['type'] == 'rft_vertex_tensor_container'
    recon = decode_tensor(container, verify_checksum=False)
    assert recon.dtype == ints.dtype
    assert np.array_equal(recon, ints)


def test_boolean_tensor_raw_container():
    bools = (np.arange(33) % 2 == 0).reshape(3, 11)
    container = encode_tensor(bools)
    assert container['type'] == 'rft_raw_tensor_container'
    recon = decode_tensor(container, verify_checksum=False)
    assert recon.dtype == bools.dtype
    assert np.array_equal(recon, bools)


def test_quantized_checksum_fallback_accepts():
    vec = np.linspace(-1, 1, 257, dtype=np.float64)
    container = encode_tensor(vec, tolerance=1e-8)
    container['checksum'] = 'mismatch'
    recon, status = decode_tensor(container, verify_checksum=True, return_status=True)
    assert status['used_secondary_checksum'] is True
    assert np.allclose(recon, vec)


def test_lossy_quantization_and_pruning():
    rng = np.random.default_rng(321)
    vec = rng.standard_normal(1024, dtype=np.float64)
    container = encode_tensor(
        vec,
        chunk_size=256,
        prune_threshold=0.05,
        quant_bits_amplitude=8,
        quant_bits_phase=9,
        ans_precision=12,
    )
    assert container['codec']['mode'] == 'lossy'
    recon = decode_tensor(container, verify_checksum=False)
    assert recon.shape == vec.shape
    max_err = float(np.max(np.abs(recon - vec)))
    assert max_err < 0.2  # lossy pipeline should stay within reasonable distortion
    chunk_modes = {chunk.get('codec', {}).get('mode') for chunk in container['chunks']}
    assert chunk_modes - {None}  # ensure codec metadata populated


def test_encode_state_dict_lossy_metadata():
    weights = {
        'linear.weight': np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(32, 4)
    }
    encoded = encode_state_dict(
        weights,
        prune_threshold=0.02,
        quant_bits_amplitude=6,
        quant_bits_phase=6,
        ans_precision=10,
    )
    assert encoded['codec']['prune_threshold'] == 0.02
    tensor_container = encoded['tensors']['linear.weight']
    assert tensor_container['codec']['mode'] == 'lossy'
    recon = decode_tensor(tensor_container, verify_checksum=False)
    assert recon.shape == weights['linear.weight'].shape
    assert float(np.max(np.abs(recon - weights['linear.weight']))) < 0.25

if __name__ == '__main__':  # Manual ad-hoc run
    test_roundtrip_small_vector()
    test_roundtrip_matrix()
    test_stateful_multiple_chunks()
    print('RFT Vertex Codec basic tests passed.')
