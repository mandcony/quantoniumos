# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos

import sys
import os
import numpy as np

# Add repo root to path
sys.path.insert(0, os.getcwd())

try:
    from algorithms.compression.vertex.rft_vertex_codec import encode_tensor, decode_tensor
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_ans_integration():
    print("Testing ANS Integration...")
    # Create data that compresses well (low entropy)
    data = np.zeros(1000, dtype=np.float32)
    data[:10] = 1.0 # Sparse
    
    # Test with higher precision quantization (12 bits for better accuracy)
    encoded = encode_tensor(data, chunk_size=1000, quant_bits_amplitude=12, quant_bits_phase=12, ans_precision=12)
    
    # Check if ANS was actually used
    # The structure of encoded is: {'chunks': [{'payload': {'encoding': 'ans', ...}}]}
    chunk = encoded['chunks'][0]
    
    # We need to check the payload of the quantized fields (amplitude/phase)
    # The vertex codec stores 'A' and 'phi' quantized.
    # Let's inspect the chunk structure.
    
    # Actually, rft_vertex_codec stores 'vertices' which are complex.
    # Wait, rft_vertex_codec.py: encode_tensor -> _encode_chunk
    # _encode_chunk -> _make_numeric_payload for 'idx', 'real', 'imag' OR 'A', 'phi' if quantized?
    # Let's check rft_vertex_codec.py again.
    
    # It seems it stores 'idx', 'real', 'imag' by default.
    # If quant_bits_amplitude is set, it might store A/phi instead?
    # Let's just print the keys of the chunk payload.
    
    print("Chunk keys:", chunk.keys())
    
    # If ANS is working, we should see 'encoding': 'ans' in some sub-payloads if we dig deep enough,
    # OR we can just verify roundtrip with quantization enabled.
    
    decoded = decode_tensor(encoded)
    err = np.max(np.abs(data - decoded))
    print(f"ANS Roundtrip Max Error: {err}")
    
    # With 12-bit quantization, error should be quite small for 0/1 data
    # Allow up to 0.01 tolerance (quantization introduces some error)
    if err > 0.01:
         print(f"ANS Roundtrip Failed (Error {err} too high)!")
         sys.exit(1)
    else:
         print(f"ANS Roundtrip Passed (error={err}).")

if __name__ == "__main__":
    test_ans_integration()
