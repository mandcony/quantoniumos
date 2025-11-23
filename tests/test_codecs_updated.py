
import sys
import os
import numpy as np

# Add repo root to path
# sys.path.insert(0, os.getcwd())

try:
    from algorithms.compression.vertex.rft_vertex_codec import encode_tensor, decode_tensor
    from algorithms.compression.hybrid.rft_hybrid_codec import encode_tensor_hybrid
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_vertex_codec():
    print("Testing Vertex Codec...")
    data = np.random.randn(100).astype(np.float32)
    encoded = encode_tensor(data, chunk_size=100)
    decoded = decode_tensor(encoded)
    
    err = np.max(np.abs(data - decoded))
    print(f"Vertex Codec Max Error: {err}")
    if err > 1e-5:
        print("Vertex Codec Failed!")
        sys.exit(1)
    else:
        print("Vertex Codec Passed.")

def test_hybrid_codec():
    print("Testing Hybrid Codec...")
    data = np.random.randn(100).astype(np.float32)
    # Hybrid is lossy, so we just check it runs
    encoded = encode_tensor_hybrid(data, prune_threshold=0.0, quant_amp_bits=8, quant_phase_bits=8)
    print("Hybrid Codec Encode Successful.")
    # Note: Hybrid decode logic is more complex and might require residual predictor, 
    # but the encode path exercises the transform.

if __name__ == "__main__":
    test_vertex_codec()
    test_hybrid_codec()
