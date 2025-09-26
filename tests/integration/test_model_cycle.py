#!/usr/bin/env python3
"""Integration test covering end-to-end encode/decode on a tiny Hugging Face model."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
from transformers import AutoModel  # noqa: E402

from src.core.rft_vertex_codec import decode_state_dict, encode_state_dict  # noqa: E402


@pytest.mark.integration
def test_encode_decode_tiny_gpt2_roundtrip():
    """Load sshleifer/tiny-gpt2, run encode/decode, and validate tensors via torch.allclose."""
    model_id = "sshleifer/tiny-gpt2"

    try:
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    except Exception as exc:  # pragma: no cover - covers offline/unauthorized scenarios
        pytest.skip(f"Unable to download model '{model_id}': {exc}")

    model.eval()
    state = model.state_dict()

    encoded = encode_state_dict(state, tolerance=1e-10)
    decoded_state = decode_state_dict(encoded)

    for name, tensor in state.items():
        assert name in decoded_state, f"Decoded state missing tensor '{name}'"
        decoded = torch.from_numpy(decoded_state[name]).to(dtype=tensor.dtype)
        assert decoded.shape == tensor.shape, f"Shape mismatch for tensor '{name}'"
        if torch.is_floating_point(tensor):
            assert torch.allclose(decoded, tensor.cpu(), rtol=1e-6, atol=1e-10), f"Mismatch for '{name}'"
        else:
            assert torch.equal(decoded, tensor.cpu()), f"Mismatch for '{name}'"

    # Clean up to free memory
    del model