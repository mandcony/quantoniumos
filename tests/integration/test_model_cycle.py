#!/usr/bin/env python3
"""Integration tests validating GPT-2 variants with the RFT codec (lossless + quantized)."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import pytest

pytest.importorskip("torch")
import torch

pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.core.rft_vertex_codec import decode_state_dict, encode_state_dict  # noqa: E402


MODEL_VARIANTS = (
    "sshleifer/tiny-gpt2",
    "gpt2",
    "gpt2-medium",
)

SAMPLE_PROMPTS = (
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence enables new kinds of creativity.",
)


def _load_model(model_id: str) -> AutoModelForCausalLM:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    except Exception as exc:  # pragma: no cover - handles offline/no auth environments
        pytest.skip(f"Unable to download model '{model_id}': {exc}")
    model.eval()
    return model


def _decoded_state_to_torch(decoded: Mapping[str, Any], reference_state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, ref_tensor in reference_state.items():
        assert name in decoded, f"Decoded tensors missing entry '{name}'"
        value = decoded[name]
        torch_tensor = torch.from_numpy(value).to(dtype=ref_tensor.dtype)
        out[name] = torch_tensor
    return out


@pytest.mark.integration
@pytest.mark.parametrize("model_id", MODEL_VARIANTS)
def test_encode_decode_gpt2_variants_lossless(model_id: str) -> None:
    """Ensure lossless encode/decode round-trips exactly for GPT-2 variants."""

    model = _load_model(model_id)
    state = model.state_dict()

    encoded = encode_state_dict(state, tolerance=1e-10)
    decoded_state = decode_state_dict(encoded)

    for name, tensor in state.items():
        assert name in decoded_state, f"Decoded state missing tensor '{name}'"
        recon = torch.from_numpy(decoded_state[name]).to(dtype=tensor.dtype)
        assert recon.shape == tensor.shape, f"Shape mismatch for tensor '{name}'"
        if torch.is_floating_point(tensor):
            assert torch.allclose(recon, tensor.cpu(), rtol=1e-6, atol=1e-10), f"Mismatch for '{name}'"
        else:
            assert torch.equal(recon, tensor.cpu()), f"Mismatch for '{name}'"

    del model


@pytest.mark.integration
@pytest.mark.parametrize("model_id", MODEL_VARIANTS)
def test_quantized_gpt2_variants_perplexity_drift(model_id: str) -> None:
    """Quantized encode/decode should keep perplexity within 0.5% relative drift."""

    model = _load_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    state = model.state_dict()
    encoded = encode_state_dict(
        state,
        tolerance=1e-10,
        prune_threshold=0.0,
        quant_bits_amplitude=12,
        quant_bits_phase=10,
        ans_precision=12,
    )

    decoded_state = decode_state_dict(encoded)
    quantized_model = AutoModelForCausalLM.from_config(model.config)
    quantized_model.eval()

    quantized_state = _decoded_state_to_torch(decoded_state, state)
    missing = set(quantized_model.state_dict().keys()) - set(quantized_state.keys())
    assert not missing, f"Quantized state missing parameters: {sorted(missing)[:5]}"
    quantized_model.load_state_dict(quantized_state, strict=True)

    baseline_losses = []
    quantized_losses = []
    for prompt in SAMPLE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        labels = inputs["input_ids"]
        with torch.no_grad():
            baseline_loss = model(**inputs, labels=labels).loss.detach().cpu()
            quantized_loss = quantized_model(**inputs, labels=labels).loss.detach().cpu()
        baseline_losses.append(baseline_loss)
        quantized_losses.append(quantized_loss)

    baseline_perplexity = torch.exp(torch.stack(baseline_losses)).mean().item()
    quantized_perplexity = torch.exp(torch.stack(quantized_losses)).mean().item()
    relative_drift = abs(quantized_perplexity - baseline_perplexity) / baseline_perplexity

    assert relative_drift < 5e-3, (
        f"Perplexity drift {relative_drift:.4%} exceeds 0.5% budget for model '{model_id}'"
    )

    del model
    del quantized_model