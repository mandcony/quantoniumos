import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.apps.compressed_model_router import CompressedModelRouter
from src.core.rft_hybrid_codec import encode_tensor_hybrid


class DummyAutoTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 0

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class DummyAutoConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class DummyAutoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.loaded_state = None

    @classmethod
    def from_config(cls, config):
        return cls()

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state = state_dict

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids=None, max_new_tokens=1, **kwargs):
        prefix = input_ids.shape[1] if input_ids is not None else 0
        total = prefix + max_new_tokens
        return torch.zeros((1, total), dtype=torch.long)


@pytest.fixture(autouse=True)
def stub_huggingface(monkeypatch):
    monkeypatch.setattr("src.apps.compressed_model_router.AutoTokenizer", DummyAutoTokenizer)
    monkeypatch.setattr("src.apps.compressed_model_router.AutoConfig", DummyAutoConfig)
    monkeypatch.setattr("src.apps.compressed_model_router.AutoModelForCausalLM", DummyAutoModel)
    yield


def _write_manifest(path: Path, manifest: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh)


def test_discover_state_dict_models_registers_entry(tmp_path):
    decoded_dir = tmp_path / "decoded_models" / "toy_model"
    decoded_dir.mkdir(parents=True)
    state_path = decoded_dir / "pytorch_model.bin"
    torch.save({"linear.weight": torch.zeros(1, 1)}, state_path)

    manifest = {
        "model_name": "toy-model",
        "metrics": {
            "original_size_bytes": 1024,
            "encoded_size_bytes": 128,
        },
        "manifests": [
            {
                "tensors": [
                    {"tensor_name": "linear.weight", "numel": 1},
                ]
            }
        ],
    }
    _write_manifest(tmp_path / "encoded_models" / "toy_model_lossless" / "manifest.json", manifest)

    router = CompressedModelRouter(base_path=tmp_path)
    key = "toy-model::rft"
    assert key in router.model_registry
    entry = router.model_registry[key]
    assert entry["storage_type"] == "state_dict"
    assert entry["compression_method"] == "rft_vertex_lossless"
    assert entry["original_parameters"] == 1


def test_load_state_dict_model_uses_stubbed_hf(tmp_path):
    decoded_dir = tmp_path / "decoded_models" / "toy_model"
    decoded_dir.mkdir(parents=True)
    state_path = decoded_dir / "pytorch_model.bin"
    torch.save({"linear.weight": torch.zeros(1, 1)}, state_path)

    router = CompressedModelRouter(base_path=tmp_path)
    key = "toy_model::rft"
    # Manually register entry to avoid manifest requirement for this focused test
    router.model_registry[key] = {
        "model_id": "toy-model",
        "file_path": str(state_path),
        "storage_type": "state_dict",
        "hf_reference": "toy-model",
    }

    loaded = router.load_model(key)
    assert loaded is not None
    assert loaded["type"] == "hf_transformer"
    assert isinstance(loaded["model"], DummyAutoModel)


def test_hybrid_manifest_discovery_and_loading(tmp_path):
    encoded_dir = tmp_path / "encoded_models" / "hybrid_test"
    tensors_dir = encoded_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    tensor = np.random.randn(16).astype(np.float32).reshape(4, 4)
    result = encode_tensor_hybrid(tensor, prune_threshold=0.0, quant_amp_bits=6, quant_phase_bits=5, tensor_name="linear.weight")
    container_path = tensors_dir / "linear_weight.json"
    with container_path.open("w", encoding="utf-8") as fh:
        json.dump(result.container, fh)

    manifest = {
        "model_name": "hybrid-model",
        "tensors": [
            {
                "tensor_name": "linear.weight",
                "file": "tensors/linear_weight.json",
                "kept_coeff": result.container["kept_coeff"],
                "total_coeff": result.container["total_coeff"],
            }
        ],
        "metrics": {
            "coeff_total": result.container["total_coeff"],
            "kept_coeff_total": result.container["kept_coeff"],
        },
        "codec": {
            "prune_threshold": 0.0,
            "quant_amp_bits": 6,
            "quant_phase_bits": 5,
        },
    }
    _write_manifest(encoded_dir / "manifest_hybrid.json", manifest)

    router = CompressedModelRouter(base_path=tmp_path)
    key = "hybrid-model::hybrid"
    assert key in router.model_registry
    entry = router.model_registry[key]
    assert entry["compression_method"] == "rft_hybrid"
    assert entry["storage_type"] == "hybrid"

    loaded = router.load_model(key)
    assert loaded is not None
    assert loaded["type"] == "hf_transformer"