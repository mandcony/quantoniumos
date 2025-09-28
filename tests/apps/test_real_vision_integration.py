import os
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def test_real_image_adapter_uses_local_path(monkeypatch, tmp_path):
    """Ensure the real adapter resolves model paths and generates images."""
    generated_kwargs = {}

    dummy_module = types.ModuleType("ai.inference.quantum_image_generator")

    class DummyQuantumImageGenerator:
        def __init__(self, model_name, device, enable_memory_efficient, use_quantum_enhancement, local_files_only):
            self.model_name = model_name
            self.device = device
            self.enable_memory_efficient = enable_memory_efficient
            self.use_quantum_enhancement = use_quantum_enhancement
            self.local_files_only = local_files_only
            self.loaded_model = model_name
            self.quantum_templates = {
                "enhance": "demo",
                "artistic": "demo",
            }

        def generate_image(self, prompt, num_images=1, **kwargs):
            generated_kwargs.update(kwargs)
            return [Image.new("RGB", (32, 32), color="white") for _ in range(num_images)]

        def save_images(self, images, output_dir, prefix):
            os.makedirs(output_dir, exist_ok=True)
            path = Path(output_dir) / f"{prefix}.png"
            images[0].save(path)
            return [str(path)]

        def get_model_info(self):
            return {
                "model_name": self.loaded_model,
                "device": self.device,
                "memory_efficient": self.enable_memory_efficient,
                "quantum_enhanced": self.use_quantum_enhancement,
                "available_styles": list(self.quantum_templates.keys()),
            }

        def clear_memory(self):
            pass

    dummy_module.QuantumImageGenerator = DummyQuantumImageGenerator
    monkeypatch.setitem(sys.modules, "ai.inference.quantum_image_generator", dummy_module)

    from dev.tools import real_image_generator_adapter as adapter_mod

    local_model = tmp_path / "sd"
    local_model.mkdir()
    monkeypatch.setenv("QUANTONIUM_VISION_MODEL_PATH", str(local_model))

    adapter = adapter_mod.RealImageGeneratorAdapter(device="cpu", local_files_only=True)

    assert adapter.model_path == str(local_model)

    image = adapter.generate_image("test prompt", enhancement_style="enhance")
    assert isinstance(image, Image.Image)
    assert generated_kwargs["negative_prompt"]

    out_dir = tmp_path / "out"
    saved_path = adapter.save_image(image, output_dir=str(out_dir), prefix="sample")
    assert os.path.exists(saved_path)

    status = adapter.get_status()
    assert status["generator_type"] == "stable_diffusion"
    assert status["local_files_only"] is True


def test_hf_guided_generator_prefers_real_backend(monkeypatch):
    dummy_module = types.ModuleType("ai.inference.quantum_image_generator")

    class DummyQuantumImageGenerator:
        def __init__(self, *args, **kwargs):
            self.quantum_templates = {"enhance": "demo"}

        def get_model_info(self):
            return {
                "available_styles": list(self.quantum_templates.keys()),
                "model_name": "stub",
                "device": "cpu",
                "memory_efficient": True,
                "quantum_enhanced": True,
            }

        def generate_image(self, *args, **kwargs):
            return [Image.new("RGB", (4, 4), color="black")]

        def save_images(self, images, output_dir, prefix):
            path = Path(output_dir) / f"{prefix}.png"
            os.makedirs(output_dir, exist_ok=True)
            images[0].save(path)
            return [str(path)]

        def clear_memory(self):
            pass

    dummy_module.QuantumImageGenerator = DummyQuantumImageGenerator
    monkeypatch.setitem(sys.modules, "ai.inference.quantum_image_generator", dummy_module)

    from dev.tools import hf_guided_quantum_generator as hf_mod

    class StubAdapter:
        def __init__(self):
            self.called = False

        def generate_image(self, prompt, **kwargs):
            self.called = True
            return Image.new("RGB", (16, 16), color="blue")

        def get_status(self):
            return {
                "generator_type": "stable_diffusion",
                "available_styles": ["enhance"],
                "local_files_only": True,
            }

    stub_instance = StubAdapter()
    monkeypatch.setattr(hf_mod, "REAL_IMAGE_ADAPTER_AVAILABLE", True)
    monkeypatch.setattr(hf_mod, "RealImageGeneratorAdapter", lambda: stub_instance)

    generator = hf_mod.HFGuidedQuantumGenerator()
    assert generator.generator_mode == "stable_diffusion"

    image = generator.generate_image_with_hf_style("test subject", style="stable-diffusion-v1-5")
    assert isinstance(image, Image.Image)
    assert stub_instance.called is True


def test_essential_ai_uses_real_adapter(monkeypatch):
    dummy_module = types.ModuleType("ai.inference.quantum_image_generator")

    class DummyQuantumImageGenerator:
        def __init__(self, *args, **kwargs):
            self.quantum_templates = {"enhance": "demo"}
            self.loaded_model = "stub"
            self.device = "cpu"

        def get_model_info(self):
            return {
                "available_styles": list(self.quantum_templates.keys()),
                "model_name": self.loaded_model,
                "device": self.device,
                "memory_efficient": True,
                "quantum_enhanced": True,
            }

        def generate_image(self, *args, **kwargs):
            return [Image.new("RGB", (4, 4), color="black")]

        def save_images(self, images, output_dir, prefix):
            path = Path(output_dir) / f"{prefix}.png"
            os.makedirs(output_dir, exist_ok=True)
            images[0].save(path)
            return [str(path)]

        def clear_memory(self):
            pass

    dummy_module.QuantumImageGenerator = DummyQuantumImageGenerator
    monkeypatch.setitem(sys.modules, "ai.inference.quantum_image_generator", dummy_module)

    from dev.tools import essential_quantum_ai as eq_mod

    class StubAdapter:
        def __init__(self):
            self.called = False

        def generate_image(self, prompt, **kwargs):
            self.called = True
            return Image.new("RGB", (8, 8), color="green")

        def save_image(self, image, output_dir="results", prefix="test"):
            os.makedirs(output_dir, exist_ok=True)
            path = Path(output_dir) / f"{prefix}.png"
            image.save(path)
            return str(path)

        def get_status(self):
            return {
                "generator_type": "stable_diffusion",
                "local_files_only": True,
                "model_path": "stub",
                "available_styles": ["enhance"],
            }

    stub_instance = StubAdapter()
    monkeypatch.setattr(eq_mod, "REAL_IMAGE_GENERATION_AVAILABLE", True)
    monkeypatch.setattr(eq_mod, "ENCODED_IMAGE_GENERATION_AVAILABLE", False)
    monkeypatch.setattr(eq_mod, "RealImageGeneratorAdapter", lambda: stub_instance)
    monkeypatch.setenv("QUANTONIUM_REAL_MODEL_DISABLED", "1")

    ai = eq_mod.EssentialQuantumAI(enable_image_generation=True)
    assert ai.image_generator is stub_instance

    image = ai.generate_image_only("render a test scene")
    assert isinstance(image, Image.Image)
    assert stub_instance.called is True

    status = ai.get_status()
    assert status["image_generation"]["generator_type"] == "stable_diffusion"
