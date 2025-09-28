#!/usr/bin/env python3
"""
Real Stable Diffusion adapter for QuantoniumOS vision stack.

Provides a thin wrapper around ``QuantumImageGenerator`` that
emphasizes local/real model loading and a simplified interface that
matches the expectations of the legacy quantum-encoded generator.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from ai.inference.quantum_image_generator import QuantumImageGenerator
except ImportError as exc:  # pragma: no cover - handled at call site
    raise RuntimeError(
        "QuantumImageGenerator is required for real vision integration."
    ) from exc

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed"


def _resolve_local_model_path(model_id: str) -> str:
    """Resolve a model identifier to a local filesystem path when available."""
    # Environment variable override takes precedence
    env_path = os.getenv("QUANTONIUM_VISION_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        logger.info("📁 Using vision model from QUANTONIUM_VISION_MODEL_PATH=%s", env_path)
        return env_path

    formatted = model_id.replace("/", "--")
    local_root = os.path.join(REPO_ROOT, "hf_models", f"models--{formatted}")
    if os.path.isdir(local_root):
        snapshots_dir = os.path.join(local_root, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(
                glob.glob(os.path.join(snapshots_dir, "*")),
                key=os.path.getmtime,
                reverse=True,
            )
            if snapshots:
                logger.info("📁 Using local HF snapshot for %s: %s", model_id, snapshots[0])
                return snapshots[0]
        logger.info("📁 Using local HF cache directory for %s: %s", model_id, local_root)
        return local_root

    logger.info("🌐 Falling back to model identifier %s", model_id)
    return model_id


class RealImageGeneratorAdapter:
    """Adapter that exposes a simplified interface for Stable Diffusion."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        local_files_only: bool = True,
        negative_prompt: Optional[str] = None,
    ) -> None:
        self.base_model_id = model_name or os.getenv("QUANTONIUM_VISION_MODEL", DEFAULT_MODEL_ID)
        self.device = device or os.getenv("QUANTONIUM_VISION_DEVICE", "auto")
        self.local_files_only = local_files_only
        self.negative_prompt = negative_prompt or os.getenv(
            "QUANTONIUM_VISION_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT
        )
        self.model_path = _resolve_local_model_path(self.base_model_id)
        memory_efficient = os.getenv("QUANTONIUM_VISION_MEMORY_EFFICIENT", "1") != "0"

        logger.info(
            "⚙️ Initializing real vision generator: model=%s device=%s local_only=%s",
            self.model_path,
            self.device,
            self.local_files_only,
        )

        self._generator = QuantumImageGenerator(
            model_name=self.model_path,
            device=self.device,
            enable_memory_efficient=memory_efficient,
            use_quantum_enhancement=True,
            local_files_only=self.local_files_only,
        )

    # ---------------------------------------------------------------------
    # Public API mirroring legacy quantum generator behaviour
    # ---------------------------------------------------------------------
    def generate_image(self, prompt: str, **kwargs: Any) -> Optional[Image.Image]:
        """Generate a single image and return the first PIL Image."""
        kwargs.setdefault("negative_prompt", self.negative_prompt)
        num_images = kwargs.pop("num_images", 1)
        kwargs.setdefault("num_inference_steps", 20)
        kwargs.setdefault("guidance_scale", 7.5)

        images = self._generator.generate_image(prompt, num_images=num_images, **kwargs)
        if not images:
            logger.warning("❌ Stable Diffusion returned no images for prompt: %s", prompt)
            return None
        return images[0]

    def generate_images(self, prompt: str, **kwargs: Any) -> List[Image.Image]:
        """Generate and return all images from the underlying pipeline."""
        kwargs.setdefault("negative_prompt", self.negative_prompt)
        return self._generator.generate_image(prompt, **kwargs)

    def save_image(
        self,
        image: Image.Image,
        output_dir: str = "results/generated_images",
        prefix: str = "stable_diffusion",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        paths = self._generator.save_images([image], output_dir=output_dir, prefix=prefix)
        return paths[0] if paths else ""

    def get_status(self) -> Dict[str, Any]:
        info = self._generator.get_model_info()
        info.update(
            {
                "generator_type": "stable_diffusion",
                "model_path": self.model_path,
                "local_files_only": self.local_files_only,
                "parameter_sets": None,
                "total_encoded_features": 0,
                "feature_types": info.get("available_styles", []),
                "quantum_encoding_enabled": False,
            }
        )
        return info

    def list_available_styles(self) -> List[str]:
        return list(self._generator.quantum_templates.keys())

    def clear_memory(self) -> None:
        self._generator.clear_memory()

    # Convenience helpers -------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._generator.loaded_model or self.base_model_id

    @property
    def device_map(self) -> str:
        return self._generator.device


__all__ = ["RealImageGeneratorAdapter", "_resolve_local_model_path"]
