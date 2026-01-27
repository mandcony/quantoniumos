#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
HF-Guided Quantum Image Generator
===============================
Enhances quantum-encoded generation with Hugging Face model style guidance
No large downloads required - uses metadata and style patterns
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image
import time
import hashlib

# Add paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "dev", "tools"))

from quantum_encoded_image_generator import QuantumEncodedImageGenerator, EncodedImageParameters

try:
    from real_image_generator_adapter import RealImageGeneratorAdapter
    REAL_IMAGE_ADAPTER_AVAILABLE = True
except ImportError as exc:
    REAL_IMAGE_ADAPTER_AVAILABLE = False
    print(f"⚠️ Real image adapter unavailable for HF-guided generator: {exc}")

class HFGuidedQuantumGenerator(QuantumEncodedImageGenerator):
    """Quantum generator enhanced with HF model style guidance"""
    
    def __init__(self):
        super().__init__(use_quantum_encoding=False)
        self.hf_style_guides = self._load_builtin_style_guides()
        self.hf_encoded_dir = os.path.join(BASE_DIR, "data", "weights", "hf_encoded")
        self.generator_mode = "stable_diffusion" if REAL_IMAGE_ADAPTER_AVAILABLE else "quantum_encoded"
        self.real_generator: Optional[RealImageGeneratorAdapter] = None
        
        # Load any downloaded HF guidance
        self._load_hf_guidance()
        
        if REAL_IMAGE_ADAPTER_AVAILABLE:
            try:
                self.real_generator = RealImageGeneratorAdapter()
                print("✅ HF-guided generator using Stable Diffusion backend")
            except Exception as exc:
                print(f"⚠️ Failed to initialize Stable Diffusion backend: {exc}")
                self.real_generator = None

        if self.real_generator is None:
            self.use_quantum_encoding = True
            try:
                self.encoded_params = EncodedImageParameters()
                self.generator_mode = "quantum_encoded"
                print("🎛️ Falling back to quantum-encoded image generation")
            except Exception as exc:
                print(f"❌ Could not initialize quantum-encoded fallback: {exc}")
                self.generator_mode = "disabled"
                self.encoded_params = None

        print("🎨 HF-Guided Quantum Generator initialized")
        print(f"   Style guides: {len(self.hf_style_guides)}")
    
    def _load_builtin_style_guides(self) -> Dict:
        """Built-in style guidance based on popular HF models"""
        return {
            "stable-diffusion-v1-5": {
                "prompt_enhancers": [
                    "high quality, detailed",
                    "professional photography", 
                    "8k resolution, sharp focus",
                    "masterpiece"
                ],
                "style_weights": {
                    "photorealistic": 0.8,
                    "detailed": 0.9,
                    "sharp": 0.7,
                    "professional": 0.8
                },
                "color_bias": "natural",
                "preferred_subjects": ["portraits", "landscapes", "objects", "scenes"]
            },
            "stable-diffusion-2-1": {
                "prompt_enhancers": [
                    "masterpiece, best quality",
                    "highly detailed, ultra realistic",
                    "professional lighting",
                    "award winning photography"
                ],
                "style_weights": {
                    "realistic": 0.9,
                    "detailed": 0.95,
                    "lighting": 0.8,
                    "quality": 0.9
                },
                "color_bias": "enhanced",
                "preferred_subjects": ["portraits", "art", "photography", "detailed scenes"]
            },
            "dreamlike-diffusion": {
                "prompt_enhancers": [
                    "dreamlike art style",
                    "fantasy, surreal",
                    "artistic, painterly",
                    "vibrant colors"
                ],
                "style_weights": {
                    "artistic": 0.9,
                    "fantasy": 0.8,
                    "vibrant": 0.85,
                    "painterly": 0.9
                },
                "color_bias": "vibrant",
                "preferred_subjects": ["fantasy", "abstract", "artistic", "surreal"]
            },
            "analog-diffusion": {
                "prompt_enhancers": [
                    "analog photo style",
                    "film grain, vintage",
                    "retro aesthetic",
                    "nostalgic atmosphere"
                ],
                "style_weights": {
                    "vintage": 0.9,
                    "analog": 0.95,
                    "nostalgic": 0.8,
                    "film": 0.85
                },
                "color_bias": "warm",
                "preferred_subjects": ["portraits", "street", "vintage", "lifestyle"]
            }
        }
    
    def _load_hf_guidance(self):
        """Load any downloaded HF model guidance"""
        if os.path.exists(self.hf_encoded_dir):
            for filename in os.listdir(self.hf_encoded_dir):
                if filename.endswith('.json') and filename.startswith('hf_encoded_'):
                    try:
                        filepath = os.path.join(self.hf_encoded_dir, filename)
                        with open(filepath, 'r') as f:
                            hf_data = json.load(f)
                        
                        model_id = hf_data['model_info']['model_id']
                        if 'generation_capabilities' in hf_data:
                            # Add downloaded guidance
                            guidance = hf_data['generation_capabilities']
                            style_key = model_id.replace('/', '-').replace('_', '-')
                            
                            self.hf_style_guides[style_key] = {
                                "prompt_enhancers": guidance.get('recommended_prompts', []),
                                "style_weights": guidance.get('style_hints', {}),
                                "source": "downloaded_hf"
                            }
                            
                            print(f"   ✅ Loaded HF guidance: {model_id}")
                            
                    except Exception as e:
                        print(f"   ⚠️ Failed to load {filename}: {e}")
    
    def generate_image_with_hf_style(self, prompt: str, style: str = "stable-diffusion-v1-5", **kwargs) -> Image.Image:
        """Generate image with HF model style guidance"""
        
        if style not in self.hf_style_guides:
            print(f"⚠️ Style '{style}' not found, using quantum-encoded generation")
            return self.generate_image(prompt, **kwargs)
        
        style_guide = self.hf_style_guides[style]
        
        # Enhance prompt with style guidance
        enhanced_prompt = self._enhance_prompt_with_style(prompt, style_guide)
        
        # Apply style-specific quantum parameters
        enhanced_kwargs = self._apply_style_parameters(kwargs, style_guide, style)
        style_guidance = enhanced_kwargs.pop('style_guidance', {})
        style_name = enhanced_kwargs.pop('style_name', style)
        
        print(f"🎨 Generating with {style} style guidance...")
        print(f"   Original: {prompt}")
        print(f"   Enhanced: {enhanced_prompt}")
        
        # Generate with real Stable Diffusion backend when available
        if self.real_generator:
            real_kwargs = self._prepare_real_generation_kwargs(style, style_guidance, enhanced_kwargs)
            image = self.real_generator.generate_image(
                enhanced_prompt,
                **real_kwargs
            )
            if image is None:
                raise RuntimeError("Stable Diffusion generation returned no image")
            return image

        # Fallback to encoded parameters
        for key in list(enhanced_kwargs.keys()):
            if key in ['color_enhancement', 'color_temperature', 'color_saturation']:
                enhanced_kwargs.pop(key)

        image = super().generate_image_from_encoded_params(enhanced_prompt, **enhanced_kwargs)
        if style_guidance:
            image = self._apply_style_post_processing(image, style_guidance, style_name)
        return image
    
    def _enhance_prompt_with_style(self, prompt: str, style_guide: Dict) -> str:
        """Enhance prompt with style-specific guidance"""
        enhanced = prompt
        
        # Add style enhancers if not already present
        for enhancer in style_guide.get('prompt_enhancers', []):
            if not any(word in enhanced.lower() for word in enhancer.lower().split()):
                enhanced += f", {enhancer}"
        
        return enhanced
    
    def _apply_style_parameters(self, kwargs: Dict, style_guide: Dict, style_name: str) -> Dict:
        """Apply style-specific generation parameters"""
        enhanced_kwargs = kwargs.copy()
        
        # Apply color bias
        color_bias = style_guide.get('color_bias', 'natural')
        if color_bias == 'vibrant':
            enhanced_kwargs['color_enhancement'] = 1.2
        elif color_bias == 'warm':
            enhanced_kwargs['color_temperature'] = 'warm'
        elif color_bias == 'enhanced':
            enhanced_kwargs['color_saturation'] = 1.1
        
        # Apply style weights to quantum generation
        style_weights = style_guide.get('style_weights', {})
        enhanced_kwargs['style_guidance'] = style_weights
        enhanced_kwargs['style_name'] = style_name
        
        return enhanced_kwargs

    def _prepare_real_generation_kwargs(self, style: str, style_guidance: Dict, kwargs: Dict) -> Dict:
        """Prepare kwargs for the real Stable Diffusion adapter."""
        real_kwargs = kwargs.copy()

        # Remove keys not understood by the Stable Diffusion pipeline
        for key in ['color_enhancement', 'color_temperature', 'color_saturation']:
            real_kwargs.pop(key, None)

        # Map style to enhancement preset
        real_kwargs.setdefault('enhancement_style', self._map_style_to_enhancement(style))
        real_kwargs.setdefault('num_images', 1)

        # Adjust guidance scale based on style hints
        if style_guidance:
            detail_weight = style_guidance.get('detailed') or style_guidance.get('detail', 0.0)
            realism_weight = style_guidance.get('realistic') or style_guidance.get('photorealistic', 0.0)

            base_guidance = 7.5 + 0.5 * realism_weight + 0.4 * detail_weight
            real_kwargs.setdefault('guidance_scale', round(base_guidance, 2))

            if detail_weight and detail_weight > 0.8:
                real_kwargs.setdefault('num_inference_steps', 32)

        return real_kwargs

    @staticmethod
    def _map_style_to_enhancement(style: str) -> str:
        mapping = {
            'stable-diffusion-v1-5': 'enhance',
            'stable-diffusion-2-1': 'enhance',
            'dreamlike-diffusion': 'artistic',
            'analog-diffusion': 'enhance',
        }
        return mapping.get(style, 'enhance')
    
    def generate_image_from_encoded_params(self, prompt: str, **kwargs) -> Image.Image:
        """Enhanced quantum generation with style guidance"""
        
        # Get style guidance if provided
        style_guidance = kwargs.pop('style_guidance', {})
        style_name = kwargs.pop('style_name', 'quantum')
        
        # Remove style-specific kwargs that the base class doesn't understand
        for key in list(kwargs.keys()):
            if key in ['color_enhancement', 'color_temperature', 'color_saturation']:
                kwargs.pop(key)
        
        # Generate base image using quantum encoding
        image = super().generate_image_from_encoded_params(prompt, **kwargs)
        
        # Apply style-specific post-processing
        if style_guidance:
            image = self._apply_style_post_processing(image, style_guidance, style_name)
        
        return image
    
    def _apply_style_post_processing(self, image: Image.Image, style_guidance: Dict, style_name: str) -> Image.Image:
        """Apply style-specific post-processing to the generated image"""
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Apply style-specific adjustments
        if 'vibrant' in style_guidance and style_guidance['vibrant'] > 0.8:
            # Enhance saturation for vibrant styles
            img_array = self._enhance_saturation(img_array, 1.2)
        
        if 'vintage' in style_guidance and style_guidance['vintage'] > 0.8:
            # Add vintage effect
            img_array = self._apply_vintage_effect(img_array)
        
        if 'detailed' in style_guidance and style_guidance['detailed'] > 0.8:
            # Enhance detail perception
            img_array = self._enhance_details(img_array)
        
        # Convert back to PIL Image
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def _enhance_saturation(self, img_array: np.ndarray, factor: float) -> np.ndarray:
        """Enhance color saturation"""
        # Simple saturation enhancement
        mean = img_array.mean(axis=2, keepdims=True)
        enhanced = mean + (img_array - mean) * factor
        return np.clip(enhanced, 0, 255)
    
    def _apply_vintage_effect(self, img_array: np.ndarray) -> np.ndarray:
        """Apply vintage film effect"""
        # Warm color cast and slight grain
        vintage = img_array.copy()
        vintage[:, :, 0] = np.clip(vintage[:, :, 0] * 1.1, 0, 255)  # Enhance reds
        vintage[:, :, 2] = np.clip(vintage[:, :, 2] * 0.9, 0, 255)  # Reduce blues
        return vintage
    
    def _enhance_details(self, img_array: np.ndarray) -> np.ndarray:
        """Enhance perceived detail"""
        try:
            # Simple sharpening effect using scipy if available
            from scipy import ndimage
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            if len(img_array.shape) == 3:
                enhanced = np.zeros_like(img_array)
                for i in range(img_array.shape[2]):
                    enhanced[:, :, i] = ndimage.convolve(img_array[:, :, i], kernel)
                return np.clip(enhanced, 0, 255)
        except ImportError:
            # Fallback if scipy not available
            print("   ⚠️ Scipy not available for detail enhancement")
        return img_array
    
    def list_available_styles(self) -> List[str]:
        """List all available HF styles"""
        return list(self.hf_style_guides.keys())
    
    def get_style_info(self, style: str) -> Dict:
        """Get information about a specific style"""
        return self.hf_style_guides.get(style, {})

def main():
    """Test HF-guided quantum generation"""
    print("🎯 HF-Guided Quantum Generator Test")
    print("=" * 50)
    
    generator = HFGuidedQuantumGenerator()
    
    print(f"\\n📋 Available styles: {', '.join(generator.list_available_styles())}")
    
    # Test different styles
    test_prompts = [
        ("a beautiful landscape", "stable-diffusion-v1-5"),
        ("fantasy dragon", "dreamlike-diffusion"), 
        ("vintage portrait", "analog-diffusion"),
        ("quantum cat in space", "stable-diffusion-2-1")
    ]
    
    for prompt, style in test_prompts:
        print(f"\\n🎨 Testing: '{prompt}' with {style} style")
        try:
            image = generator.generate_image_with_hf_style(prompt, style=style)
            
            # Save test image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hf_guided_{style}_{timestamp}.png"
            os.makedirs("results/generated_images", exist_ok=True)
            image_path = os.path.join("results", "generated_images", filename)
            image.save(image_path)
            
            print(f"✅ Generated: {image_path}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()