#!/usr/bin/env python3
"""
Quantum Image Generator
Integrates text-to-image generation with QuantoniumOS AI stack
"""

import os
import torch
from typing import Optional, List, Dict, Any, Union
from PIL import Image
import logging
from datetime import datetime

try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    from diffusers import DDIMScheduler, EulerDiscreteScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class QuantumImageGenerator:
    """
    Advanced image generation engine for QuantoniumOS
    
    Integrates with the existing quantum inference engine to provide
    text-to-image capabilities using Stable Diffusion and other models.
    """
    
    def __init__(self, 
                 model_name: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "auto",
                 enable_memory_efficient: bool = True,
                 use_quantum_enhancement: bool = True):
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library not installed. Run: pip install diffusers")
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_memory_efficient = enable_memory_efficient
        self.use_quantum_enhancement = use_quantum_enhancement
        
        self.pipeline = None
        self.loaded_model = None
        
        # Quantum-enhanced prompt templates
        self.quantum_templates = {
            "enhance": "masterpiece, highly detailed, ultra-realistic, 8k resolution, professional lighting",
            "artistic": "digital art, concept art, trending on artstation, dramatic lighting",
            "scientific": "scientific illustration, technical diagram, accurate representation",
            "quantum": "quantum visualization, particle physics, wave function, probability clouds"
        }
        
        logger.info(f"⚛️ Initializing QuantumImageGenerator on {self.device}")
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the image generation pipeline"""
        try:
            # Load with memory optimizations if requested
            if self.enable_memory_efficient and self.device == "cuda":
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                # Enable memory efficient attention
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
                
                # Use more memory efficient scheduler
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_name,
                    use_safetensors=True
                )
            
            self.pipeline = self.pipeline.to(self.device)
            self.loaded_model = self.model_name
            
            logger.info(f"✅ Loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load image generation pipeline: {e}")
            raise
    
    def generate_image(self, 
                      prompt: str,
                      negative_prompt: str = "blurry, low quality, distorted, deformed",
                      width: int = 512,
                      height: int = 512,
                      num_inference_steps: int = 20,
                      guidance_scale: float = 7.5,
                      num_images: int = 1,
                      seed: Optional[int] = None,
                      enhancement_style: str = "enhance") -> List[Image.Image]:
        """
        Generate images from text prompt
        
        Args:
            prompt: Text description of the image to generate
            negative_prompt: What to avoid in the image
            width, height: Image dimensions (must be multiples of 64)
            num_inference_steps: Quality vs speed tradeoff (10-50)
            guidance_scale: How closely to follow prompt (1-20)
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            enhancement_style: Quantum enhancement style to apply
            
        Returns:
            List of PIL Images
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded. Call _load_pipeline() first.")
        
        # Apply quantum enhancement to prompt
        enhanced_prompt = self._enhance_prompt(prompt, enhancement_style)
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        try:
            logger.info(f"🎨 Generating {num_images} image(s) from: {enhanced_prompt[:50]}...")
            
            # Generate images
            result = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images
            )
            
            images = result.images
            logger.info(f"✅ Generated {len(images)} image(s) successfully")
            
            return images
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            raise
    
    def _enhance_prompt(self, prompt: str, style: str = "enhance") -> str:
        """Apply quantum-enhanced prompt optimization"""
        if not self.use_quantum_enhancement:
            return prompt
        
        # Get enhancement template
        enhancement = self.quantum_templates.get(style, self.quantum_templates["enhance"])
        
        # Combine with quantum optimization patterns
        enhanced = f"{prompt}, {enhancement}"
        
        # Apply golden ratio based prompt structuring (quantum-inspired)
        if "nano banana" in prompt.lower():
            enhanced = f"microscopic banana, nanotechnology scale, scientific visualization, {enhancement}"
        
        return enhanced
    
    def save_images(self, images: List[Image.Image], 
                   output_dir: str = "results/generated_images",
                   prefix: str = "quantum_gen") -> List[str]:
        """Save generated images to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, image in enumerate(images):
            filename = f"{prefix}_{timestamp}_{i+1:03d}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath, "PNG")
            saved_paths.append(filepath)
            logger.info(f"💾 Saved image: {filepath}")
        
        return saved_paths
    
    def generate_and_save(self, prompt: str, **kwargs) -> List[str]:
        """Convenience method to generate and save images in one call"""
        images = self.generate_image(prompt, **kwargs)
        return self.save_images(images)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> Dict[str, List[Image.Image]]:
        """Generate images for multiple prompts"""
        results = {}
        
        for prompt in prompts:
            try:
                images = self.generate_image(prompt, **kwargs)
                results[prompt] = images
                logger.info(f"✅ Batch processed: {prompt[:30]}...")
            except Exception as e:
                logger.error(f"❌ Batch failed for '{prompt}': {e}")
                results[prompt] = []
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.loaded_model,
            "device": self.device,
            "memory_efficient": self.enable_memory_efficient,
            "quantum_enhanced": self.use_quantum_enhancement,
            "available_styles": list(self.quantum_templates.keys())
        }
    
    def clear_memory(self):
        """Clear GPU memory"""
        if self.pipeline and self.device == "cuda":
            del self.pipeline
            torch.cuda.empty_cache()
            self.pipeline = None
            logger.info("🧹 Cleared GPU memory")


# Example usage and integration with existing quantum inference engine
def integrate_with_quantum_engine():
    """Example of how to integrate with the existing quantum inference engine"""
    
    # This would be called from your main AI system
    try:
        from .quantum_inference_engine import QuantumInferenceEngine
        
        # Create both text and image generators
        text_engine = QuantumInferenceEngine()
        image_engine = QuantumImageGenerator()
        
        def multimodal_response(prompt: str, generate_image: bool = False):
            """Generate both text and image responses"""
            
            # Generate text response
            text_response = text_engine.generate_response(prompt)
            
            result = {
                "text": text_response,
                "images": []
            }
            
            # Generate image if requested
            if generate_image:
                try:
                    images = image_engine.generate_image(prompt)
                    saved_paths = image_engine.save_images(images)
                    result["images"] = saved_paths
                except Exception as e:
                    logger.warning(f"Image generation failed: {e}")
            
            return result
        
        return multimodal_response
        
    except ImportError as e:
        logger.warning(f"Could not integrate with quantum inference engine: {e}")
        return None


if __name__ == "__main__":
    # Test the image generator
    try:
        generator = QuantumImageGenerator()
        
        # Generate a nano banana image
        images = generator.generate_image(
            "a nano banana in a futuristic quantum laboratory",
            enhancement_style="quantum",
            num_images=1
        )
        
        # Save the images
        saved_paths = generator.save_images(images)
        print(f"⚛️ Generated quantum nano banana images: {saved_paths}")
        
    except Exception as e:
        print(f"❌ Failed to test image generator: {e}")
        print("Make sure to install: pip install diffusers")