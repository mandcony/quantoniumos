#!/usr/bin/env python3
"""
QuantoniumOS Hybrid Image Generator
Combines quantum-encoded parameters with traditional diffusion models
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np

# Import quantum-encoded generator
try:
    from quantum_encoded_image_generator import QuantumEncodedImageGenerator
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

# Import traditional diffusion models
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    import torch
    DIFFUSION_AVAILABLE = True
    print("✅ Diffusion models available")
except ImportError as e:
    DIFFUSION_AVAILABLE = False
    print(f"⚠️ Diffusion models not available: {e}")

class HybridImageGenerator:
    """
    Hybrid image generator supporting both:
    1. Quantum-encoded parameter streaming (your architecture)
    2. Traditional diffusion models (Stable Diffusion)
    """
    
    def __init__(self):
        self.quantum_generator = None
        self.diffusion_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() and DIFFUSION_AVAILABLE else "cpu"
        
        # Initialize quantum-encoded generator
        if QUANTUM_AVAILABLE:
            try:
                self.quantum_generator = QuantumEncodedImageGenerator()
                print("✅ Quantum-encoded image generator loaded")
            except Exception as e:
                print(f"⚠️ Quantum generator failed: {e}")
        
        # Initialize diffusion pipeline (lazy loading)
        self.diffusion_model = None
        
    def load_diffusion_model(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Load a Stable Diffusion model (on-demand)"""
        if not DIFFUSION_AVAILABLE:
            print("❌ Diffusion models not available - install diffusers and torch")
            return False
            
        try:
            print(f"🔄 Loading diffusion model: {model_name}")
            
            # Load with optimizations
            self.diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            
            # Optimize for speed
            self.diffusion_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.diffusion_pipeline.scheduler.config
            )
            
            # Move to device
            self.diffusion_pipeline = self.diffusion_pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.diffusion_pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    self.diffusion_pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ XFormers memory optimization enabled")
                except:
                    pass
            
            self.diffusion_model = model_name
            print(f"✅ Diffusion model loaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load diffusion model: {e}")
            return False
    
    def generate_quantum_encoded(self, prompt: str, **kwargs) -> Optional[Image.Image]:
        """Generate image using quantum-encoded parameters"""
        if not self.quantum_generator:
            print("❌ Quantum generator not available")
            return None
            
        try:
            return self.quantum_generator.generate_image_from_encoded_params(prompt, **kwargs)
        except Exception as e:
            print(f"❌ Quantum generation failed: {e}")
            return None
    
    def generate_diffusion(self, prompt: str, **kwargs) -> Optional[Image.Image]:
        """Generate image using Stable Diffusion"""
        if not self.diffusion_pipeline:
            print("❌ Diffusion pipeline not loaded. Call load_diffusion_model() first.")
            return None
            
        try:
            # Default parameters
            params = {
                "prompt": prompt,
                "num_inference_steps": kwargs.get("steps", 20),
                "guidance_scale": kwargs.get("guidance", 7.5),
                "width": kwargs.get("width", 512),
                "height": kwargs.get("height", 512),
            }
            
            print(f"🎨 Generating with Stable Diffusion: '{prompt[:50]}...'")
            start_time = time.time()
            
            with torch.autocast(self.device):
                result = self.diffusion_pipeline(**params)
                
            generation_time = time.time() - start_time
            print(f"✅ Diffusion generation completed in {generation_time:.1f}s")
            
            return result.images[0]
            
        except Exception as e:
            print(f"❌ Diffusion generation failed: {e}")
            return None
    
    def generate_image(self, prompt: str, method: str = "auto", **kwargs) -> Optional[Image.Image]:
        """
        Generate image using specified method
        
        Args:
            prompt: Text description
            method: "quantum", "diffusion", or "auto"
            **kwargs: Additional parameters
        """
        
        if method == "quantum" or (method == "auto" and self.quantum_generator):
            print("🔬 Using quantum-encoded generation")
            return self.generate_quantum_encoded(prompt, **kwargs)
            
        elif method == "diffusion" or method == "auto":
            # Auto-load a diffusion model if not loaded
            if not self.diffusion_pipeline and method == "auto":
                self.load_diffusion_model()
            
            if self.diffusion_pipeline:
                print("🎨 Using Stable Diffusion generation")
                return self.generate_diffusion(prompt, **kwargs)
        
        print("❌ No generation method available")
        return None
    
    def compare_methods(self, prompt: str, save_dir: str = "results/comparison"):
        """Generate the same prompt with both methods for comparison"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {}
        
        # Quantum method
        if self.quantum_generator:
            print("🔬 Generating with quantum-encoded method...")
            start_time = time.time()
            quantum_img = self.generate_quantum_encoded(prompt)
            quantum_time = time.time() - start_time
            
            if quantum_img:
                quantum_path = os.path.join(save_dir, f"quantum_{timestamp}.png")
                quantum_img.save(quantum_path)
                results["quantum"] = {
                    "path": quantum_path,
                    "time": quantum_time,
                    "method": "Quantum-Encoded Parameters"
                }
                print(f"✅ Quantum: {quantum_time:.1f}s -> {quantum_path}")
        
        # Diffusion method
        if not self.diffusion_pipeline:
            self.load_diffusion_model()
            
        if self.diffusion_pipeline:
            print("🎨 Generating with Stable Diffusion...")
            start_time = time.time()
            diffusion_img = self.generate_diffusion(prompt)
            diffusion_time = time.time() - start_time
            
            if diffusion_img:
                diffusion_path = os.path.join(save_dir, f"diffusion_{timestamp}.png")
                diffusion_img.save(diffusion_path)
                results["diffusion"] = {
                    "path": diffusion_path,
                    "time": diffusion_time,
                    "method": "Stable Diffusion"
                }
                print(f"✅ Diffusion: {diffusion_time:.1f}s -> {diffusion_path}")
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of available generation methods"""
        return {
            "quantum_available": self.quantum_generator is not None,
            "diffusion_available": self.diffusion_pipeline is not None,
            "diffusion_model": self.diffusion_model,
            "device": self.device,
            "cuda_available": torch.cuda.is_available() if DIFFUSION_AVAILABLE else False,
        }

def main():
    """Test hybrid image generation"""
    print("🚀 QuantoniumOS Hybrid Image Generator Test")
    print("=" * 50)
    
    generator = HybridImageGenerator()
    
    # Show status
    status = generator.get_status()
    print(f"📊 Status:")
    print(f"   • Quantum: {'✅' if status['quantum_available'] else '❌'}")
    print(f"   • Diffusion: {'✅' if status['diffusion_available'] else '❌'}")
    print(f"   • Device: {status['device']}")
    print(f"   • CUDA: {'✅' if status['cuda_available'] else '❌'}")
    
    # Test prompts
    test_prompts = [
        "a quantum computer in a laboratory",
        "golden ratio spiral in space",
        "nano banana floating in quantum field"
    ]
    
    for prompt in test_prompts:
        print(f"\n🎨 Testing: '{prompt}'")
        
        # Try quantum method
        img = generator.generate_image(prompt, method="quantum")
        if img:
            path = f"results/hybrid_test_quantum_{int(time.time())}.png"
            os.makedirs("results", exist_ok=True)
            img.save(path)
            print(f"✅ Quantum result: {path}")
        
        # Try diffusion method if you want to test it
        # Uncomment the following lines to test Stable Diffusion:
        # img = generator.generate_image(prompt, method="diffusion")
        # if img:
        #     path = f"results/hybrid_test_diffusion_{int(time.time())}.png"
        #     img.save(path)
        #     print(f"✅ Diffusion result: {path}")

if __name__ == "__main__":
    main()