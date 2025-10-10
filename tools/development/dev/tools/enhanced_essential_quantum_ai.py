#!/usr/bin/env python3
"""
Enhanced Essential Quantum AI with Hugging Face Integration
==========================================================
Combines quantum-encoded parameters with photorealistic HF models
Uses the same streaming architecture for both approaches
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image
import time

# Add paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "dev", "tools"))

from essential_quantum_ai import EssentialQuantumAI
from quantum_encoded_image_generator import QuantumEncodedImageGenerator

class EnhancedEssentialQuantumAI(EssentialQuantumAI):
    """Enhanced version with HF photorealistic generation"""
    
    def __init__(self, enable_image_generation=True, enable_hf_models=True):
        # Initialize base Essential AI
        super().__init__(enable_image_generation=enable_image_generation)
        
        self.enable_hf_models = enable_hf_models
        self.hf_models = {}
        self.hf_pipelines = {}
        
        if enable_hf_models:
            self._load_hf_models()
    
    def _load_hf_models(self):
        """Load available HF encoded models"""
        hf_encoded_dir = os.path.join(BASE_DIR, "data", "weights", "hf_encoded")
        
        if not os.path.exists(hf_encoded_dir):
            print("📁 No HF encoded models found. Run hf_streaming_encoder.py to install.")
            return
        
        print("🔍 Loading HF encoded models...")
        
        for filename in os.listdir(hf_encoded_dir):
            if filename.endswith('.json') and filename.startswith('hf_encoded_'):
                filepath = os.path.join(hf_encoded_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        model_data = json.load(f)
                    
                    model_id = model_data['model_info']['model_id']
                    self.hf_models[model_id] = model_data
                    
                    print(f"✅ Loaded HF model: {model_id}")
                    print(f"   Parameters: {model_data['total_parameters']:,}")
                    print(f"   States: {model_data['quantum_encoding']['state_count']}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to load {filename}: {e}")
        
        if self.hf_models:
            print(f"🎯 {len(self.hf_models)} HF models ready for streaming generation")
        
        # Try to load actual pipelines for photorealistic generation
        self._load_hf_pipelines()
    
    def _load_hf_pipelines(self):
        """Load actual HF pipelines for photorealistic generation"""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            # Check for available models
            hf_cache_dir = os.path.join(BASE_DIR, "hf_models")
            
            for model_id in self.hf_models.keys():
                try:
                    print(f"🔄 Loading pipeline: {model_id}")
                    
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        cache_dir=hf_cache_dir,
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                        local_files_only=True  # Use only downloaded files
                    )
                    
                    self.hf_pipelines[model_id] = pipeline
                    print(f"✅ Pipeline loaded: {model_id}")
                    
                except Exception as e:
                    print(f"⚠️ Pipeline load failed for {model_id}: {e}")
                    
        except ImportError:
            print("⚠️ Diffusers not available for photorealistic generation")
    
    def generate_image_enhanced(self, prompt: str, method: str = "auto", width: int = 512, height: int = 512, **kwargs) -> Image.Image:
        """
        Enhanced image generation with multiple methods:
        - 'quantum': Your quantum-encoded parameters (fast, your architecture)
        - 'photorealistic': HF Stable Diffusion (slower, photorealistic)
        - 'auto': Choose best method automatically
        """
        
        if method == "auto":
            # Auto-select based on prompt and available models
            if self.hf_pipelines and any(word in prompt.lower() for word in ['photo', 'realistic', 'portrait', 'landscape']):
                method = "photorealistic"
            else:
                method = "quantum"
        
        if method == "photorealistic" and self.hf_pipelines:
            return self._generate_photorealistic(prompt, width, height, **kwargs)
        else:
            # Use quantum-encoded generation
            if hasattr(self, 'image_generator'):
                return self.image_generator.generate_image(prompt, width=width, height=height)
            else:
                # Fallback to quantum encoded
                generator = QuantumEncodedImageGenerator()
                return generator.generate_image(prompt, width=width, height=height)
    
    def _generate_photorealistic(self, prompt: str, width: int, height: int, **kwargs) -> Image.Image:
        """Generate photorealistic image using HF Stable Diffusion"""
        
        # Select best available model
        preferred_models = [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1", 
            "CompVis/stable-diffusion-v1-4"
        ]
        
        pipeline = None
        model_used = None
        
        for model_id in preferred_models:
            if model_id in self.hf_pipelines:
                pipeline = self.hf_pipelines[model_id]
                model_used = model_id
                break
        
        if not pipeline:
            # Use any available pipeline
            if self.hf_pipelines:
                model_used = list(self.hf_pipelines.keys())[0]
                pipeline = self.hf_pipelines[model_used]
        
        if not pipeline:
            raise Exception("No HF pipelines available for photorealistic generation")
        
        print(f"🎨 Generating photorealistic image with {model_used}...")
        print(f"   Prompt: {prompt}")
        print(f"   Size: {width}x{height}")
        
        try:
            # Generate image
            start_time = time.time()
            
            image = pipeline(
                prompt,
                width=width,
                height=height,
                num_inference_steps=kwargs.get('steps', 20),
                guidance_scale=kwargs.get('guidance', 7.5)
            ).images[0]
            
            generation_time = time.time() - start_time
            print(f"✅ Photorealistic generation completed in {generation_time:.1f}s")
            
            return image
            
        except Exception as e:
            print(f"❌ Photorealistic generation failed: {e}")
            # Fallback to quantum encoding
            print("🔄 Falling back to quantum-encoded generation...")
            return self.generate_image_enhanced(prompt, method="quantum", width=width, height=height)
    
    def process_message_enhanced(self, message: str, image_method: str = "auto") -> Any:
        """Enhanced message processing with image method selection"""
        
        # Check for image generation with method specification
        if any(keyword in message.lower() for keyword in ['image:', 'generate image', 'create image', 'draw', 'picture', 'visualize']):
            
            # Extract method preference from message
            if 'photorealistic' in message.lower() or 'realistic' in message.lower() or 'photo' in message.lower():
                image_method = "photorealistic"
            elif 'quantum' in message.lower() or 'encoded' in message.lower():
                image_method = "quantum"
            
            # Extract image prompt
            image_prompt = message
            for prefix in ['image:', 'generate image', 'create image', 'draw', 'picture of', 'visualize']:
                image_prompt = image_prompt.lower().replace(prefix, '').strip()
            
            # Remove method specifiers from prompt
            for method_word in ['photorealistic', 'realistic', 'quantum', 'encoded']:
                image_prompt = image_prompt.replace(method_word, '').strip()
            
            try:
                # Generate image
                print(f"🎨 Image request detected: '{image_prompt}' (method: {image_method})")
                
                image = self.generate_image_enhanced(image_prompt, method=image_method)
                
                # Save image
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                method_prefix = "photorealistic" if image_method == "photorealistic" else "enhanced"
                filename = f"{method_prefix}_{timestamp}.png"
                
                os.makedirs("results/generated_images", exist_ok=True)
                image_path = os.path.join("results", "generated_images", filename)
                image.save(image_path)
                
                # Generate text response
                text_response = self.process_message(f"I generated a {image_method} image of: {image_prompt}")
                
                if hasattr(text_response, 'text'):
                    response_text = text_response.text
                else:
                    response_text = str(text_response)
                
                # Combined response
                response = f"🎨 {response_text}\n\n🖼️ Generated {image_method} image: {image_path}"
                
                return ResponseObject(response, 0.95)
                
            except Exception as e:
                error_msg = f"⚠️ Enhanced image generation failed: {str(e)}"
                return ResponseObject(error_msg, 0.3)
        
        else:
            # Regular text processing
            return super().process_message(message)
    
    def get_enhanced_status(self) -> Dict:
        """Get enhanced status including HF models"""
        base_status = self.get_status()
        
        hf_status = {
            'hf_models_available': len(self.hf_models),
            'hf_pipelines_loaded': len(self.hf_pipelines),
            'photorealistic_enabled': len(self.hf_pipelines) > 0,
            'hf_models': list(self.hf_models.keys()),
            'generation_methods': ['quantum', 'photorealistic', 'auto']
        }
        
        base_status.update(hf_status)
        return base_status

class ResponseObject:
    """Response object for compatibility"""
    def __init__(self, text: str, confidence: float):
        self.text = text
        self.confidence = confidence

def main():
    """Test enhanced Essential Quantum AI"""
    print("🎯 Enhanced Essential Quantum AI Test")
    print("=" * 50)
    
    # Initialize enhanced AI
    ai = EnhancedEssentialQuantumAI(enable_image_generation=True, enable_hf_models=True)
    
    # Show status
    status = ai.get_enhanced_status()
    print(f"\n📊 Enhanced AI Status:")
    print(f"   Text parameters: {status.get('total_parameters', 0):,}")
    print(f"   HF models: {status['hf_models_available']}")
    print(f"   Photorealistic: {status['photorealistic_enabled']}")
    print(f"   Methods: {', '.join(status['generation_methods'])}")
    
    # Test prompts
    test_prompts = [
        "hello enhanced AI",
        "generate image: quantum cat in space",
        "create photorealistic image: beautiful sunset landscape",
        "draw quantum encoded: nano banana in laboratory"
    ]
    
    for prompt in test_prompts:
        print(f"\n🔹 Input: {prompt}")
        try:
            response = ai.process_message_enhanced(prompt)
            if hasattr(response, 'text'):
                print(f"🔸 Output: {response.text}")
            else:
                print(f"🔸 Output: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()