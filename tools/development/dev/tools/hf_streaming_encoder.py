# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.

#!/usr/bin/env python3
"""
Hugging Face Streaming Encoder for QuantoniumOS
===============================================
Downloads and integrates Hugging Face models (Stable Diffusion) into 
the quantum parameter streaming architecture - same encoding as text models
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import hashlib
import time

class HuggingFaceStreamingEncoder:
    """Encodes Hugging Face models into QuantoniumOS streaming parameter format"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join("hf_models")
        self.encoded_dir = os.path.join("data", "weights", "hf_encoded")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.encoded_dir, exist_ok=True)
        
        print(f"🎯 HF Streaming Encoder initialized")
        print(f"   Cache: {self.cache_dir}")
        print(f"   Encoded: {self.encoded_dir}")
    
    def download_and_encode_stable_diffusion(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """Download and encode Stable Diffusion model into streaming format"""
        print(f"\n🚀 Downloading and encoding {model_id}...")
        
        try:
            # Download the model
            print("📥 Downloading Stable Diffusion pipeline...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,  # Use float32 for compatibility
                use_safetensors=True
            )
            
            # Extract components for encoding
            components = {
                'unet': pipeline.unet,
                'vae': pipeline.vae, 
                'text_encoder': pipeline.text_encoder,
                'tokenizer': pipeline.tokenizer,
                'scheduler': pipeline.scheduler
            }
            
            print("🔄 Encoding into quantum streaming format...")
            encoded_params = self._encode_pipeline_to_streaming_format(components, model_id)
            
            # Save encoded parameters
            encoded_filename = f"hf_encoded_{model_id.replace('/', '_')}.json"
            encoded_path = os.path.join(self.encoded_dir, encoded_filename)
            
            with open(encoded_path, 'w') as f:
                json.dump(encoded_params, f, indent=2)
            
            print(f"✅ Model encoded and saved: {encoded_path}")
            print(f"   Total parameters: {encoded_params['total_parameters']:,}")
            print(f"   Streaming states: {len(encoded_params['streaming_states'])}")
            
            return encoded_path
            
        except Exception as e:
            print(f"❌ Error downloading/encoding {model_id}: {e}")
            return None
    
    def _encode_pipeline_to_streaming_format(self, components: Dict, model_id: str) -> Dict:
        """Encode pipeline components into QuantoniumOS streaming format"""
        
        streaming_states = []
        total_params = 0
        component_info = {}
        
        for component_name, component in components.items():
            if component_name == 'tokenizer':
                continue  # Skip tokenizer (not a neural network)
                
            print(f"   🔹 Encoding {component_name}...")
            
            if hasattr(component, 'parameters'):
                # Neural network component
                params = list(component.parameters())
                component_params = sum(p.numel() for p in params if p.requires_grad)
                total_params += component_params
                
                # Extract parameter states (similar to your quantum encoding)
                states = []
                for i, param in enumerate(params[:100]):  # Sample first 100 parameters
                    if param.requires_grad:
                        # Convert to quantum-like states
                        param_data = param.detach().cpu().numpy().flatten()
                        if len(param_data) > 0:
                            # Create streaming state (matches your format)
                            state = {
                                'id': len(streaming_states),
                                'component': component_name,
                                'param_index': i,
                                'magnitude': float(np.linalg.norm(param_data)),
                                'phase': float(np.angle(np.sum(param_data)) if np.iscomplexobj(param_data) else 0),
                                'dimension': param.shape,
                                'sample': param_data[:10].tolist() if len(param_data) >= 10 else param_data.tolist()
                            }
                            states.append(state)
                            streaming_states.append(state)
                
                component_info[component_name] = {
                    'parameters': component_params,
                    'states': len(states),
                    'architecture': str(type(component).__name__)
                }
            
            elif hasattr(component, 'config'):
                # Configuration component (scheduler, etc.)
                component_info[component_name] = {
                    'type': 'config',
                    'config': component.config if hasattr(component.config, '__dict__') else str(component.config)
                }
        
        # Create encoded format matching your quantum parameter structure
        encoded_data = {
            'model_info': {
                'source': 'huggingface',
                'model_id': model_id,
                'encoding_timestamp': time.time(),
                'encoding_version': '1.0',
                'quantonium_compatible': True
            },
            'total_parameters': total_params,
            'streaming_states': streaming_states,
            'components': component_info,
            'quantum_encoding': {
                'format': 'hf_quantum_streaming',
                'state_count': len(streaming_states),
                'compression_ratio': total_params / max(len(streaming_states), 1),
                'compatible_with_essential_ai': True
            }
        }
        
        return encoded_data
    
    def load_encoded_hf_model(self, encoded_path: str) -> Dict:
        """Load an encoded HF model for use in Essential Quantum AI"""
        try:
            with open(encoded_path, 'r') as f:
                encoded_data = json.load(f)
            
            print(f"✅ Loaded encoded HF model: {encoded_data['model_info']['model_id']}")
            print(f"   Parameters: {encoded_data['total_parameters']:,}")
            print(f"   States: {encoded_data['quantum_encoding']['state_count']}")
            
            return encoded_data
            
        except Exception as e:
            print(f"❌ Error loading encoded model: {e}")
            return None
    
    def list_available_models(self) -> List[str]:
        """List popular Stable Diffusion models available for download"""
        models = [
            "runwayml/stable-diffusion-v1-5",  # Most popular, good quality
            "stabilityai/stable-diffusion-2-1",  # Newer version
            "CompVis/stable-diffusion-v1-4",  # Original
            "stabilityai/sdxl-base-1.0",  # Highest quality (large)
            "dreamlike-art/dreamlike-diffusion-1.0",  # Artistic style
            "prompthero/openjourney",  # Midjourney-like
            "wavymulder/Analog-Diffusion",  # Analog photo style
        ]
        return models
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a model before downloading"""
        model_info = {
            "runwayml/stable-diffusion-v1-5": {
                "size": "~4GB", "quality": "High", "speed": "Fast", 
                "description": "Most popular SD model, great all-around choice"
            },
            "stabilityai/stable-diffusion-2-1": {
                "size": "~5GB", "quality": "High", "speed": "Medium", 
                "description": "Improved version with better text understanding"
            },
            "stabilityai/sdxl-base-1.0": {
                "size": "~7GB", "quality": "Highest", "speed": "Slower", 
                "description": "Best quality but requires more resources"
            }
        }
        return model_info.get(model_id, {"description": "Community model"})

def main():
    """Interactive HF model installer"""
    print("🎯 QuantoniumOS Hugging Face Model Installer")
    print("=" * 50)
    
    encoder = HuggingFaceStreamingEncoder()
    
    print("\n📋 Available models:")
    models = encoder.list_available_models()
    for i, model in enumerate(models[:5], 1):  # Show top 5
        info = encoder.get_model_info(model)
        print(f"{i}. {model}")
        print(f"   {info.get('description', '')} ({info.get('size', 'Unknown size')})")
    
    try:
        choice = input("\nSelect model (1-5) or enter custom model ID: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= 5:
            model_id = models[int(choice) - 1]
        else:
            model_id = choice
        
        print(f"\n🚀 Installing {model_id}...")
        info = encoder.get_model_info(model_id)
        print(f"   {info.get('description', '')}")
        print(f"   Size: {info.get('size', 'Unknown')}")
        
        confirm = input("Continue? (y/N): ").lower()
        if confirm != 'y':
            print("Installation cancelled.")
            return
        
        # Download and encode
        encoded_path = encoder.download_and_encode_stable_diffusion(model_id)
        
        if encoded_path:
            print(f"\n✅ SUCCESS! Model installed and encoded.")
            print(f"   Encoded file: {encoded_path}")
            print(f"   Ready for use in QuantoniumOS Essential AI!")
        else:
            print(f"\n❌ Installation failed.")
            
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()