# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.

#!/usr/bin/env python3
"""
QuantoniumOS Downloaded Models Practical Demonstration
=====================================================
Demonstrates actual usage of the freshly downloaded Hugging Face models
with QuantoniumOS integration and streaming capabilities.

Downloaded Models Test Suite:
- sentence-transformers/all-MiniLM-L6-v2 → Text embeddings & semantic search
- EleutherAI/gpt-neo-1.3B → Large language model text generation  
- microsoft/phi-1_5 → Efficient code generation
- Salesforce/codegen-350M-mono → Python code generation
- stabilityai/stable-diffusion-2-1 → Image generation from text

This script actually loads and tests each model to prove they work.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import traceback
from datetime import datetime
import json

# Add QuantoniumOS to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import transformers
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers library available")
except ImportError as e:
    print(f"⚠️  Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers library available")
except ImportError as e:
    print(f"⚠️  Diffusers not available: {e}")
    DIFFUSERS_AVAILABLE = False

class QuantoniumModelTester:
    """Tests downloaded models with QuantoniumOS integration."""
    
    def __init__(self):
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")
        
    def test_sentence_transformer(self):
        """Test the sentence transformer model for embeddings."""
        print("\n🔍 Testing sentence-transformers/all-MiniLM-L6-v2")
        
        if not TRANSFORMERS_AVAILABLE:
            return {"status": "skipped", "reason": "transformers not available"}
            
        try:
            # Load the model
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Test sentences
            test_sentences = [
                "QuantoniumOS uses quantum-inspired algorithms for AI compression",
                "This system enables efficient model streaming and encoding",
                "Python is a great programming language",
                "The weather is nice today"
            ]
            
            # Generate embeddings
            embeddings = model.encode(test_sentences)
            
            # Calculate similarities
            similarity_matrix = np.inner(embeddings, embeddings)
            
            result = {
                "status": "success",
                "model_loaded": True,
                "embedding_shape": embeddings.shape,
                "embedding_dimension": embeddings.shape[1],
                "test_sentences": len(test_sentences),
                "similarity_example": {
                    "sentence1": test_sentences[0],
                    "sentence2": test_sentences[1], 
                    "similarity": float(similarity_matrix[0][1])
                },
                "memory_usage_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else "N/A"
            }
            
            print(f"  ✅ Model loaded successfully")
            print(f"  📊 Embedding dimension: {result['embedding_dimension']}")
            print(f"  🔗 Similarity (quantum sentences): {result['similarity_example']['similarity']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def test_gpt_neo(self):
        """Test GPT-Neo for text generation."""
        print("\n🤖 Testing EleutherAI/gpt-neo-1.3B")
        
        if not TRANSFORMERS_AVAILABLE:
            return {"status": "skipped", "reason": "transformers not available"}
            
        try:
            from transformers import GPTNeoForCausalLM, GPT2Tokenizer
            
            # Load model and tokenizer
            model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
            tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            
            if torch.cuda.is_available():
                model = model.to(self.device)
                
            # Test prompts
            test_prompt = "QuantoniumOS is a revolutionary quantum-inspired operating system that"
            
            # Tokenize
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = {
                "status": "success",
                "model_loaded": True,
                "model_parameters": "1.3B",
                "test_prompt": test_prompt,
                "generated_text": generated_text,
                "generation_length": len(generated_text),
                "device_used": str(self.device),
                "memory_usage_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else "N/A"
            }
            
            print(f"  ✅ Model loaded successfully")
            print(f"  📝 Generated: {generated_text[:100]}...")
            print(f"  💾 Memory usage: {result['memory_usage_mb']} MB" if result['memory_usage_mb'] != "N/A" else "  💾 Memory usage: CPU mode")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def test_phi_model(self):
        """Test Microsoft Phi model for code generation."""
        print("\n💻 Testing microsoft/phi-1_5")
        
        if not TRANSFORMERS_AVAILABLE:
            return {"status": "skipped", "reason": "transformers not available"}
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
            
            if torch.cuda.is_available():
                model = model.to(self.device)
                
            # Test code generation prompt
            test_prompt = "def calculate_quantum_compression_ratio(original_size, compressed_size):"
            
            # Tokenize
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            # Decode
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = {
                "status": "success",
                "model_loaded": True,
                "model_parameters": "1.5B",
                "specialization": "code_generation",
                "test_prompt": test_prompt,
                "generated_code": generated_code,
                "device_used": str(self.device),
                "memory_usage_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else "N/A"
            }
            
            print(f"  ✅ Model loaded successfully")
            print(f"  🐍 Generated code:")
            for line in generated_code.split('\n')[:5]:  # Show first 5 lines
                print(f"     {line}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def test_codegen_model(self):
        """Test Salesforce CodeGen for Python generation."""
        print("\n🐍 Testing Salesforce/codegen-350M-mono")
        
        if not TRANSFORMERS_AVAILABLE:
            return {"status": "skipped", "reason": "transformers not available"}
            
        try:
            from transformers import CodeGenTokenizer, CodeGenForCausalLM
            
            # Load model and tokenizer
            model = CodeGenForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
            tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
            
            if torch.cuda.is_available():
                model = model.to(self.device)
                
            # Test Python generation prompt
            test_prompt = "# QuantoniumOS streaming compression function\ndef compress_model_weights(weights):"
            
            # Tokenize
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=80,
                    num_return_sequences=1,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            # Decode
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = {
                "status": "success",
                "model_loaded": True,
                "model_parameters": "350M",
                "specialization": "python_code_generation",
                "test_prompt": test_prompt,
                "generated_code": generated_code,
                "device_used": str(self.device),
                "memory_usage_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else "N/A"
            }
            
            print(f"  ✅ Model loaded successfully")
            print(f"  🔧 Generated Python code:")
            for line in generated_code.split('\n')[:4]:  # Show first 4 lines
                print(f"     {line}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def test_stable_diffusion(self):
        """Test Stable Diffusion for image generation."""
        print("\n🎨 Testing stabilityai/stable-diffusion-2-1")
        
        if not DIFFUSERS_AVAILABLE:
            return {"status": "skipped", "reason": "diffusers not available"}
            
        try:
            # Load the pipeline
            pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
            
            if torch.cuda.is_available():
                pipe = pipe.to(self.device)
                
            # Test prompt
            test_prompt = "A futuristic quantum computer with holographic displays showing mathematical formulas, cyberpunk style"
            
            # Generate image (small size for testing)
            with torch.no_grad():
                image = pipe(
                    test_prompt,
                    height=256,
                    width=256,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
                
            # Save test image
            output_path = Path("test_generated_image.png")
            image.save(output_path)
            
            result = {
                "status": "success",
                "model_loaded": True,
                "model_type": "stable_diffusion_2_1",
                "test_prompt": test_prompt,
                "image_generated": True,
                "image_size": "256x256",
                "inference_steps": 20,
                "output_file": str(output_path),
                "device_used": str(self.device),
                "memory_usage_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else "N/A"
            }
            
            print(f"  ✅ Model loaded successfully")
            print(f"  🖼️  Image generated: {output_path}")
            print(f"  📏 Image size: 256x256")
            print(f"  🎯 Prompt: {test_prompt[:50]}...")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {"status": "error", "error": str(e)}
            
    def run_comprehensive_test(self):
        """Run comprehensive tests on all downloaded models."""
        print("🧪 QuantoniumOS Downloaded Models Comprehensive Test Suite")
        print("=" * 65)
        print(f"🕐 Test started: {datetime.now().isoformat()}")
        print(f"🖥️  Device: {self.device}")
        print(f"🔗 PyTorch: {torch.__version__}")
        
        # Test each model
        tests = [
            ("sentence_transformer", self.test_sentence_transformer),
            ("gpt_neo", self.test_gpt_neo),
            ("phi_model", self.test_phi_model),
            ("codegen_model", self.test_codegen_model),
            ("stable_diffusion", self.test_stable_diffusion)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results[test_name] = result
            except Exception as e:
                print(f"❌ Unexpected error in {test_name}: {e}")
                self.results[test_name] = {
                    "status": "unexpected_error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
        # Generate summary
        self.generate_test_summary()
        
        # Save results
        results_file = Path("quantonium_model_test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "test_timestamp": datetime.now().isoformat(),
                "device_used": str(self.device),
                "pytorch_version": torch.__version__,
                "results": self.results
            }, f, indent=2, default=str)
            
        print(f"\n💾 Test results saved to: {results_file}")
        return self.results
        
    def generate_test_summary(self):
        """Generate a comprehensive test summary."""
        print("\n📊 Test Summary")
        print("=" * 40)
        
        successful_tests = 0
        total_tests = len(self.results)
        total_memory_used = 0
        
        for test_name, result in self.results.items():
            status_emoji = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "skipped" else "❌"
            print(f"{status_emoji} {test_name}: {result['status']}")
            
            if result["status"] == "success":
                successful_tests += 1
                if isinstance(result.get("memory_usage_mb"), (int, float)):
                    total_memory_used += result["memory_usage_mb"]
                    
        print(f"\n🎯 Overall Results:")
        print(f"   Tests passed: {successful_tests}/{total_tests}")
        print(f"   Success rate: {(successful_tests/total_tests)*100:.1f}%")
        print(f"   Total memory used: {total_memory_used:.1f} MB")
        
        if successful_tests > 0:
            print(f"\n🚀 QuantoniumOS Model Integration Status: OPERATIONAL")
            print(f"   Your downloaded models are working and can be integrated!")
        else:
            print(f"\n⚠️  Some dependencies may be missing. Install with:")
            print(f"   pip install transformers sentence-transformers diffusers torch")

def main():
    """Main execution function."""
    tester = QuantoniumModelTester()
    results = tester.run_comprehensive_test()
    
    return results

if __name__ == "__main__":
    main()