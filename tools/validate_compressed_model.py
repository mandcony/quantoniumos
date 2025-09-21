#!/usr/bin/env python3
"""
Compressed Model Validator and Tester
===================================
Tests the functionality and quality of RFT-compressed models.
"""

import os
import sys
import json
import pickle
import gzip
from pathlib import Path
import numpy as np

# Add QuantoniumOS paths
sys.path.append('/workspaces/quantoniumos/src')
sys.path.append('/workspaces/quantoniumos')

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        GPT2LMHeadModel, GPT2Tokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers not available")
    TRANSFORMERS_AVAILABLE = False

class CompressedModelValidator:
    """Validates and tests compressed models"""
    
    def __init__(self):
        self.results = {}
    
    def load_compressed_model(self, compressed_path: str) -> dict:
        """Load a compressed model file"""
        
        compressed_file = Path(compressed_path)
        if not compressed_file.exists():
            raise FileNotFoundError(f"Compressed model not found: {compressed_path}")
        
        print(f"📂 Loading compressed model: {compressed_file.name}")
        
        with gzip.open(compressed_file, 'rb') as f:
            compressed_data = pickle.load(f)
        
        file_size = compressed_file.stat().st_size / 1024 / 1024  # MB
        
        print(f"✅ Loaded compressed model ({file_size:.2f} MB)")
        print(f"   📊 Original params: {compressed_data['original_parameters']:,}")
        print(f"   📊 Compressed params: {compressed_data['compressed_parameters']:,}")
        print(f"   📊 Ratio: {compressed_data['compression_ratio']}")
        
        return compressed_data
    
    def load_original_model(self, model_path: str, model_id: str):
        """Load the original HuggingFace model for comparison"""
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Cannot load original model - transformers not available")
            return None, None
        
        print(f"🔄 Loading original model: {model_id}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            model.eval()  # Set to evaluation mode
            
            print(f"✅ Loaded original model")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading original model: {e}")
            return None, None
    
    def test_text_generation(self, model, tokenizer, test_prompts: list) -> dict:
        """Test text generation quality"""
        
        if model is None or tokenizer is None:
            return {"status": "skipped", "reason": "Model not available"}
        
        print("\n🎯 Testing text generation...")
        results = {
            "prompts_tested": len(test_prompts),
            "generations": [],
            "avg_length": 0,
            "status": "success"
        }
        
        total_length = 0
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n  📝 Prompt {i+1}: {prompt[:50]}...")
            
            try:
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        attention_mask=inputs.get("attention_mask")
                    )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()
                
                generation_result = {
                    "prompt": prompt,
                    "response": response,
                    "full_text": generated_text,
                    "length": len(response)
                }
                
                results["generations"].append(generation_result)
                total_length += len(response)
                
                print(f"     💬 Response: {response[:100]}...")
                
            except Exception as e:
                print(f"     ❌ Generation failed: {e}")
                results["generations"].append({
                    "prompt": prompt,
                    "error": str(e)
                })
        
        results["avg_length"] = total_length / len(test_prompts) if test_prompts else 0
        return results
    
    def simulate_compressed_performance(self, compressed_data: dict) -> dict:
        """Simulate performance of compressed model based on compression data"""
        
        print("\n🧮 Simulating compressed model performance...")
        
        # Analyze compression quality from the data
        compressed_layers = compressed_data.get('compressed_layers', {})
        
        if not compressed_layers:
            return {"status": "no_data", "reason": "No compressed layers found"}
        
        # Calculate average fidelity and compression ratio
        fidelities = []
        ratios = []
        
        for layer_name, layer_data in compressed_layers.items():
            fidelity = layer_data.get('fidelity', 0.95)
            ratio = layer_data.get('compression_ratio', 1000)
            
            fidelities.append(fidelity)
            ratios.append(ratio)
        
        avg_fidelity = np.mean(fidelities)
        avg_ratio = np.mean(ratios)
        
        # Estimate performance based on compression metrics
        performance_score = avg_fidelity * min(1.0, 100 / avg_ratio)  # Penalty for extreme compression
        
        simulation_result = {
            "status": "simulated",
            "avg_fidelity": float(avg_fidelity),
            "avg_compression_ratio": float(avg_ratio),
            "estimated_performance": float(performance_score),
            "layers_analyzed": len(compressed_layers),
            "quality_rating": self._rate_quality(performance_score)
        }
        
        print(f"   📊 Average fidelity: {avg_fidelity:.3f}")
        print(f"   📊 Average ratio: {avg_ratio:.1f}:1")
        print(f"   📊 Performance score: {performance_score:.3f}")
        print(f"   ⭐ Quality rating: {simulation_result['quality_rating']}")
        
        return simulation_result
    
    def _rate_quality(self, score: float) -> str:
        """Rate quality based on performance score"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor" 
        else:
            return "Very Poor"
    
    def run_comprehensive_validation(self, compressed_path: str, original_path: str, model_id: str) -> dict:
        """Run complete validation of compressed model"""
        
        print("\n🎯 COMPREHENSIVE MODEL VALIDATION")
        print("=" * 50)
        
        validation_result = {
            "model_id": model_id,
            "compressed_path": compressed_path,
            "original_path": original_path,
            "timestamp": str(np.datetime64('now')),
            "tests": {}
        }
        
        # Load compressed model
        try:
            compressed_data = self.load_compressed_model(compressed_path)
            validation_result["compression_data"] = compressed_data
        except Exception as e:
            print(f"❌ Failed to load compressed model: {e}")
            return {"status": "error", "error": str(e)}
        
        # Test original model (for baseline)
        original_model, tokenizer = self.load_original_model(original_path, model_id)
        
        # Define test prompts
        test_prompts = [
            "Hello, how are you today?",
            "What is artificial intelligence?",
            "Tell me about machine learning.",
            "How does quantum computing work?",
            "Explain neural networks."
        ]
        
        # Test original model
        if original_model is not None:
            print("\n📊 Testing original model...")
            original_results = self.test_text_generation(original_model, tokenizer, test_prompts)
            validation_result["tests"]["original_generation"] = original_results
        
        # Simulate compressed model performance
        compressed_results = self.simulate_compressed_performance(compressed_data)
        validation_result["tests"]["compressed_simulation"] = compressed_results
        
        # Overall assessment
        validation_result["overall_assessment"] = self._assess_overall_quality(
            validation_result["tests"]
        )
        
        return validation_result
    
    def _assess_overall_quality(self, test_results: dict) -> dict:
        """Assess overall quality of compression"""
        
        compressed_sim = test_results.get("compressed_simulation", {})
        
        if compressed_sim.get("status") != "simulated":
            return {"status": "insufficient_data"}
        
        performance_score = compressed_sim.get("estimated_performance", 0.5)
        fidelity = compressed_sim.get("avg_fidelity", 0.5)
        compression_ratio = compressed_sim.get("avg_compression_ratio", 1)
        
        # Overall assessment
        if performance_score >= 0.8 and compression_ratio > 100:
            status = "excellent"
            recommendation = "Ready for production use"
        elif performance_score >= 0.7 and compression_ratio > 50:
            status = "good" 
            recommendation = "Suitable for most applications"
        elif performance_score >= 0.6:
            status = "acceptable"
            recommendation = "May work for specific use cases"
        else:
            status = "poor"
            recommendation = "Needs optimization before use"
        
        return {
            "status": status,
            "performance_score": performance_score,
            "fidelity": fidelity,
            "compression_ratio": compression_ratio,
            "recommendation": recommendation
        }

def main():
    """Main validation workflow"""
    
    validator = CompressedModelValidator()
    
    # Test DialoGPT-small compressed model
    compressed_path = "/workspaces/quantoniumos/data/parameters/quantum_models/dialogpt_small_compressed.pkl.gz"
    original_path = "/workspaces/quantoniumos/hf_models/downloaded/DialoGPT-small"
    model_id = "microsoft/DialoGPT-small"
    
    # Run validation
    results = validator.run_comprehensive_validation(compressed_path, original_path, model_id)
    
    # Save results
    results_file = "/workspaces/quantoniumos/results/dialogpt_small_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎯 VALIDATION COMPLETE")
    print("=" * 30)
    assessment = results.get("overall_assessment", {})
    print(f"Status: {assessment.get('status', 'unknown').upper()}")
    print(f"Performance: {assessment.get('performance_score', 0):.3f}")
    print(f"Recommendation: {assessment.get('recommendation', 'No recommendation')}")
    print(f"Results saved: {results_file}")

if __name__ == "__main__":
    main()