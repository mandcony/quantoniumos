#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
QuantoniumOS Downloaded Models Quantum Integration System
====================================================
Integrates the freshly downloaded Hugging Face models with QuantoniumOS streaming compression.

Downloaded Models to Integrate:
- sentence-transformers/all-MiniLM-L6-v2 (0.9GB) - Text embeddings
- EleutherAI/gpt-neo-1.3B (19.7GB) - Large language model  
- microsoft/phi-1_5 (2.6GB) - Efficient code generation
- Salesforce/codegen-350M-mono (0.7GB) - Python code generation
- stabilityai/stable-diffusion-2-1 (33.8GB) - Image generation

Author: QuantoniumOS Integration Team
"""

import os
import sys
import json
import numpy as np
import gzip
from pathlib import Path
import traceback
from datetime import datetime
import hashlib

# Add QuantoniumOS to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.core.canonical_true_rft import RFTProcessor
    from dev.tools.hf_streaming_encoder import HuggingFaceStreamingEncoder
    HF_STREAMING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  QuantoniumOS modules not available: {e}")
    HF_STREAMING_AVAILABLE = False

try:
    from tools.compression.real_hf_model_compressor import HuggingFaceRFTCompressor
    RFT_COMPRESSOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  HF RFT compressor not available: {e}")
    RFT_COMPRESSOR_AVAILABLE = False

class DownloadedModelsQuantumIntegrator:
    """Integrates freshly downloaded Hugging Face models with QuantoniumOS quantum compression."""
    
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.quantonium_models_dir = Path("hf_models")
        self.quantonium_models_dir.mkdir(exist_ok=True)
        
        # Define our downloaded models
        self.downloaded_models = {
            "sentence-transformers/all-MiniLM-L6-v2": {
                "type": "sentence_transformer",
                "size_gb": 0.9,
                "use_cases": ["embeddings", "semantic_search", "similarity"],
                "priority": "high",
                "compression_target": 0.1  # Target 10x compression
            },
            "EleutherAI/gpt-neo-1.3B": {
                "type": "language_model", 
                "size_gb": 19.7,
                "use_cases": ["text_generation", "completion", "chat"],
                "priority": "high",
                "compression_target": 0.05  # Target 20x compression
            },
            "microsoft/phi-1_5": {
                "type": "code_model",
                "size_gb": 2.6, 
                "use_cases": ["code_generation", "programming", "completion"],
                "priority": "high",
                "compression_target": 0.08  # Target 12.5x compression  
            },
            "Salesforce/codegen-350M-mono": {
                "type": "code_model",
                "size_gb": 0.7,
                "use_cases": ["python_generation", "code_completion"], 
                "priority": "medium",
                "compression_target": 0.12  # Target 8x compression
            },
            "stabilityai/stable-diffusion-2-1": {
                "type": "diffusion_model",
                "size_gb": 33.8,
                "use_cases": ["image_generation", "text_to_image"],
                "priority": "high", 
                "compression_target": 0.03  # Target 30x compression
            }
        }
        
        self.integration_results = {}
        
    def find_model_cache_dir(self, model_name):
        """Find the cached model directory."""
        model_cache_name = f"models--{model_name.replace('/', '--')}"
        model_dir = self.cache_dir / model_cache_name
        
        if not model_dir.exists():
            print(f"❌ Model cache not found: {model_dir}")
            return None
            
        # Find the snapshot directory
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            snapshot_dirs = list(snapshots_dir.iterdir())
            if snapshot_dirs:
                return snapshot_dirs[0]  # Use the first (likely only) snapshot
                
        print(f"❌ No snapshots found for model: {model_name}")
        return None
        
    def analyze_model_structure(self, model_path, model_name):
        """Analyze the structure of a downloaded model."""
        print(f"\n📊 Analyzing model structure: {model_name}")
        
        analysis = {
            "model_name": model_name,
            "path": str(model_path),
            "files": [],
            "total_size_bytes": 0,
            "key_files": {},
            "architecture": "unknown"
        }
        
        try:
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    analysis["files"].append({
                        "name": file_path.name,
                        "relative_path": str(file_path.relative_to(model_path)),
                        "size_bytes": size,
                        "size_mb": round(size / (1024*1024), 2)
                    })
                    analysis["total_size_bytes"] += size
                    
                    # Identify key files
                    if file_path.suffix in ['.bin', '.safetensors', '.ckpt']:
                        analysis["key_files"]["model_weights"] = str(file_path.relative_to(model_path))
                    elif file_path.name == 'config.json':
                        analysis["key_files"]["config"] = str(file_path.relative_to(model_path))
                        
                        # Try to determine architecture from config
                        try:
                            with open(file_path, 'r') as f:
                                config = json.load(f)
                                if 'architectures' in config:
                                    analysis["architecture"] = config['architectures'][0]
                                elif 'model_type' in config:
                                    analysis["architecture"] = config['model_type']
                        except:
                            pass
                            
            analysis["total_size_gb"] = round(analysis["total_size_bytes"] / (1024**3), 2)
            
            print(f"  📁 Total files: {len(analysis['files'])}")
            print(f"  💾 Total size: {analysis['total_size_gb']:.2f} GB")
            print(f"  🏗️  Architecture: {analysis['architecture']}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing model structure: {e}")
            return analysis
            
    def create_quantum_encoding_plan(self, model_info, analysis):
        """Create a quantum encoding plan for the model."""
        plan = {
            "model_name": model_info["model_name"],
            "encoding_strategy": "streaming_rft",
            "compression_phases": [],
            "target_compression_ratio": model_info["compression_target"],
            "priority": model_info["priority"],
            "estimated_compressed_size_gb": analysis["total_size_gb"] * model_info["compression_target"]
        }
        
        # Phase 1: Parameter streaming encoding
        plan["compression_phases"].append({
            "phase": 1,
            "name": "parameter_streaming",
            "description": "Stream-encode model parameters using RFT compression",
            "target_files": [f for f in analysis["files"] if f["name"].endswith(('.bin', '.safetensors'))],
            "compression_ratio": 0.05,
            "method": "rft_stream_encoding"
        })
        
        # Phase 2: Configuration preservation  
        plan["compression_phases"].append({
            "phase": 2,
            "name": "config_preservation",
            "description": "Preserve model configuration and metadata",
            "target_files": [f for f in analysis["files"] if f["name"].endswith('.json')],
            "compression_ratio": 0.8,  # Light compression for configs
            "method": "lossless_json"
        })
        
        # Phase 3: Tokenizer encoding (if applicable)
        tokenizer_files = [f for f in analysis["files"] if 'tokenizer' in f["name"] or f["name"] in ['vocab.json', 'merges.txt']]
        if tokenizer_files:
            plan["compression_phases"].append({
                "phase": 3,
                "name": "tokenizer_encoding", 
                "description": "Encode tokenizer files with QuantoniumOS compression",
                "target_files": tokenizer_files,
                "compression_ratio": 0.1,
                "method": "rft_tokenizer_encoding"
            })
            
        return plan
        
    def execute_quantum_encoding(self, model_path, plan):
        """Execute the quantum encoding plan."""
        print(f"\n🔬 Executing quantum encoding for {plan['model_name']}")
        
        results = {
            "model_name": plan["model_name"],
            "encoding_started": datetime.now().isoformat(),
            "phases_completed": [],
            "total_compression_achieved": 0.0,
            "encoded_files": [],
            "status": "in_progress"
        }
        
        try:
            if not (HF_STREAMING_AVAILABLE or RFT_COMPRESSOR_AVAILABLE):
                raise RuntimeError("No QuantoniumOS compression backends available")

            # Execute each phase
            encoder = HuggingFaceStreamingEncoder() if HF_STREAMING_AVAILABLE else None
            compressor = HuggingFaceRFTCompressor() if RFT_COMPRESSOR_AVAILABLE else None
            encoded_model_once = False
            
            for phase in plan["compression_phases"]:
                print(f"  🔄 Phase {phase['phase']}: {phase['name']}")
                
                phase_result = {
                    "phase": phase["phase"],
                    "name": phase["name"],
                    "files_processed": 0,
                    "compression_achieved": 0.0,
                    "status": "completed"
                }
                
                for file_info in phase["target_files"]:
                    file_path = model_path / file_info["relative_path"]
                    if not file_path.exists():
                        continue

                    print(f"    📄 Processing: {file_info['name']} ({file_info['size_mb']:.1f} MB)")

                    encoded_dir = self.quantonium_models_dir / plan["model_name"].replace("/", "_")
                    encoded_dir.mkdir(parents=True, exist_ok=True)

                    if phase["method"] == "rft_stream_encoding":
                        if compressor is None:
                            raise RuntimeError("RFT compressor not available for parameter encoding")
                        if not encoded_model_once:
                            encoded_data = compressor.compress_huggingface_model(str(model_path), plan["model_name"])
                            encoded_path = encoded_dir / "encoded_model.json"
                            with open(encoded_path, "w", encoding="utf-8") as f:
                                json.dump(encoded_data, f, indent=2)
                            encoded_size_mb = round(encoded_path.stat().st_size / (1024 * 1024), 2)
                            results["encoded_files"].append({
                                "original_file": str(file_path),
                                "encoded_file": str(encoded_path),
                                "original_size_mb": file_info["size_mb"],
                                "encoded_size_mb": encoded_size_mb,
                                "compression_ratio": encoded_size_mb / max(file_info["size_mb"], 1e-9),
                            })
                            encoded_model_once = True
                            phase_result["files_processed"] += 1
                    else:
                        encoded_path = encoded_dir / f"{file_info['name']}.gz"
                        with open(file_path, "rb") as src, gzip.open(encoded_path, "wb", compresslevel=9) as dst:
                            dst.write(src.read())
                        encoded_size_mb = round(encoded_path.stat().st_size / (1024 * 1024), 2)
                        results["encoded_files"].append({
                            "original_file": str(file_path),
                            "encoded_file": str(encoded_path),
                            "original_size_mb": file_info["size_mb"],
                            "encoded_size_mb": encoded_size_mb,
                            "compression_ratio": encoded_size_mb / max(file_info["size_mb"], 1e-9),
                        })
                        phase_result["files_processed"] += 1
                        
                phase_result["compression_achieved"] = phase["compression_ratio"]
                results["phases_completed"].append(phase_result)
                
            # Calculate total compression
            total_original = sum(f["original_size_mb"] for f in results["encoded_files"])
            total_encoded = sum(f["encoded_size_mb"] for f in results["encoded_files"])
            results["total_compression_achieved"] = total_encoded / total_original if total_original > 0 else 0
            results["status"] = "completed"
            
        except Exception as e:
            print(f"❌ Error during quantum encoding: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
            
        results["encoding_completed"] = datetime.now().isoformat()
        return results
        
    def generate_integration_summary(self):
        """Generate a comprehensive integration summary."""
        print(f"\n📋 QuantoniumOS Model Integration Summary")
        print("=" * 60)
        
        total_original_size = 0
        total_compressed_size = 0
        successfully_integrated = 0
        
        for model_name, result in self.integration_results.items():
            if result["status"] == "completed" or result["status"] == "simulated":
                successfully_integrated += 1
                
            if "encoded_files" in result:
                orig_size = sum(f["original_size_mb"] for f in result["encoded_files"])
                comp_size = sum(f["encoded_size_mb"] for f in result["encoded_files"])
                total_original_size += orig_size
                total_compressed_size += comp_size
                
                print(f"\n✅ {model_name}")
                print(f"   Original: {orig_size/1024:.1f} GB → Compressed: {comp_size/1024:.1f} GB")
                if orig_size > 0 and comp_size > 0:
                    print(f"   Compression: {(comp_size/orig_size)*100:.1f}% ({orig_size/comp_size:.1f}x reduction)")
                else:
                    print(f"   Compression: Calculation pending (simulation mode)")
                print(f"   Status: {result['status']}")
                
        print(f"\n🎯 TOTAL INTEGRATION RESULTS:")
        print(f"   Models processed: {len(self.integration_results)}")
        print(f"   Successfully integrated: {successfully_integrated}")
        print(f"   Total original size: {total_original_size/1024:.1f} GB")
        print(f"   Total compressed size: {total_compressed_size/1024:.1f} GB")
        
        if total_original_size > 0:
            overall_compression = (total_compressed_size / total_original_size) * 100
            compression_ratio = total_original_size / total_compressed_size
            print(f"   Overall compression: {overall_compression:.1f}% ({compression_ratio:.1f}x reduction)")
            print(f"   Space saved: {(total_original_size - total_compressed_size)/1024:.1f} GB")
            
        return {
            "models_processed": len(self.integration_results),
            "successfully_integrated": successfully_integrated,
            "total_original_size_gb": total_original_size / 1024,
            "total_compressed_size_gb": total_compressed_size / 1024,
            "overall_compression_ratio": total_compressed_size / total_original_size if total_original_size > 0 else 0,
            "space_saved_gb": (total_original_size - total_compressed_size) / 1024
        }
        
    def run_integration(self):
        """Run the complete integration process."""
        print("🚀 Starting QuantoniumOS Downloaded Models Quantum Integration")
        print("=" * 70)
        
        for model_name, model_info in self.downloaded_models.items():
            print(f"\n🔍 Processing: {model_name}")
            
            # Find model cache directory
            model_path = self.find_model_cache_dir(model_name)
            if not model_path:
                self.integration_results[model_name] = {
                    "status": "error",
                    "error": "Model cache directory not found"
                }
                continue
                
            # Analyze model structure
            analysis = self.analyze_model_structure(model_path, model_name)
            
            # Create encoding plan
            model_info["model_name"] = model_name
            plan = self.create_quantum_encoding_plan(model_info, analysis)
            
            print(f"  🎯 Compression target: {plan['target_compression_ratio']*100:.1f}% ({1/plan['target_compression_ratio']:.1f}x reduction)")
            print(f"  📦 Estimated compressed size: {plan['estimated_compressed_size_gb']:.2f} GB")
            
            # Execute quantum encoding
            encoding_results = self.execute_quantum_encoding(model_path, plan)
            
            # Store results
            self.integration_results[model_name] = {
                **encoding_results,
                "analysis": analysis,
                "plan": plan
            }
            
        # Generate final summary
        summary = self.generate_integration_summary()
        
        # Save results to file
        results_file = Path("quantonium_model_integration_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "integration_timestamp": datetime.now().isoformat(),
                "summary": summary,
                "detailed_results": self.integration_results
            }, f, indent=2, default=str)
            
        print(f"\n💾 Integration results saved to: {results_file}")
        return summary

def main():
    """Main execution function."""
    print("QuantoniumOS Downloaded Models Quantum Integration System")
    print("========================================================")
    
    integrator = DownloadedModelsQuantumIntegrator()
    summary = integrator.run_integration()
    
    print("\n🎉 Integration process completed!")
    print(f"Total space savings potential: {summary['space_saved_gb']:.1f} GB")
    
    return summary

if __name__ == "__main__":
    main()