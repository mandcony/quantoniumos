#!/usr/bin/env python3
"""
RFT Compression Curve Generator
===============================
Generates trustworthy compression-fidelity curves with implementation path tracking
"""

import json
import numpy as np
import time
import sys
from datetime import datetime
from pathlib import Path

sys.path.append('src/core')

def generate_compression_curve():
    """Generate compression-fidelity curves for different sparsity levels"""
    
    try:
        from canonical_true_rft import CanonicalTrueRFT
        implementation = "python_reference"
    except ImportError:
        print("⚠ Cannot import RFT implementation")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "metadata": {
            "timestamp": timestamp,
            "implementation": implementation,
            "test_type": "rft_compression_fidelity_curve",
            "description": "Compression ratios vs fidelity for top-k coefficient retention"
        },
        "curves": []
    }
    
    # Test different state sizes
    sizes = [64, 128, 256, 512]
    
    for size in sizes:
        print(f"Testing size {size}...")
        
        rft = CanonicalTrueRFT(size)
        
        # Generate structured quantum state (more compressible)
        np.random.seed(42)  # Reproducible
        raw_state = np.random.random(size) + 1j * np.random.random(size)
        
        # Add some structure (decay pattern - more realistic for quantum states)
        for i in range(size):
            raw_state[i] *= np.exp(-i / (size * 0.3))
            
        quantum_state = raw_state / np.linalg.norm(raw_state)
        
        # Apply RFT transform
        start_time = time.time()
        rft_coeffs = rft.forward_transform(quantum_state)
        transform_time = (time.time() - start_time) * 1000
        
        # Test different sparsity levels (top-k retention)
        curve_data = []
        
        # Sort coefficients by magnitude
        sorted_indices = np.argsort(np.abs(rft_coeffs))[::-1]
        
        # Test different retention ratios
        retention_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        
        for ratio in retention_ratios:
            keep_count = max(1, int(size * ratio))
            
            # Create sparse representation
            sparse_coeffs = np.zeros_like(rft_coeffs)
            sparse_coeffs[sorted_indices[:keep_count]] = rft_coeffs[sorted_indices[:keep_count]]
            
            # Reconstruct
            reconstructed = rft.inverse_transform(sparse_coeffs)
            
            # Calculate fidelity and compression metrics
            fidelity = abs(np.vdot(quantum_state, reconstructed))**2
            reconstruction_error = np.linalg.norm(quantum_state - reconstructed)
            compression_ratio = size / keep_count
            
            curve_data.append({
                "retention_ratio": ratio,
                "coefficients_kept": keep_count,
                "compression_ratio": float(compression_ratio),
                "fidelity": float(fidelity),
                "reconstruction_error": float(reconstruction_error),
                "compression_percentage": float(((compression_ratio - 1) / compression_ratio) * 100)
            })
        
        results["curves"].append({
            "size": size,
            "transform_time_ms": transform_time,
            "unitarity_error": float(rft.get_unitarity_error()),
            "curve_points": curve_data
        })
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"rft_compression_curve_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Compression curve saved to {output_file}")
    
    # Print summary
    print("\nCOMPRESSION-FIDELITY SUMMARY:")
    for curve in results["curves"]:
        size = curve["size"]
        best_90_fidelity = None
        best_99_fidelity = None
        
        for point in curve["curve_points"]:
            if point["fidelity"] >= 0.99 and best_99_fidelity is None:
                best_99_fidelity = point
            if point["fidelity"] >= 0.90 and best_90_fidelity is None:
                best_90_fidelity = point
        
        print(f"Size {size:3d}: ", end="")
        if best_99_fidelity:
            print(f"99% fidelity @ {best_99_fidelity['compression_ratio']:.1f}x compression", end="")
        if best_90_fidelity:
            print(f", 90% fidelity @ {best_90_fidelity['compression_ratio']:.1f}x compression")
        else:
            print()

def generate_model_compression_summary():
    """Generate model file compression summary"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    
    results = {
        "metadata": {
            "timestamp": timestamp,
            "test_type": "model_file_compression_summary", 
            "description": "Actual file size reductions for stored model representations"
        },
        "model_compressions": []
    }
    
    # Check actual compressed model files
    model_files = [
        {
            "name": "Phi-3 Mini Quantum Resonance",
            "file_path": "data/parameters/quantum_models/phi3_mini_quantum_resonance.pkl.gz",
            "reference_size_gb": 7.6,  # Typical Phi-3 Mini size
            "reference_params": "3.8B"
        }
    ]
    
    for model in model_files:
        file_path = Path(model["file_path"])
        
        if file_path.exists():
            compressed_size_bytes = file_path.stat().st_size
            compressed_size_gb = compressed_size_bytes / (1024**3)
            
            compression_ratio = model["reference_size_gb"] / compressed_size_gb
            compression_percentage = ((compression_ratio - 1) / compression_ratio) * 100
            
            results["model_compressions"].append({
                "model_name": model["name"],
                "reference_size_gb": model["reference_size_gb"],
                "reference_params": model["reference_params"],
                "compressed_size_bytes": compressed_size_bytes,
                "compressed_size_gb": compressed_size_gb,
                "compression_ratio": compression_ratio,
                "compression_percentage": compression_percentage,
                "file_exists": True
            })
        else:
            results["model_compressions"].append({
                "model_name": model["name"],
                "file_exists": False,
                "note": f"File not found: {model['file_path']}"
            })
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"model_file_compression_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Model compression summary saved to {output_file}")
    
    # Print summary
    print("\nMODEL FILE COMPRESSION SUMMARY:")
    for model in results["model_compressions"]:
        if model.get("file_exists"):
            print(f"• {model['model_name']}: {model['reference_size_gb']}GB → {model['compressed_size_gb']:.4f}GB")
            print(f"  Ratio: {model['compression_ratio']:.0f}x ({model['compression_percentage']:.2f}% compression)")
        else:
            print(f"• {model['model_name']}: {model.get('note', 'Not available')}")

if __name__ == "__main__":
    print("=== RFT COMPRESSION ANALYSIS SUITE ===")
    print()
    
    print("1. Generating compression-fidelity curves...")
    generate_compression_curve()
    
    print("\n2. Generating model compression summary...")
    generate_model_compression_summary()
    
    print("\n✅ All compression analysis artifacts generated!")
    print("📁 Check results/ directory for timestamped JSON files")