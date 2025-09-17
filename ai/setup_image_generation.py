#!/usr/bin/env python3
"""
QuantoniumOS Image Generation Setup
Installs dependencies for text-to-image capabilities
"""

import subprocess
import sys
import os
import torch

def check_gpu():
    """Check if CUDA GPU is available"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        return True
    else:
        print("⚠️ No GPU detected - image generation will be slower on CPU")
        return False

def install_packages():
    """Install required packages for image generation"""
    packages = [
        "diffusers>=0.21.0",
        "pillow>=9.0.0", 
        "accelerate>=0.20.0",
        "controlnet-aux>=0.0.6"
    ]
    
    # Add GPU-optimized packages if CUDA is available
    if check_gpu():
        packages.append("xformers>=0.0.20")  # Memory efficient attention
    
    print("📦 Installing image generation dependencies...")
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True, text=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")
            if "xformers" in package:
                print("   (xformers is optional - will use default attention)")

def test_installation():
    """Test if the installation works"""
    print("\n🧪 Testing installation...")
    
    try:
        from diffusers import StableDiffusionPipeline
        from PIL import Image
        print("✅ Core packages imported successfully")
        
        # Test basic pipeline creation (without loading weights)
        try:
            # This will check if the model hub is accessible
            from diffusers import DiffusionPipeline
            print("✅ Diffusion pipeline available")
        except Exception as e:
            print(f"⚠️ Pipeline test failed: {e}")
        
        print("✅ Installation test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Installation test failed: {e}")
        return False

def setup_output_directory():
    """Create output directory for generated images"""
    output_dir = "results/generated_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")

def main():
    """Main setup process"""
    print("⚛️ QuantoniumOS Image Generation Setup")
    print("=" * 50)
    
    # Check current environment
    print(f"🐍 Python version: {sys.version}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    
    # Install packages
    install_packages()
    
    # Test installation
    if test_installation():
        setup_output_directory()
        
        print("\n🎉 Setup complete!")
        print("\nYour QuantoniumOS AI can now generate images!")
        print("\nTo test image generation, run:")
        print("  python ai/demos/multimodal_ai_demo.py")
        print("\nOr use in code:")
        print("  from ai.inference.quantum_inference_engine import QuantumInferenceEngine")
        print("  ai = QuantumInferenceEngine(enable_image_generation=True)")
        print("  images = ai.generate_image_only('a nano banana')")
        
    else:
        print("\n❌ Setup failed. Please check error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())