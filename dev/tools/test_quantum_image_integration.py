#!/usr/bin/env python3
"""
Test script for integrated quantum image generation with HF style guidance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dev.tools.essential_quantum_ai import EssentialQuantumAI

def test_image_generation():
    """Test the integrated image generation system"""
    print("🧪 Testing Quantum Image Generation Integration")
    
    # Initialize the AI system with image generation enabled
    ai = EssentialQuantumAI(enable_image_generation=True)
    
    # Test text generation
    print("\n📝 Testing text generation...")
    text_response = ai.process_message("What is quantum computing?")
    print(f"Text response: {text_response.response_text[:100]}...")
    
    # Test image generation
    print("\n🎨 Testing image generation...")
    test_prompts = [
        "a serene mountain landscape",
        "abstract geometric patterns", 
        "futuristic cityscape at sunset"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🖼️ Generating image {i+1}/3: '{prompt}'")
        
        try:
            image_obj = ai.generate_image_only(prompt, method="quantum")
            if image_obj:
                filename = f"test_quantum_image_{i+1}.png"
                filepath = os.path.join("results", filename)
                
                # Ensure results directory exists
                os.makedirs("results", exist_ok=True)
                
                # Save the PIL Image object
                image_obj.save(filepath)
                
                print(f"✅ Image saved: {filepath}")
                print(f"   Size: {image_obj.size}")
            else:
                print("❌ No image data returned")
                
        except Exception as e:
            print(f"❌ Error generating image: {e}")
    
    print("\n✅ Integration test complete!")

if __name__ == "__main__":
    test_image_generation()