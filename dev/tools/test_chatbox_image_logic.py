#!/usr/bin/env python3
"""
Test the chatbox image generation logic directly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dev.tools.essential_quantum_ai import EssentialQuantumAI

def test_chatbox_image_logic():
    """Test the same logic the chatbox uses for image generation"""
    print("🧪 Testing Chatbox Image Generation Logic")
    
    # Initialize AI like the chatbox does
    ai = EssentialQuantumAI(enable_image_generation=True)
    
    # Test various image generation prompts
    test_prompts = [
        "generate image of a mountain landscape",
        "create an image of a futuristic city",
        "draw a cat sitting on a windowsill",
        "show me a sunset over the ocean",
        "make a picture of abstract art"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎨 Test {i}: '{prompt}'")
        
        # Test image detection logic
        def _is_image_generation_request(prompt: str) -> bool:
            image_keywords = [
                'generate image', 'create image', 'make image', 'draw', 'picture', 
                'visualize', 'show me', 'image of', 'create a visualization',
                'generate a picture', 'make a drawing', 'create art'
            ]
            prompt_lower = prompt.lower()
            return any(keyword in prompt_lower for keyword in image_keywords)
        
        def _extract_image_prompt(prompt: str) -> str:
            prompt_lower = prompt.lower()
            prefixes_to_remove = [
                'generate image of', 'create image of', 'make image of',
                'generate an image of', 'create an image of', 'make an image of',
                'draw', 'picture of', 'visualize', 'show me', 'image of',
                'generate a picture of', 'create a picture of', 'make a drawing of'
            ]
            
            cleaned_prompt = prompt
            for prefix in prefixes_to_remove:
                if prompt_lower.startswith(prefix):
                    cleaned_prompt = prompt[len(prefix):].strip()
                    break
            
            return cleaned_prompt if cleaned_prompt else prompt
        
        # Check if it's detected as an image request
        is_image_req = _is_image_generation_request(prompt)
        print(f"  Detected as image request: {is_image_req}")
        
        if is_image_req:
            # Extract the clean prompt
            clean_prompt = _extract_image_prompt(prompt)
            print(f"  Extracted prompt: '{clean_prompt}'")
            
            # Generate the image
            try:
                image = ai.generate_image_only(clean_prompt)
                if image:
                    # Save like chatbox would
                    import time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"chatbox_test_{i}_{timestamp}.png"
                    filepath = os.path.join("results", "generated_images", filename)
                    os.makedirs("results/generated_images", exist_ok=True)
                    image.save(filepath)
                    
                    print(f"  ✅ Image saved: {filepath}")
                    print(f"     Size: {image.size}")
                else:
                    print("  ❌ No image generated")
            except Exception as e:
                print(f"  ❌ Generation error: {e}")
        else:
            print("  ⏭️ Not detected as image request")
    
    print("\n✅ Chatbox image logic test complete!")

if __name__ == "__main__":
    test_chatbox_image_logic()