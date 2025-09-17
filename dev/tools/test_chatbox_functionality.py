#!/usr/bin/env python3
"""
Quick test script to verify the chatbox image generation functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dev.tools.essential_quantum_ai import EssentialQuantumAI

def test_chatbox_functionality():
    """Test the functionality that would be used by the chatbox"""
    print("📱 Testing Chatbox-style Integration")
    
    # Initialize AI like the chatbox does
    ai = EssentialQuantumAI(enable_image_generation=True)
    
    # Test a mixed request (text + image)
    print("\n🗨️ Testing mixed text/image request...")
    
    # First, process a text message
    text_msg = "Hello! Can you tell me about quantum computing?"
    response = ai.process_message(text_msg)
    print(f"Text Response: {response.response_text[:150]}...")
    
    # Then generate an image as if it was requested in chat
    print("\n🎨 Testing image generation request...")
    image_prompt = "a beautiful quantum computer visualization"
    try:
        image = ai.generate_image_only(image_prompt)
        if image:
            # Save like the chatbox would
            output_path = os.path.join("results", "chatbox_test_image.png")
            image.save(output_path)
            print(f"✅ Image generated and saved: {output_path}")
            print(f"   Image size: {image.size}")
        else:
            print("❌ No image generated")
    except Exception as e:
        print(f"❌ Image generation error: {e}")
    
    # Test status like the chatbox might
    print("\n📊 Testing status check...")
    status = ai.get_status()
    print("System Status:")
    print(f"  Engines: {status.get('engine_count', 0)}")
    print(f"  Parameters: {status.get('encoded_parameter_count', 0)}")
    if 'image_generation' in status:
        img_status = status['image_generation']
        print(f"  Image features: {img_status.get('total_encoded_features', 0)}")
        print(f"  Parameter sets: {img_status.get('parameter_sets', 0)}")
    
    print("\n✅ Chatbox functionality test complete!")

if __name__ == "__main__":
    test_chatbox_functionality()