#!/usr/bin/env python3
"""
QuantoniumOS Multimodal AI Demo
Demonstrates text and image generation capabilities
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.inference.quantum_inference_engine import QuantumInferenceEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_text_generation():
    """Demo basic text generation"""
    print("[EMOJI] QuantoniumOS AI Demo - Text Generation")
    print("=" * 50)
    
    try:
        # Initialize the AI engine
        ai = QuantumInferenceEngine()
        
        # Test prompts
        prompts = [
            "What is quantum computing?",
            "Explain the golden ratio in nature",
            "How does QuantoniumOS work?"
        ]
        
        for prompt in prompts:
            print(f"\n[EMOJI] Human: {prompt}")
            response, confidence = ai.generate_response(prompt)
            print(f"[BOT] QuantumAssist (confidence: {confidence:.2f}): {response}")
            
    except Exception as e:
        print(f"FAIL Text generation demo failed: {e}")

def demo_image_generation():
    """Demo image generation capabilities"""
    print("\n[EMOJI] QuantoniumOS AI Demo - Image Generation")
    print("=" * 50)
    
    try:
        # Initialize AI with image generation
        ai = QuantumInferenceEngine(enable_image_generation=True)
        
        if not ai.is_image_generation_available():
            print("FAIL Image generation not available. Install: pip install diffusers")
            return
        
        # Test image prompts
        image_prompts = [
            "a nano banana in a futuristic laboratory",
            "quantum particle visualization with wave functions",
            "golden ratio spiral in a galaxy"
        ]
        
        for prompt in image_prompts:
            print(f"\n[EMOJI] Generating image: {prompt}")
            try:
                image_paths = ai.generate_image_only(
                    prompt, 
                    num_images=1,
                    enhancement_style="quantum"
                )
                print(f"OK Image saved: {image_paths[0]}")
            except Exception as e:
                print(f"WARNING Failed to generate image: {e}")
                
    except Exception as e:
        print(f"FAIL Image generation demo failed: {e}")

def demo_multimodal():
    """Demo combined text and image generation"""
    print("\n[EMOJI] QuantoniumOS AI Demo - Multimodal")
    print("=" * 50)
    
    try:
        ai = QuantumInferenceEngine(enable_image_generation=True)
        
        # Test multimodal prompts
        multimodal_prompts = [
            "Create an image of a nano banana and explain what nanotechnology is",
            "Show me quantum entanglement and describe how it works",
            "Visualize the golden ratio and tell me why it's important"
        ]
        
        for prompt in multimodal_prompts:
            print(f"\n[EMOJI] Multimodal request: {prompt}")
            
            result = ai.generate_multimodal_response(
                prompt, 
                include_image=True,
                image_params={"enhancement_style": "scientific"}
            )
            
            print(f"[EMOJI] Text Response: {result['text'][:200]}...")
            if result['image_paths']:
                print(f"[EMOJI] Images generated: {len(result['image_paths'])}")
                for path in result['image_paths']:
                    print(f"   [DIR] {path}")
            
    except Exception as e:
        print(f"FAIL Multimodal demo failed: {e}")

def interactive_mode():
    """Interactive chat with multimodal capabilities"""
    print("\n[EMOJI] QuantoniumOS Interactive AI")
    print("=" * 50)
    print("Commands:")
    print("  'image: <prompt>' - Generate only an image")
    print("  'both: <prompt>' - Generate text and image")  
    print("  'quit' - Exit")
    print("  Anything else - Generate text response")
    print()
    
    try:
        ai = QuantumInferenceEngine(enable_image_generation=True)
        capabilities = ai.get_capabilities()
        
        print(f"[CONFIG] Capabilities: {capabilities}")
        
        context = ""
        
        while True:
            user_input = input("\n[EMOJI] You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("[EMOJI] Goodbye!")
                break
                
            if user_input.startswith('image:'):
                prompt = user_input[6:].strip()
                if ai.is_image_generation_available():
                    try:
                        paths = ai.generate_image_only(prompt)
                        print(f"[EMOJI] Generated image: {paths[0]}")
                    except Exception as e:
                        print(f"FAIL Image generation failed: {e}")
                else:
                    print("FAIL Image generation not available")
                    
            elif user_input.startswith('both:'):
                prompt = user_input[5:].strip()
                result = ai.generate_multimodal_response(prompt, include_image=True)
                print(f"[BOT] {result['text']}")
                
            else:
                response, confidence = ai.generate_response(user_input, context)
                print(f"[BOT] QuantumAssist: {response}")
                
                # Update context for next response
                context += f"Human: {user_input}\nAssistant: {response}\n"
                
    except KeyboardInterrupt:
        print("\n[EMOJI] Goodbye!")
    except Exception as e:
        print(f"FAIL Interactive mode failed: {e}")

def main():
    """Run all demos"""
    print("[LAUNCH] QuantoniumOS AI System - Full Capabilities Demo")
    print("=" * 60)
    
    # Run demos
    demo_text_generation()
    demo_image_generation()
    demo_multimodal()
    
    # Ask if user wants interactive mode
    try:
        choice = input("\n[BOT] Start interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_mode()
    except KeyboardInterrupt:
        print("\n[EMOJI] Demo complete!")

if __name__ == "__main__":
    main()