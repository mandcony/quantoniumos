#!/usr/bin/env python3
"""
QuantoniumOS Encoded-Only Conversation Trainer
==============================================

This trainer ONLY loads the essential encoded parameters:
- 120B GPT-OSS quantum states (14,221 states)
- 7B Llama2 quantum parameters (2,000,006 parameters)
- Quantum compression from ASSEMBLY kernel

NO unnecessary engines. Direct parameter access only.
"""

import os
import sys

# Add path for the encoded AI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "tools"))

from minimal_encoded_ai_trainer import EncodedParameterAI

class EncodedOnlyConversationTrainer:
    """
    Conversation trainer using ONLY encoded 120B+7B parameters.
    No engine loading, no fallbacks - just direct parameter access.
    """
    
    def __init__(self):
        print("🎯 Initializing Encoded-Only Conversation Trainer...")
        self.ai = EncodedParameterAI()
        print(f"✅ Ready with {self.ai.parameter_count:,} encoded parameters")
    
    def process_message(self, message: str):
        """Process message using only encoded parameters"""
        return self.ai.process_message(message)
    
    def get_status(self):
        """Get trainer status"""
        status = self.ai.get_status()
        status['conversation_trainer'] = 'encoded_only'
        return status

# Test the conversation trainer
if __name__ == "__main__":
    print("🧪 Testing Encoded-Only Conversation Trainer...")
    
    try:
        trainer = EncodedOnlyConversationTrainer()
        
        # Test conversation
        messages = [
            "Hello, how are you?",
            "What can you do with quantum parameters?",
            "Explain your capabilities"
        ]
        
        for msg in messages:
            print(f"\n💬 User: {msg}")
            response = trainer.process_message(msg)
            print(f"🤖 AI: {response.response_text}")
            print(f"   Confidence: {response.confidence:.2f}")
        
        # Show status
        print(f"\n📊 Status: {trainer.get_status()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
