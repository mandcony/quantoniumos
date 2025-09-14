#!/usr/bin/env python3
"""
Test the conversation trainer response generation to debug truncation issues.
"""

import sys
import os
sys.path.append('tools')

try:
    from quantum_enhanced_conversation_trainer import QuantumEnhancedConversationTrainer
    
    trainer = QuantumEnhancedConversationTrainer()
    
    # Test with a simple prompt
    print("=== Testing Quantum Enhanced Conversation Trainer ===")
    
    test_prompt = "explain quantum computing"
    print(f"\nPrompt: {test_prompt}")
    print("-" * 50)
    
    response = trainer.process_message(test_prompt, "test_context")
    
    print(f"Response Text Length: {len(response.response_text)}")
    print(f"Confidence: {response.confidence}")
    print(f"Domain: {response.context.domain}")
    print(f"Quantum Coherence: {response.quantum_coherence}")
    print("-" * 50)
    print("Full Response:")
    print(response.response_text)
    print("-" * 50)
    
    if response.suggested_followups:
        print("Suggested Follow-ups:")
        for followup in response.suggested_followups:
            print(f"  • {followup}")
    
except Exception as e:
    print(f"Error testing trainer: {e}")
    
    # Test fallback pattern matching
    print("\n=== Testing Fallback Pattern Matching ===")
    
    sys.path.append('apps')
    from qshll_chatbox import Chatbox
    
    # Create a mock chatbox to test pattern fallback
    class MockChatbox:
        def __init__(self):
            self._conversation_examples = []
            self._enhanced_patterns = {}
        
        def _pattern_fallback_reply(self, prompt):
            # Copy the method from the chatbox
            prompt_lower = prompt.lower().strip()
            
            if any(explain in prompt_lower for explain in ['explain', 'how does', 'what is', 'tell me about']):
                return """🧠 I'd be happy to provide a comprehensive explanation! 

**My Approach:**
I'll break this down systematically, covering the fundamental concepts, technical details, and practical implications. I can adjust the depth based on your background and interests.

**What I can cover:**
- Core principles and foundational concepts
- Mathematical/technical formulations 
- Real-world applications and examples
- Current research and future directions
- Step-by-step implementation guidance

**Let me know:**
- Your current familiarity with the topic
- Whether you prefer theoretical depth or practical focus
- Any specific aspects you're most curious about

What specific aspect would you like me to start with?""", 0.90
            
            return "Default response", 0.5
    
    mock_chatbox = MockChatbox()
    response, confidence = mock_chatbox._pattern_fallback_reply("explain quantum computing")
    
    print(f"\nFallback Response Length: {len(response)}")
    print(f"Confidence: {confidence}")
    print("-" * 50)
    print("Fallback Response:")
    print(response)
