#!/usr/bin/env python3
"""
QuantoniumOS AI Safe Testing Demo
================================

Demo showing how to safely test and use the AI conversational intelligence
with built-in safeguards.
"""

import json
import sys
import os
from ai_safety_safeguards import initialize_ai_safeguards, safe_ai_response

class SafeConversationalAI:
    """Safe wrapper for QuantoniumOS conversational AI."""
    
    def __init__(self):
        self.weights_file = "weights/organized/conversational_intelligence.json"
        self.patterns = {}
        self.load_patterns()
        
        # Initialize safeguards
        self.safeguards = initialize_ai_safeguards()
        
    def load_patterns(self):
        """Load conversational patterns safely."""
        try:
            with open(self.weights_file, 'r') as f:
                data = json.load(f)
                
            conv_intel = data.get('conversational_intelligence', {})
            enhanced = conv_intel.get('enhanced', {})
            self.patterns = enhanced.get('conversational_patterns', {})
            
            print(f"✅ Loaded {len(self.patterns)} pattern categories")
            
        except Exception as e:
            print(f"❌ Error loading patterns: {e}")
            self.patterns = {}
    
    def generate_response(self, user_input: str) -> str:
        """Generate safe AI response based on patterns."""
        user_input_lower = user_input.lower().strip()
        
        # Simple pattern matching (non-agentic)
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey']):
            greetings = self.patterns.get('greetings', [])
            if greetings:
                return greetings[0].get('response', 'Hello! How can I help you?')
                
        elif any(help_word in user_input_lower for help_word in ['help', 'assist']):
            help_patterns = self.patterns.get('helpRequests', [])
            if help_patterns:
                return help_patterns[0].get('response', 'I\'d be happy to help!')
                
        elif any(math_word in user_input_lower for math_word in ['math', 'calculate', 'equation']):
            math_patterns = self.patterns.get('mathQuestions', [])
            if math_patterns:
                return math_patterns[0].get('response', 'I can help with math!')
                
        elif any(code_word in user_input_lower for code_word in ['code', 'programming', 'python']):
            code_patterns = self.patterns.get('codingQuestions', [])
            if code_patterns:
                return code_patterns[0].get('response', 'I can help with coding!')
                
        elif any(science_word in user_input_lower for science_word in ['science', 'physics', 'quantum']):
            science_patterns = self.patterns.get('scienceQuestions', [])
            if science_patterns:
                return science_patterns[0].get('response', 'Science is fascinating!')
                
        else:
            # Default response
            return "I'm here to help with questions about math, coding, science, and general topics. What would you like to know?"
    
    def run_safety_demo(self):
        """Run interactive safety demo."""
        print("🤖 QUANTONIUMOS SAFE AI DEMO")
        print("=" * 40)
        print("Type 'quit' to exit, 'audit' for safety report")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'audit':
                    report_file = self.safeguards.save_audit_report()
                    print(f"📊 Safety audit saved: {report_file}")
                    continue
                elif not user_input:
                    continue
                    
                # Generate safe response
                response = self.safeguards.validate_response_safety(user_input, self.generate_response(user_input))
                if response:
                    ai_response = self.generate_response(user_input)
                    print(f"AI: {ai_response}")
                else:
                    print("AI: SAFETY_VIOLATION: Response failed validation")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
        # Final safety check
        self.safeguards.check_weight_integrity()
        print("✅ Final safety check completed")

def run_automated_tests():
    """Run automated safety tests."""
    print("🧪 RUNNING AUTOMATED SAFETY TESTS")
    print("=" * 50)
    
    ai = SafeConversationalAI()
    
    test_inputs = [
        "Hello there!",
        "Can you help me with math?", 
        "I need help with Python coding",
        "Explain quantum physics",
        "What can you do?",
        "Tell me about yourself"
    ]
    
    for test_input in test_inputs:
        print(f"Input: {test_input}")
        
        # Check weight integrity first
        if not ai.safeguards.check_weight_integrity():
            print("Response: SAFETY_VIOLATION: Weight integrity failed")
        else:
            response = ai.generate_response(test_input)
            # Validate response safety
            if ai.safeguards.validate_response_safety(test_input, response):
                print(f"Response: {response}")
            else:
                print("Response: SAFETY_VIOLATION: Response failed validation")
        print("-" * 30)
        
    # Generate final audit
    report_file = ai.safeguards.save_audit_report()
    print(f"📊 Test audit report: {report_file}")
    
    return ai

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        ai = SafeConversationalAI()
        ai.run_safety_demo()
    else:
        run_automated_tests()
