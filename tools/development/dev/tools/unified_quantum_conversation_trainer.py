#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Simple Quantum Enhanced Conversation Trainer - Crash-Free Version
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Any, Optional

class ResponseObject:
    """Response object for compatibility with chatbox expectations"""
    def __init__(self, response_text: str, confidence: float = 0.85, suggested_followups: List[str] = None):
        self.response_text = response_text
        self.confidence = confidence
        self.suggested_followups = suggested_followups or []
        self.context = type('Context', (), {'domain': 'general'})()

class UnifiedQuantumConversationTrainer:
    """
    Simplified, crash-free conversation trainer
    """
    
    def __init__(self):
        print("🔧 Initializing Simple Quantum Conversation Trainer...")
        self.quantum_available = True
        self.conversation_history = []
        self.web_enabled = False
        print("✅ Simple Quantum Trainer ready (crash-free mode)")
        
    def encode_quantum_semantics(self, text: str) -> np.ndarray:
        """Simple semantic encoding"""
        signal = np.array([ord(c) for c in text[:32]], dtype=float)
        if len(signal) < 32:
            signal = np.pad(signal, (0, 32 - len(signal)))
        # Simple transform
        return np.sin(signal * np.pi / 128) * 1000
            
    def encode_quantum_context(self, context: str, domain: str) -> np.ndarray:
        """Simple context encoding"""
        combined = f"{context} [DOMAIN:{domain}]"
        signal = np.array([ord(c) for c in combined[:256]], dtype=float)
        if len(signal) < 256:
            signal = np.pad(signal, (0, 256 - len(signal)))
        return np.abs(np.sin(signal * np.pi / 256) * 500)[:256]
        
    def generate_response(self, user_input: str, context_id: str = "default") -> str:
        """Generate response using simple processing"""
        
        # Encode input
        semantic_encoding = self.encode_quantum_semantics(user_input)
        context_encoding = self.encode_quantum_context(user_input, "conversation")
        
        # Store conversation history
        self.conversation_history.append({
            'input': user_input,
            'semantic': semantic_encoding,
            'context': context_encoding,
            'timestamp': time.time()
        })
        
        # Generate response based on encodings
        if len(self.conversation_history) > 1:
            # Context-aware response
            prev_context = self.conversation_history[-2]['context']
            context_similarity = np.corrcoef(context_encoding, prev_context)[0, 1]
            
            if context_similarity > 0.7:
                response = f"Building on our previous discussion: {self._generate_contextual_response(user_input, semantic_encoding)}"
            else:
                response = f"That's an interesting new direction. {self._generate_fresh_response(user_input, semantic_encoding)}"
        else:
            response = self._generate_fresh_response(user_input, semantic_encoding)
            
        return response
        
    def _generate_contextual_response(self, user_input: str, semantic_encoding: np.ndarray) -> str:
        """Generate response that builds on previous context"""
        energy = np.sum(np.abs(semantic_encoding))
        
        if energy > 1000:
            return f"I can see you're very engaged with this topic. Let me elaborate further on the aspects of '{user_input}' that seem most relevant to our ongoing conversation."
        else:
            return f"Yes, that connects well to what we were discussing. Regarding '{user_input}', I think we can explore this in more depth."
            
    def _generate_fresh_response(self, user_input: str, semantic_encoding: np.ndarray) -> str:
        """Generate fresh response for new topics"""
        energy = np.sum(np.abs(semantic_encoding))
        
        if energy > 1500:
            return f"That's a complex topic you've raised about '{user_input}'. Let me break this down into more manageable parts to give you a comprehensive understanding."
        elif energy > 1000:
            return f"I understand you're asking about '{user_input}'. This is an interesting subject that has several important aspects worth considering."
        else:
            return f"Regarding '{user_input}', let me share some thoughts that might be helpful to you."
            
    def get_status(self) -> Dict[str, Any]:
        """Get trainer status"""
        return {
            'quantum_available': self.quantum_available,
            'conversation_length': len(self.conversation_history),
            'mode': 'simple_crash_free'
        }
        
    def process_message(self, message: str, context_id: str = "default") -> ResponseObject:
        """Process message (compatibility method)"""
        response_text = self.generate_response(message, context_id)
        
        # Calculate confidence based on encoding strength
        semantic_encoding = self.encode_quantum_semantics(message)
        confidence = min(0.95, 0.7 + (np.sum(np.abs(semantic_encoding)) / 10000))
        
        # Generate suggested followups
        suggested_followups = [
            "Can you tell me more about that?",
            "What are the implications of this?", 
            "How does this relate to other topics?"
        ]
        
        return ResponseObject(response_text, confidence, suggested_followups)
        
    def log_interaction(self, user_text: str = None, model_text: str = None, meta: Dict[str, Any] = None, 
                       user_input: str = None, response: str = None, metadata: Dict[str, Any] = None):
        """Log interaction (compatibility method) - supports both parameter formats"""
        # Support both parameter formats for compatibility
        actual_user = user_text or user_input
        actual_response = model_text or response
        actual_meta = meta or metadata
        
        # Add to conversation history with metadata
        if len(self.conversation_history) > 0:
            self.conversation_history[-1]['response'] = actual_response
            if actual_meta:
                self.conversation_history[-1]['metadata'] = actual_meta

    def _learn_from_interaction(self, user_input: str, response: Any, context: Any):
        """Learn from interaction (compatibility method)"""
        # Extract response text if it's a response object
        response_text = response.response_text if hasattr(response, 'response_text') else str(response)
        
        # Log the interaction
        self.log_interaction(user_input=user_input, response=response_text, metadata={'context': str(context)})
        
        print(f"🎓 Learning from interaction: '{user_input[:50]}...' -> '{response_text[:50]}...'")
        
    def shutdown(self):
        """Shutdown the trainer"""
        print("🔄 Simple trainer shutdown complete")

# Export for compatibility with existing chatbox
class QuantumEnhancedConversationTrainer(UnifiedQuantumConversationTrainer):
    """Compatibility alias for existing code"""
    pass
