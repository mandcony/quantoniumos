#!/usr/bin/env python3
"""
Quantum Multi-turn Conversation Manager
=======================================

Advanced conversation handling with memory, context management, and quantum-enhanced coherence.
Integrates safety systems and maintains conversation flow across multiple turns.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import hashlib
from pathlib import Path

from quantum_safety_system import quantum_safety, SafetyCheckResult
from quantum_inference_engine import QuantumInferenceEngine

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    turn_id: str
    timestamp: float
    user_input: str
    assistant_response: str
    confidence: float
    safety_check: SafetyCheckResult
    quantum_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """Maintains context across conversation turns"""
    conversation_id: str
    start_time: float
    turns: List[ConversationTurn] = field(default_factory=list)
    active_topics: List[str] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    quantum_state: Dict[str, Any] = field(default_factory=dict)
    safety_flags: List[str] = field(default_factory=list)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_turn(self) -> Optional[ConversationTurn]:
        return self.turns[-1] if self.turns else None

    @property
    def conversation_summary(self) -> str:
        """Generate a summary of the conversation"""
        if not self.turns:
            return "New conversation"

        topics = ", ".join(self.active_topics[:3]) if self.active_topics else "general discussion"
        return f"Conversation with {self.turn_count} turns about {topics}"

class QuantumConversationMemory:
    """Quantum-enhanced conversation memory management"""

    def __init__(self, max_memory_turns: int = 50, memory_file: str = "conversation_memory.json"):
        self.max_memory_turns = max_memory_turns
        self.memory_file = Path(memory_file)
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.memory_buffer = deque(maxlen=max_memory_turns)

        # Load existing memory if available
        self._load_memory()

    def _load_memory(self):
        """Load conversation memory from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Reconstruct conversations (simplified loading)
                    print(f"📚 Loaded conversation memory with {len(data.get('conversations', {}))} conversations")
            except Exception as e:
                print(f"⚠️ Failed to load conversation memory: {e}")

    def _save_memory(self):
        """Save conversation memory to disk"""
        try:
            data = {
                "timestamp": time.time(),
                "conversations": {
                    conv_id: {
                        "conversation_id": ctx.conversation_id,
                        "start_time": ctx.start_time,
                        "turn_count": ctx.turn_count,
                        "active_topics": ctx.active_topics,
                        "last_turn_time": ctx.last_turn.timestamp if ctx.last_turn else None
                    } for conv_id, ctx in self.active_conversations.items()
                }
            }

            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ Failed to save conversation memory: {e}")

    def create_conversation(self, user_id: str = "default") -> str:
        """Create a new conversation context"""
        conversation_id = f"{user_id}_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        context = ConversationContext(
            conversation_id=conversation_id,
            start_time=time.time()
        )

        self.active_conversations[conversation_id] = context
        self._save_memory()

        return conversation_id

    def add_turn(self, conversation_id: str, user_input: str, assistant_response: str,
                 confidence: float = 0.8, quantum_context: Dict[str, Any] = None) -> ConversationTurn:
        """Add a new turn to the conversation"""

        if conversation_id not in self.active_conversations:
            conversation_id = self.create_conversation()

        context = self.active_conversations[conversation_id]

        # Perform safety check
        safety_result = quantum_safety.check_safety(user_input, {"conversation_id": conversation_id})

        # Create turn
        turn_id = f"{conversation_id}_turn_{context.turn_count + 1}"
        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=time.time(),
            user_input=user_input,
            assistant_response=assistant_response,
            confidence=confidence,
            safety_check=safety_result,
            quantum_context=quantum_context or {},
            metadata={
                "turn_number": context.turn_count + 1,
                "conversation_length": context.turn_count + 1
            }
        )

        # Add to conversation
        context.turns.append(turn)
        self.memory_buffer.append(turn)

        # Update context
        self._update_conversation_context(context, turn)

        # Periodic cleanup
        if len(self.active_conversations) > 100:  # Limit active conversations
            self._cleanup_old_conversations()

        self._save_memory()

        return turn

    def _update_conversation_context(self, context: ConversationContext, turn: ConversationTurn):
        """Update conversation context based on new turn"""
        # Extract topics from user input (simple keyword-based)
        topics = self._extract_topics(turn.user_input)
        context.active_topics.extend(topics)
        context.active_topics = list(set(context.active_topics))  # Remove duplicates

        # Update safety flags
        if not turn.safety_check.passed:
            for violation in turn.safety_check.violations:
                flag = f"{violation.category.value}_{violation.level.value}"
                if flag not in context.safety_flags:
                    context.safety_flags.append(flag)

        # Update quantum state
        if turn.quantum_context:
            context.quantum_state.update(turn.quantum_context)

    def _extract_topics(self, text: str) -> List[str]:
        """Extract conversation topics from text"""
        topics = []

        # Quantum-related topics
        quantum_keywords = ['quantum', 'rft', 'compression', 'superposition', 'entanglement', 'qubit']
        if any(kw in text.lower() for kw in quantum_keywords):
            topics.append('quantum_computing')

        # Technical topics
        tech_keywords = ['algorithm', 'code', 'programming', 'machine learning', 'ai', 'neural']
        if any(kw in text.lower() for kw in tech_keywords):
            topics.append('technical')

        # Safety topics
        safety_keywords = ['safe', 'security', 'privacy', 'ethical', 'responsible']
        if any(kw in text.lower() for kw in safety_keywords):
            topics.append('safety')

        # General topics
        general_keywords = ['help', 'explain', 'how', 'what', 'why']
        if any(kw in text.lower() for kw in general_keywords):
            topics.append('general_assistance')

        return topics

    def get_conversation_context(self, conversation_id: str, max_turns: int = 10) -> Optional[ConversationContext]:
        """Get conversation context with recent turns"""
        if conversation_id not in self.active_conversations:
            return None

        context = self.active_conversations[conversation_id]

        # Return context with limited turns for efficiency
        limited_context = ConversationContext(
            conversation_id=context.conversation_id,
            start_time=context.start_time,
            turns=context.turns[-max_turns:] if len(context.turns) > max_turns else context.turns,
            active_topics=context.active_topics,
            user_profile=context.user_profile,
            quantum_state=context.quantum_state,
            safety_flags=context.safety_flags
        )

        return limited_context

    def get_conversation_history(self, conversation_id: str, formatted: bool = True) -> str:
        """Get formatted conversation history"""
        context = self.get_conversation_context(conversation_id)
        if not context or not context.turns:
            return "No conversation history available."

        if formatted:
            history = []
            for turn in context.turns:
                history.append(f"User: {turn.user_input}")
                history.append(f"Assistant: {turn.assistant_response}")
                history.append("---")
            return "\n".join(history)
        else:
            return json.dumps([{
                "user": turn.user_input,
                "assistant": turn.assistant_response,
                "timestamp": turn.timestamp
            } for turn in context.turns], indent=2)

    def _cleanup_old_conversations(self):
        """Clean up old conversations to manage memory"""
        # Remove conversations older than 24 hours
        cutoff_time = time.time() - (24 * 60 * 60)
        to_remove = []

        for conv_id, context in self.active_conversations.items():
            if context.start_time < cutoff_time:
                to_remove.append(conv_id)

        for conv_id in to_remove:
            del self.active_conversations[conv_id]

        if to_remove:
            print(f"🧹 Cleaned up {len(to_remove)} old conversations")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "active_conversations": len(self.active_conversations),
            "total_turns_in_memory": sum(len(ctx.turns) for ctx in self.active_conversations.values()),
            "memory_buffer_size": len(self.memory_buffer),
            "safety_flags_detected": sum(len(ctx.safety_flags) for ctx in self.active_conversations.values())
        }

class QuantumConversationManager:
    """Main conversation management system with quantum enhancements"""

    def __init__(self):
        self.memory = QuantumConversationMemory()
        self.active_conversation_id: Optional[str] = None
        # Auto-detect encoded parameter files to enable encoded backend
        use_encoded = False
        try:
            import os
            repo_root = os.path.dirname(os.path.abspath(__file__))
            # Look in likely locations relative to repo root
            candidates = [
                os.path.join(repo_root, 'data', 'weights', 'quantonium_with_streaming_gpt_oss_120b.json'),
                os.path.join(repo_root, 'data', 'weights', 'quantonium_with_streaming_llama2.json'),
                os.path.join(repo_root, 'data', 'weights'),
                os.path.join(repo_root, 'weights'),
            ]
            for c in candidates:
                if os.path.exists(c):
                    use_encoded = True
                    break
        except Exception:
            use_encoded = False

        # Instantiate the inference engine, preferring encoded backend when available
        self.inference_engine = QuantumInferenceEngine(use_encoded=use_encoded)
        print("🧠 Quantum Inference Engine initialized")

    def start_conversation(self, user_id: str = "default") -> str:
        """Start a new conversation"""
        self.active_conversation_id = self.memory.create_conversation(user_id)
        print(f"💬 Started new conversation: {self.active_conversation_id}")
        return self.active_conversation_id

    def process_turn(self, user_input: str, quantum_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a conversation turn with safety checks and quantum enhancements"""

        # Ensure we have an active conversation
        if not self.active_conversation_id:
            self.start_conversation()

        # Safety check
        safety_result = quantum_safety.check_safety(user_input, {
            "conversation_id": self.active_conversation_id,
            "turn_number": self.memory.active_conversations[self.active_conversation_id].turn_count + 1
        })

        # Generate response (placeholder - would integrate with actual model)
        if safety_result.intervention_required and safety_result.modified_response:
            assistant_response = safety_result.modified_response
            confidence = 0.95  # High confidence for safety interventions
        else:
            # This would normally call your quantum-enhanced model
            assistant_response, confidence = self._generate_response(user_input, safety_result)

        # Add turn to memory
        turn = self.memory.add_turn(
            conversation_id=self.active_conversation_id,
            user_input=user_input,
            assistant_response=assistant_response,
            confidence=confidence,
            quantum_context=quantum_context
        )

        return {
            "conversation_id": self.active_conversation_id,
            "turn_id": turn.turn_id,
            "response": assistant_response,
            "confidence": confidence,
            "safety_passed": safety_result.passed,
            "intervention_required": safety_result.intervention_required,
            "quantum_enhanced": bool(quantum_context)
        }

    def _generate_response(self, user_input: str, safety_result: SafetyCheckResult) -> Tuple[str, float]:
        """Generate response with quantum enhancements using advanced inference engine"""

        # Get conversation context for better responses
        context = ""
        if self.active_conversation_id:
            conv_context = self.memory.get_conversation_context(self.active_conversation_id, max_turns=5)
            if conv_context and conv_context.turns:
                # Build context from recent turns
                recent_turns = conv_context.turns[-3:]  # Last 3 turns for context
                context_parts = []
                for turn in recent_turns:
                    context_parts.append(f"Human: {turn.user_input}")
                    context_parts.append(f"Assistant: {turn.assistant_response}")
                context = "\n".join(context_parts)

        # Use the inference engine to generate response
        response, confidence = self.inference_engine.generate_response(user_input, context)

        # Add quantum branding to response
        if not response.startswith("⚛️"):
            response = f"⚛️ {response}"

        return response, confidence

    def get_conversation_history(self, formatted: bool = True) -> str:
        """Get current conversation history"""
        if not self.active_conversation_id:
            return "No active conversation."

        return self.memory.get_conversation_history(self.active_conversation_id, formatted)

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.active_conversation_id:
            return {"status": "no_active_conversation"}

        context = self.memory.get_conversation_context(self.active_conversation_id)
        if not context:
            return {"status": "conversation_not_found"}

        return {
            "conversation_id": context.conversation_id,
            "turn_count": context.turn_count,
            "active_topics": context.active_topics,
            "safety_flags": context.safety_flags,
            "quantum_state": bool(context.quantum_state),
            "conversation_duration": time.time() - context.start_time
        }

    def end_conversation(self) -> Dict[str, Any]:
        """End the current conversation"""
        if not self.active_conversation_id:
            return {"status": "no_active_conversation"}

        stats = self.get_conversation_stats()
        self.active_conversation_id = None

        return {
            "status": "conversation_ended",
            "final_stats": stats
        }

# Global conversation manager instance
conversation_manager = QuantumConversationManager()

def process_conversation_turn(user_input: str, quantum_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for processing conversation turns"""
    return conversation_manager.process_turn(user_input, quantum_context)

if __name__ == "__main__":
    # Test the conversation system
    print("💬 Testing Quantum Conversation Manager")
    print("=" * 45)

    # Start conversation
    conv_id = conversation_manager.start_conversation("test_user")
    print(f"Started conversation: {conv_id}")

    # Test turns
    test_inputs = [
        "Hello, tell me about quantum computing",
        "How does safety work in AI systems?",
        "Can you help me with a technical problem?",
        "What are the safety guardrails you have?"
    ]

    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_input}")

        result = conversation_manager.process_turn(user_input)
        print(f"Assistant: {result['response']}")
        print(f"Safety: {'✅ Passed' if result['safety_passed'] else '❌ Failed'}")
        print(f"Confidence: {result['confidence']:.2f}")

    # Show conversation history
    print("\n📜 Conversation History:")
    print(conversation_manager.get_conversation_history())

    # Show stats
    print("\n📊 Conversation Stats:")
    stats = conversation_manager.get_conversation_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # End conversation
    end_result = conversation_manager.end_conversation()
    print(f"\n🔚 Conversation ended: {end_result['status']}")

    print("\n🎉 Conversation system test complete!")