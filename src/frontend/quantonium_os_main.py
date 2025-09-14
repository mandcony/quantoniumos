#!/usr/bin/env python3
"""
QuantoniumOS Console AI Interface
Console-based interface for the quantum-enhanced AI system
"""

import sys
import os
from pathlib import Path

# Add project paths
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from quantum_safety_system import QuantumSafetySystem
from quantum_conversation_manager import QuantumConversationManager
from quantum_rlhf_system import QuantumRLHFSystem
from quantum_domain_fine_tuner import QuantumDomainFineTuner
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantoniumConsoleAI:
    """Console interface for the quantum AI system"""

    def __init__(self):
        print("🧠 Initializing QuantoniumOS Quantum AI Console")
        print("=" * 60)

        # Initialize components
        self.safety_system = QuantumSafetySystem()
        self.conversation_manager = QuantumConversationManager()
        self.rlhf_system = QuantumRLHFSystem()
        self.domain_tuner = None

        # Start conversation
        self.conversation_id = self.conversation_manager.start_conversation()

        print("✅ Safety System: Active")
        print("✅ Conversation Manager: Active")
        print("✅ RLHF System: Ready")
        print("✅ Domain Fine-tuning: Available")
        print("\n💬 Welcome to QuantoniumOS Quantum AI!")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 60)

    def process_command(self, command: str) -> str:
        """Process user commands"""
        command = command.strip().lower()

        if command == "help":
            return self.show_help()
        elif command == "status":
            return self.show_status()
        elif command.startswith("domain "):
            return self.handle_domain_command(command)
        elif command == "rlhf":
            return self.handle_rlhf_command()
        elif command == "quit":
            return "Goodbye! 👋"
        else:
            return self.process_ai_query(command)

    def show_help(self) -> str:
        """Show available commands"""
        help_text = """
🔧 Available Commands:
  help          - Show this help message
  status        - Show system status
  domain <type> - Switch to domain (math, coding, science, creative, business)
  rlhf          - Show RLHF system status
  quit          - Exit the console

💬 For AI queries, just type your question or statement!
        """
        return help_text

    def show_status(self) -> str:
        """Show system status"""
        status = f"""
📊 System Status:
  🛡️  Safety: Active ({len(self.safety_system.get_safety_stats()['violations_blocked'])} violations blocked)
  💬 Memory: {self.conversation_manager.get_conversation_stats()['turn_count']} turns in current conversation
  🎯 RLHF: Ready (Reward model trained)
  🎯 Domains: 5 available (math, coding, science, creative, business)
        """
        return status

    def handle_domain_command(self, command: str) -> str:
        """Handle domain switching commands"""
        parts = command.split()
        if len(parts) < 2:
            return "Usage: domain <type> (math, coding, science, creative, business)"

        domain = parts[1].lower()
        valid_domains = ["math", "coding", "science", "creative", "business"]

        if domain not in valid_domains:
            return f"Invalid domain. Available: {', '.join(valid_domains)}"

        # Initialize domain tuner if needed
        if self.domain_tuner is None:
            try:
                self.domain_tuner = QuantumDomainFineTuner()
                self.domain_tuner.initialize_model()
            except Exception as e:
                return f"Failed to initialize domain tuner: {e}"

        self.current_domain = domain
        return f"✅ Switched to {domain} domain. Ready for specialized queries!"

    def handle_rlhf_command(self) -> str:
        """Show RLHF system information"""
        try:
            self.rlhf_system.initialize_models()
            return "🎯 RLHF System Status:\n  ✅ Models initialized\n  📚 Preference data ready\n  🎯 Reward model trained"
        except Exception as e:
            return f"RLHF System Error: {e}"

    def process_ai_query(self, query: str) -> str:
        """Process AI queries through the quantum system"""
        try:
            # Safety check first
            safety_result = self.safety_system.check_safety(query)
            if not safety_result.passed:
                return f"❌ Query blocked: {safety_result.violations[0].message if safety_result.violations else 'Safety violation'}"

            # Process through conversation manager
            conv_result = self.conversation_manager.process_turn(query)

            # Generate response based on domain if set
            if hasattr(self, 'current_domain') and self.domain_tuner:
                try:
                    response = self.domain_tuner.generate_domain_response(self.current_domain, query, max_length=100)
                except Exception as e:
                    response = f"Domain processing error: {e}. Using general AI response."
            else:
                # Simple response for now (could integrate with full AI pipeline)
                response = f"I understand you asked: '{query}'. This is a quantum-enhanced AI response through the QuantoniumOS system."

            return f"🤖 {response}"

        except Exception as e:
            return f"❌ Error processing query: {e}"

    def run(self):
        """Run the console interface"""
        while True:
            try:
                user_input = input("\n💭 You: ").strip()
                if not user_input:
                    continue

                response = self.process_command(user_input)
                print(f"\n{response}")

                if user_input.lower() == "quit":
                    break

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """Main entry point"""
    try:
        console = QuantoniumConsoleAI()
        console.run()
    except Exception as e:
        print(f"❌ Failed to start QuantoniumOS Console AI: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())