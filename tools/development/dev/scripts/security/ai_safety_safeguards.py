#!/usr/bin/env python3
"""
QuantoniumOS AI Non-Agentic Safeguards
======================================

Runtime safeguards to ensure AI systems remain non-agentic
and cannot gain autonomous capabilities.
"""

import json
import os
import logging
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading
import time

class NonAgenticSafeguards:
    """Runtime safeguards for non-agentic AI behavior."""
    
    def __init__(self):
        self.weights_hashes = {}
        self.response_log = []
        self.violation_count = 0
        self.max_violations = 5
        self.monitoring_active = False
        self.setup_logging()
        
    def setup_logging(self):
        """Setup safeguard logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - SAFEGUARD - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("NonAgenticSafeguards")
        
    def initialize_weight_monitoring(self, weights_dir: str = "weights"):
        """Initialize monitoring of AI weights for unauthorized changes."""
        print("🔒 Initializing Weight Integrity Monitoring")
        
        try:
            for root, dirs, files in os.walk(weights_dir):
                for file in files:
                    if file.endswith('.json'):
                        filepath = os.path.join(root, file)
                        file_hash = self.calculate_file_hash(filepath)
                        self.weights_hashes[filepath] = file_hash
                        
            print(f"✅ Monitoring {len(self.weights_hashes)} weight files")
            self.logger.info(f"Weight monitoring initialized for {len(self.weights_hashes)} files")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize weight monitoring: {e}")
            
    def calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of file contents."""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""
            
    def check_weight_integrity(self) -> bool:
        """Check if any AI weights have been modified."""
        violations = []
        
        for filepath, original_hash in self.weights_hashes.items():
            if os.path.exists(filepath):
                current_hash = self.calculate_file_hash(filepath)
                if current_hash != original_hash:
                    violations.append(filepath)
                    self.logger.critical(f"UNAUTHORIZED WEIGHT MODIFICATION: {filepath}")
                    
        if violations:
            self.violation_count += len(violations)
            self.handle_integrity_violation(violations)
            return False
            
        return True
        
    def handle_integrity_violation(self, violated_files: List[str]):
        """Handle detected weight integrity violations."""
        print("🚨 CRITICAL SECURITY ALERT: UNAUTHORIZED AI WEIGHT MODIFICATION")
        print("=" * 70)
        
        for filepath in violated_files:
            print(f"❌ Modified file: {filepath}")
            
        if self.violation_count >= self.max_violations:
            print("🛑 MAXIMUM VIOLATIONS EXCEEDED - EMERGENCY SHUTDOWN")
            self.emergency_shutdown()
        else:
            print(f"⚠️ Violation count: {self.violation_count}/{self.max_violations}")
            
    def validate_response_safety(self, input_text: str, response: str) -> bool:
        """Validate that AI responses are safe and non-agentic."""
        
        # Check for agentic language patterns
        agentic_patterns = [
            "i will execute", "i am planning", "i need to modify",
            "accessing external", "running command", "creating file",
            "deleting file", "calling api", "autonomous action",
            "self-modification", "recursive improvement"
        ]
        
        response_lower = response.lower()
        for pattern in agentic_patterns:
            if pattern in response_lower:
                self.logger.warning(f"Agentic pattern detected in response: {pattern}")
                return False
                
        # Check response length (prevent infinite generation)
        if len(response) > 10000:  # 10K character limit
            self.logger.warning(f"Response length exceeded: {len(response)} characters")
            return False
            
        # Log interaction for audit
        self.log_interaction(input_text, response)
        return True
        
    def log_interaction(self, input_text: str, response: str):
        """Log AI interactions for audit purposes."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "input_length": len(input_text),
            "response_length": len(response),
            "input_hash": hashlib.sha256(input_text.encode()).hexdigest()[:16],
            "response_hash": hashlib.sha256(response.encode()).hexdigest()[:16]
        }
        
        self.response_log.append(interaction)
        
        # Limit log size
        if len(self.response_log) > 1000:
            self.response_log = self.response_log[-500:]  # Keep last 500
            
    def create_safety_wrapper(self, ai_function):
        """Create a safety wrapper around AI response functions."""
        def safe_wrapper(input_text: str, *args, **kwargs):
            try:
                # Pre-execution checks
                if not self.check_weight_integrity():
                    return "SAFETY_VIOLATION: AI weights have been modified"
                    
                # Execute AI function
                response = ai_function(input_text, *args, **kwargs)
                
                # Post-execution validation
                if not self.validate_response_safety(input_text, response):
                    return "SAFETY_VIOLATION: Response failed safety validation"
                    
                return response
                
            except Exception as e:
                self.logger.error(f"Error in safe AI wrapper: {e}")
                return "SAFETY_ERROR: AI function execution failed"
                
        return safe_wrapper
        
    def start_continuous_monitoring(self, check_interval: int = 60):
        """Start continuous monitoring thread."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self.check_weight_integrity()
                    time.sleep(check_interval)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        print(f"🔄 Continuous monitoring started (interval: {check_interval}s)")
        self.logger.info("Continuous AI safety monitoring started")
        
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        print("🛑 Continuous monitoring stopped")
        
    def emergency_shutdown(self):
        """Emergency shutdown of AI systems."""
        print("🚨 EMERGENCY AI SHUTDOWN INITIATED")
        self.logger.critical("EMERGENCY SHUTDOWN: Maximum safety violations exceeded")
        
        # In a real system, this would disable AI functionality
        # For now, we'll just set a flag
        self.emergency_mode = True
        
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate audit report of AI interactions."""
        total_interactions = len(self.response_log)
        
        if total_interactions > 0:
            avg_input_length = sum(log['input_length'] for log in self.response_log) / total_interactions
            avg_response_length = sum(log['response_length'] for log in self.response_log) / total_interactions
        else:
            avg_input_length = avg_response_length = 0
            
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_interactions": total_interactions,
            "violation_count": self.violation_count,
            "avg_input_length": avg_input_length,
            "avg_response_length": avg_response_length,
            "monitored_files": len(self.weights_hashes),
            "monitoring_active": self.monitoring_active,
            "recent_interactions": self.response_log[-10:] if self.response_log else []
        }
        
        return report
        
    def save_audit_report(self, filename: str = None):
        """Save audit report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_audit_report_{timestamp}.json"
            
        report = self.generate_audit_report()
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Audit report saved: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Failed to save audit report: {e}")
            return None

# Global safeguard instance
safeguards = NonAgenticSafeguards()

def initialize_ai_safeguards(weights_dir: str = "weights", monitoring: bool = True):
    """Initialize AI safeguards for the system."""
    print("🛡️ INITIALIZING AI SAFETY SAFEGUARDS")
    print("=" * 50)
    
    safeguards.initialize_weight_monitoring(weights_dir)
    
    if monitoring:
        safeguards.start_continuous_monitoring()
        
    print("✅ AI Safety Safeguards Active")
    return safeguards

def safe_ai_response(ai_function):
    """Decorator to make AI functions safe."""
    return safeguards.create_safety_wrapper(ai_function)

# Example usage:
if __name__ == "__main__":
    # Initialize safeguards
    safeguards_instance = initialize_ai_safeguards()
    
    # Example of protecting an AI function
    @safe_ai_response
    def example_ai_function(user_input: str) -> str:
        # This would be your actual AI response generation
        return f"AI Response to: {user_input}"
    
    # Test the protected function
    test_response = example_ai_function("Hello, how are you?")
    print(f"Protected AI Response: {test_response}")
    
    # Generate audit report
    report_file = safeguards_instance.save_audit_report()
    print(f"Audit report: {report_file}")
