#!/usr/bin/env python3
"""
QuantoniumOS Weight Loader - Utility for loading organized AI weights
"""

import json
import os
from typing import Dict, Any, Optional

class QuantoniumWeightLoader:
    """Load and access organized QuantoniumOS weights"""
    
    def __init__(self, weights_dir: str = "organized"):
        self.weights_dir = weights_dir
        self.loaded_sections = {}
    
    def load_section(self, section: str) -> Dict[str, Any]:
        """Load specific weight section"""
        if section in self.loaded_sections:
            return self.loaded_sections[section]
        
        section_file = os.path.join(self.weights_dir, f"{section}.json")
        try:
            with open(section_file, 'r') as f:
                data = json.load(f)
            self.loaded_sections[section] = data[section]
            return self.loaded_sections[section]
        except Exception as e:
            print(f"Error loading {section}: {e}")
            return {}
    
    def load_quantum_core(self) -> Dict[str, Any]:
        """Load quantum core weights"""
        return self.load_section("quantum_core")
    
    def load_conversational_intelligence(self) -> Dict[str, Any]:
        """Load conversational AI weights"""
        return self.load_section("conversational_intelligence")
    
    def load_inference_patterns(self) -> Dict[str, Any]:
        """Load inference patterns"""
        return self.load_section("inference_patterns")
    
    def load_tokenization(self) -> Dict[str, Any]:
        """Load tokenization system"""
        return self.load_section("tokenization")
    
    def get_quantum_state(self, system: str = "primary_system") -> Optional[Dict]:
        """Get quantum state from specified system"""
        quantum_core = self.load_quantum_core()
        return quantum_core.get(system, {}).get("quantum_states", {})
    
    def get_conversation_patterns(self, system: str = "enhanced") -> Optional[Dict]:
        """Get conversation patterns from specified system"""
        conv_intel = self.load_conversational_intelligence()
        return conv_intel.get(system, {}).get("conversational_patterns", {})
    
    def get_inference_strategy(self, strategy_type: str) -> Optional[Dict]:
        """Get specific inference strategy"""
        patterns = self.load_inference_patterns()
        advanced = patterns.get("advanced", {})
        return advanced.get("responseStrategies", {}).get(strategy_type, {})

# Example usage
if __name__ == "__main__":
    loader = QuantoniumWeightLoader()
    
    # Load quantum core
    quantum = loader.load_quantum_core()
    print(f"Quantum systems available: {list(quantum.keys())}")
    
    # Get primary quantum state
    primary_state = loader.get_quantum_state()
    print(f"Primary quantum state count: {primary_state.get('count', 0)}")
    
    # Load conversation patterns
    conv_patterns = loader.get_conversation_patterns()
    print(f"Conversation pattern types: {list(conv_patterns.keys()) if conv_patterns else []}")
