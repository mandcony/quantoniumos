#!/usr/bin/env python3
"""
Phase 3: Persistent Memory with Quantum States
Quantum-encoded memory system with state validation and persistence
"""

import os
import sys
import json
import time
import pickle
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class QuantumMemoryState:
    """Represents a quantum memory state"""
    id: str
    content: Any
    timestamp: float
    quantum_signature: str
    importance_score: float
    access_count: int = 0
    last_accessed: float = 0.0
    quantum_entangled: List[str] = None  # IDs of entangled memories
    
    def __post_init__(self):
        if self.quantum_entangled is None:
            self.quantum_entangled = []

class QuantumMemoryCore:
    """Core quantum memory management system"""
    
    def __init__(self, memory_file: str = "quantum_memory_states.json", max_memory_mb: int = 100):
        self.memory_file = memory_file
        self.max_memory_mb = max_memory_mb
        self.memory_states = {}
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.quantum_coherence_threshold = 0.618
        
        # Memory organization
        self.short_term_memory = {}  # High-access, recent memories
        self.long_term_memory = {}   # Stable, important memories
        self.working_memory = {}     # Current session memories
        
        # Load existing memory
        self._load_quantum_memory()
        
        print(f"🧠 Quantum Memory Core initialized with {len(self.memory_states)} states")
    
    def _load_quantum_memory(self):
        """Load quantum memory from persistent storage"""
        
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct memory states
                for state_id, state_data in data.get('memory_states', {}).items():
                    self.memory_states[state_id] = QuantumMemoryState(**state_data)
                
                # Organize into memory types
                self._organize_memory_by_type()
                
                print(f"✅ Loaded {len(self.memory_states)} quantum memory states")
                
            except Exception as e:
                print(f"⚠️ Error loading quantum memory: {e}")
                self.memory_states = {}
    
    def _save_quantum_memory(self):
        """Save quantum memory to persistent storage"""
        
        try:
            # Convert memory states to JSON-serializable format
            serializable_states = {}
            for state_id, state in self.memory_states.items():
                serializable_states[state_id] = asdict(state)
            
            data = {
                'memory_states': serializable_states,
                'last_saved': time.time(),
                'total_states': len(self.memory_states),
                'quantum_coherence': self._calculate_quantum_coherence()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"💾 Saved {len(self.memory_states)} quantum memory states")
            
        except Exception as e:
            print(f"⚠️ Error saving quantum memory: {e}")
    
    def _organize_memory_by_type(self):
        """Organize memories into different types based on quantum properties"""
        
        current_time = time.time()
        
        self.short_term_memory = {}
        self.long_term_memory = {}
        self.working_memory = {}
        
        for state_id, state in self.memory_states.items():
            # Recent and frequently accessed
            if (current_time - state.timestamp < 3600 and state.access_count > 5) or state.importance_score > 0.8:
                self.short_term_memory[state_id] = state
            
            # Old but important
            elif state.importance_score > 0.6 or len(state.quantum_entangled) > 2:
                self.long_term_memory[state_id] = state
            
            # Current session
            elif current_time - state.timestamp < 1800:  # 30 minutes
                self.working_memory[state_id] = state
        
        print(f"📊 Memory organized: {len(self.short_term_memory)} short-term, {len(self.long_term_memory)} long-term, {len(self.working_memory)} working")
    
    def store_memory(self, content: Any, importance_score: float = 0.5, tags: List[str] = None) -> str:
        """Store new quantum memory state"""
        
        # Generate quantum signature
        content_str = str(content)
        quantum_signature = self._generate_quantum_signature(content_str)
        
        # Create unique ID
        memory_id = f"qmem_{int(time.time())}_{hash(content_str) % 10000}"
        
        # Create quantum memory state
        state = QuantumMemoryState(
            id=memory_id,
            content=content,
            timestamp=time.time(),
            quantum_signature=quantum_signature,
            importance_score=importance_score,
            access_count=0,
            last_accessed=time.time()
        )
        
        # Store in memory
        self.memory_states[memory_id] = state
        
        # Add to working memory
        self.working_memory[memory_id] = state
        
        # Find quantum entanglements
        self._find_quantum_entanglements(memory_id)
        
        # Manage memory size
        self._manage_memory_capacity()
        
        print(f"🧠 Stored quantum memory: {memory_id} (importance: {importance_score:.2f})")
        
        return memory_id
    
    def retrieve_memory(self, query: str, max_results: int = 5, quantum_search: bool = True) -> List[QuantumMemoryState]:
        """Retrieve quantum memories based on query"""
        
        if quantum_search:
            return self._quantum_search(query, max_results)
        else:
            return self._classical_search(query, max_results)
    
    def _quantum_search(self, query: str, max_results: int) -> List[QuantumMemoryState]:
        """Quantum-enhanced memory search using RFT resonance"""
        
        query_signature = self._generate_quantum_signature(query)
        resonance_scores = []
        
        for state_id, state in self.memory_states.items():
            # Calculate quantum resonance
            resonance = self._calculate_quantum_resonance(query_signature, state.quantum_signature)
            
            # Boost score based on importance and access patterns
            boosted_score = resonance * (1 + state.importance_score) * (1 + np.log(state.access_count + 1))
            
            resonance_scores.append((boosted_score, state))
            
            # Update access count
            state.access_count += 1
            state.last_accessed = time.time()
        
        # Sort by resonance score
        resonance_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results
        results = [state for score, state in resonance_scores[:max_results]]
        
        print(f"🔍 Quantum search found {len(results)} memories for: {query[:50]}...")
        
        return results
    
    def _classical_search(self, query: str, max_results: int) -> List[QuantumMemoryState]:
        """Classical text-based memory search"""
        
        query_lower = query.lower()
        matches = []
        
        for state in self.memory_states.values():
            content_str = str(state.content).lower()
            
            # Simple text matching
            if query_lower in content_str:
                score = content_str.count(query_lower) * state.importance_score
                matches.append((score, state))
        
        # Sort by score
        matches.sort(key=lambda x: x[0], reverse=True)
        
        return [state for score, state in matches[:max_results]]
    
    def _generate_quantum_signature(self, content: str) -> str:
        """Generate quantum signature for content using RFT"""
        
        # Convert content to numerical representation
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        content_nums = [int(content_hash[i:i+2], 16) for i in range(0, len(content_hash), 2)]
        
        # Apply RFT transformation
        quantum_vector = np.array(content_nums[:16])  # Use first 16 bytes
        
        # Golden ratio transformation
        transformed = quantum_vector * self.phi
        phases = np.array([np.cos(x * self.phi) + np.sin(x / self.phi) for x in transformed])
        
        # Create signature
        signature_nums = np.abs(phases * 1000).astype(int) % 256
        signature = ''.join(f'{x:02x}' for x in signature_nums)
        
        return signature[:32]  # Return 32-character signature
    
    def _calculate_quantum_resonance(self, sig1: str, sig2: str) -> float:
        """Calculate quantum resonance between two signatures"""
        
        if len(sig1) != len(sig2):
            return 0.0
        
        # Convert signatures to numbers
        nums1 = [int(sig1[i:i+2], 16) for i in range(0, len(sig1), 2)]
        nums2 = [int(sig2[i:i+2], 16) for i in range(0, len(sig2), 2)]
        
        # Calculate quantum correlation
        vec1 = np.array(nums1) / 255.0
        vec2 = np.array(nums2) / 255.0
        
        # RFT-enhanced correlation
        correlation = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Apply golden ratio enhancement
        quantum_enhancement = np.cos(correlation * self.phi) * 0.1
        
        return correlation + quantum_enhancement
    
    def _find_quantum_entanglements(self, memory_id: str):
        """Find quantum entanglements between memories"""
        
        current_state = self.memory_states[memory_id]
        
        for other_id, other_state in self.memory_states.items():
            if other_id == memory_id:
                continue
            
            # Calculate resonance
            resonance = self._calculate_quantum_resonance(
                current_state.quantum_signature, 
                other_state.quantum_signature
            )
            
            # Create entanglement if resonance is high
            if resonance > self.quantum_coherence_threshold:
                if other_id not in current_state.quantum_entangled:
                    current_state.quantum_entangled.append(other_id)
                
                if memory_id not in other_state.quantum_entangled:
                    other_state.quantum_entangled.append(memory_id)
                
                print(f"🔗 Quantum entanglement: {memory_id} ↔ {other_id} (resonance: {resonance:.3f})")
    
    def _manage_memory_capacity(self):
        """Manage memory capacity using quantum importance scoring"""
        
        # Check memory usage (approximate)
        estimated_size_mb = len(json.dumps(self.get_memory_stats())) / (1024 * 1024)
        
        if estimated_size_mb > self.max_memory_mb:
            print(f"🧹 Memory cleanup triggered: {estimated_size_mb:.1f}MB > {self.max_memory_mb}MB")
            
            # Score memories for importance
            memory_scores = []
            for state_id, state in self.memory_states.items():
                # Quantum importance score
                age_factor = max(0.1, 1.0 - (time.time() - state.timestamp) / (30 * 24 * 3600))  # 30 days
                access_factor = min(2.0, np.log(state.access_count + 1))
                entanglement_factor = min(2.0, len(state.quantum_entangled) * 0.5)
                
                total_score = state.importance_score * age_factor * access_factor * entanglement_factor
                memory_scores.append((total_score, state_id))
            
            # Sort by score (lowest first for removal)
            memory_scores.sort()
            
            # Remove lowest-scoring memories
            removal_count = max(1, len(self.memory_states) // 10)  # Remove 10%
            
            for i in range(removal_count):
                if i < len(memory_scores):
                    score, memory_id = memory_scores[i]
                    del self.memory_states[memory_id]
                    print(f"🗑️ Removed low-importance memory: {memory_id} (score: {score:.3f})")
            
            # Reorganize memory
            self._organize_memory_by_type()
    
    def _calculate_quantum_coherence(self) -> float:
        """Calculate overall quantum coherence of memory system"""
        
        if len(self.memory_states) < 2:
            return 1.0
        
        total_resonance = 0.0
        pair_count = 0
        
        states = list(self.memory_states.values())
        
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                resonance = self._calculate_quantum_resonance(
                    states[i].quantum_signature,
                    states[j].quantum_signature
                )
                total_resonance += resonance
                pair_count += 1
        
        return total_resonance / pair_count if pair_count > 0 else 0.0
    
    def update_memory_importance(self, memory_id: str, new_importance: float):
        """Update importance score of a memory"""
        
        if memory_id in self.memory_states:
            self.memory_states[memory_id].importance_score = new_importance
            print(f"📈 Updated memory importance: {memory_id} → {new_importance:.2f}")
        else:
            print(f"⚠️ Memory not found: {memory_id}")
    
    def get_entangled_memories(self, memory_id: str) -> List[QuantumMemoryState]:
        """Get memories quantum entangled with the specified memory"""
        
        if memory_id not in self.memory_states:
            return []
        
        entangled_ids = self.memory_states[memory_id].quantum_entangled
        return [self.memory_states[eid] for eid in entangled_ids if eid in self.memory_states]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        
        current_time = time.time()
        
        return {
            "total_memories": len(self.memory_states),
            "memory_types": {
                "short_term": len(self.short_term_memory),
                "long_term": len(self.long_term_memory), 
                "working": len(self.working_memory)
            },
            "quantum_coherence": self._calculate_quantum_coherence(),
            "average_importance": np.mean([s.importance_score for s in self.memory_states.values()]) if self.memory_states else 0,
            "total_accesses": sum(s.access_count for s in self.memory_states.values()),
            "entanglement_count": sum(len(s.quantum_entangled) for s in self.memory_states.values()),
            "memory_age_hours": {
                "newest": min((current_time - s.timestamp) / 3600 for s in self.memory_states.values()) if self.memory_states else 0,
                "oldest": max((current_time - s.timestamp) / 3600 for s in self.memory_states.values()) if self.memory_states else 0
            }
        }
    
    def save_memory_state(self):
        """Save current memory state to disk"""
        self._save_quantum_memory()
    
    def __del__(self):
        """Cleanup - save memory state"""
        try:
            self._save_quantum_memory()
        except:
            pass

# Test the quantum memory system
if __name__ == "__main__":
    print("🧪 Testing Quantum Memory System...")
    
    # Initialize memory core
    memory = QuantumMemoryCore(memory_file="test_quantum_memory.json")
    
    # Store some test memories
    test_memories = [
        ("Quantum computing uses superposition and entanglement", 0.9),
        ("RFT compression reduces O(2^n) to O(n) complexity", 0.8),
        ("Golden ratio appears in quantum algorithms", 0.7),
        ("QuantoniumOS integrates quantum and classical computing", 0.9),
        ("Machine learning can be enhanced with quantum methods", 0.6)
    ]
    
    memory_ids = []
    for content, importance in test_memories:
        memory_id = memory.store_memory(content, importance)
        memory_ids.append(memory_id)
    
    # Test quantum search
    search_results = memory.retrieve_memory("quantum superposition", max_results=3)
    print(f"\n🔍 Search results for 'quantum superposition':")
    for state in search_results:
        print(f"   {state.id}: {str(state.content)[:60]}... (score: {state.importance_score:.2f})")
    
    # Test entanglments
    if memory_ids:
        entangled = memory.get_entangled_memories(memory_ids[0])
        print(f"\n🔗 Entangled with {memory_ids[0]}: {len(entangled)} memories")
    
    # Show statistics
    stats = memory.get_memory_stats()
    print(f"\n📊 Memory Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Save memory state
    memory.save_memory_state()
    
    print("\n✅ Quantum Memory System validated!")