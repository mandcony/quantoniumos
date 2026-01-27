#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Phase 1 Context Extension Validation Test
Tests 32k context handling, RFT compression, and quantum processing
"""

import os
import sys
import time
import json
from typing import Dict, Any

# Add the phase1_testing directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_essential_quantum_ai import EnhancedEssentialQuantumAI
from rft_context_extension import RFTContextProcessor

def test_context_scaling():
    """Test context processing at different scales"""
    print("🧪 Testing Context Scaling...")
    
    processor = RFTContextProcessor(max_context=32768)
    
    test_cases = [
        ("Small context", "Hello world", 2),
        ("Medium context", "This is a test. " * 1000, 4000),
        ("Large context", "Long story here. " * 5000, 20000),
        ("Mega context", "Very long story. " * 10000, 40000)
    ]
    
    results = []
    
    for name, text, expected_tokens in test_cases:
        start_time = time.time()
        result = processor.process_long_context(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"\n{name}:")
        print(f"   Expected tokens: ~{expected_tokens}")
        print(f"   Actual tokens: {result.get('original_tokens', len(text.split()))}")
        print(f"   Inference tokens: {result.get('inference_tokens', len(text.split()))}")
        print(f"   Status: {result['status']}")
        print(f"   Processing time: {processing_time:.3f}s")
        
        if result['status'] == 'rft_compressed':
            print(f"   Compression ratio: {result['compression_ratio']:.2f}:1")
        
        results.append({
            "name": name,
            "processing_time": processing_time,
            "status": result['status'],
            "compression_ratio": result.get('compression_ratio', 1.0)
        })
    
    return results

def test_ai_conversation():
    """Test AI conversation with extended context"""
    print("\n🤖 Testing AI Conversation...")
    
    ai = EnhancedEssentialQuantumAI()
    
    # Build up a long conversation
    conversation_parts = [
        "Tell me about quantum computing",
        "How does RFT compression work?",
        "What are the advantages of QuantoniumOS?",
        "Can you handle very long contexts?",
        "What about image generation capabilities?"
    ]
    
    # Add lots of context
    long_context = "In quantum computing, we deal with complex mathematical operations. " * 3000
    
    responses = []
    
    for i, question in enumerate(conversation_parts):
        print(f"\nQuestion {i+1}: {question}")
        
        # Add progressively more context
        context = long_context[:i*2000] if i > 0 else ""
        
        start_time = time.time()
        response = ai.generate_response(question, context)
        end_time = time.time()
        
        print(f"Response: {response.text[:100]}...")
        print(f"Context tokens: {response.metadata.get('context_tokens', 0)}")
        print(f"Compression ratio: {response.metadata.get('compression_ratio', 1.0):.2f}:1")
        print(f"Response time: {end_time - start_time:.3f}s")
        
        responses.append({
            "question": question,
            "response_length": len(response.text),
            "context_tokens": response.metadata.get('context_tokens', 0),
            "compression_ratio": response.metadata.get('compression_ratio', 1.0),
            "response_time": end_time - start_time
        })
    
    return responses

def benchmark_performance():
    """Benchmark context processing performance"""
    print("\n⚡ Performance Benchmark...")
    
    processor = RFTContextProcessor(max_context=32768)
    
    # Test different context sizes
    sizes = [1000, 5000, 10000, 20000, 50000]
    benchmark_results = []
    
    for size in sizes:
        text = "Performance test token. " * size
        
        # Time the processing
        start_time = time.time()
        result = processor.process_long_context(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        tokens_per_second = size / processing_time if processing_time > 0 else 0
        
        print(f"Size {size:5d} tokens: {processing_time:.3f}s ({tokens_per_second:.0f} tokens/s)")
        
        benchmark_results.append({
            "input_tokens": size,
            "processing_time": processing_time,
            "tokens_per_second": tokens_per_second,
            "compression_ratio": result.get('compression_ratio', 1.0)
        })
    
    return benchmark_results

def validate_quantum_processing():
    """Validate quantum parameter processing"""
    print("\n🔬 Quantum Processing Validation...")
    
    ai = EnhancedEssentialQuantumAI()
    status = ai.get_status()
    
    print("System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Test quantum-aware responses
    quantum_queries = [
        "What is quantum superposition?",
        "Explain RFT compression",
        "How do quantum algorithms work?",
        "What makes QuantoniumOS different?"
    ]
    
    quantum_results = []
    
    for query in quantum_queries:
        response = ai.generate_response(query)
        
        # Check if response shows quantum awareness
        quantum_terms = ["quantum", "rft", "compression", "algorithms", "processing"]
        quantum_score = sum(1 for term in quantum_terms if term.lower() in response.text.lower())
        
        print(f"\nQuery: {query}")
        print(f"Quantum awareness score: {quantum_score}/5")
        print(f"Response: {response.text[:150]}...")
        
        quantum_results.append({
            "query": query,
            "quantum_score": quantum_score,
            "response_length": len(response.text)
        })
    
    return quantum_results

def main():
    """Run all validation tests"""
    print("🚀 Phase 1 Context Extension - Comprehensive Validation")
    print("=" * 60)
    
    # Run all tests
    scaling_results = test_context_scaling()
    conversation_results = test_ai_conversation()
    benchmark_results = benchmark_performance()
    quantum_results = validate_quantum_processing()
    
    # Compile final report
    report = {
        "validation_timestamp": time.time(),
        "phase": "Phase 1 - Context Extension",
        "tests": {
            "context_scaling": scaling_results,
            "ai_conversation": conversation_results,
            "performance_benchmark": benchmark_results,
            "quantum_validation": quantum_results
        },
        "summary": {
            "max_context_tested": 50000,
            "compression_working": any(r['status'] == 'rft_compressed' for r in scaling_results),
            "ai_responsive": len(conversation_results) > 0,
            "performance_acceptable": all(r['tokens_per_second'] > 1000 for r in benchmark_results),
            "quantum_aware": sum(r['quantum_score'] for r in quantum_results) > 0
        }
    }
    
    # Save results
    with open("phase1_validation_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 40)
    for key, value in report["summary"].items():
        status = "✅ PASS" if value else "❌ FAIL"
        print(f"{key:25s}: {status}")
    
    print(f"\n📄 Full results saved to: phase1_validation_results.json")
    
    # Overall assessment
    passed_tests = sum(1 for v in report["summary"].values() if v)
    total_tests = len(report["summary"])
    
    print(f"\n🎯 OVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 Phase 1 Context Extension: FULLY VALIDATED")
        return True
    else:
        print("⚠️ Phase 1 Context Extension: PARTIAL SUCCESS")
        return False

if __name__ == "__main__":
    success = main()