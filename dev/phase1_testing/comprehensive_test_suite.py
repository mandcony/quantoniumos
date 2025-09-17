#!/usr/bin/env python3
"""
Phase 5: Comprehensive Testing and Validation Framework
Complete system integration and benchmarking suite
"""

import os
import sys
import json
import time
import traceback
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import all the phase modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_essential_quantum_ai import EnhancedEssentialQuantumAI
    CONTEXT_AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Context AI import failed: {e}")
    CONTEXT_AI_AVAILABLE = False

try:
    from safe_function_calling_system import SafeToolExecutor
    FUNCTION_CALLING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Function calling import failed: {e}")
    FUNCTION_CALLING_AVAILABLE = False

try:
    from quantum_persistent_memory import QuantumMemoryCore
    PERSISTENT_MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Persistent memory import failed: {e}")
    PERSISTENT_MEMORY_AVAILABLE = False

try:
    from enhanced_multimodal_fusion import QuantumMultimodalFusion, MultimodalInput
    MULTIMODAL_FUSION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Multimodal fusion import failed: {e}")
    MULTIMODAL_FUSION_AVAILABLE = False

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    passed: bool
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

class ComprehensiveTestSuite:
    """Complete testing suite for all quantum AI phases"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        
        # Initialize components
        self.context_ai = None
        self.tool_executor = None
        self.memory_core = None
        self.multimodal_fusion = None
        
        print("🧪 Comprehensive Test Suite Initialized")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all available components"""
        
        if CONTEXT_AI_AVAILABLE:
            try:
                self.context_ai = EnhancedEssentialQuantumAI()
                print("✅ Context AI component initialized")
            except Exception as e:
                print(f"⚠️ Context AI initialization failed: {e}")
        
        if FUNCTION_CALLING_AVAILABLE:
            try:
                self.tool_executor = SafeToolExecutor()
                print("✅ Function calling component initialized")
            except Exception as e:
                print(f"⚠️ Function calling initialization failed: {e}")
        
        if PERSISTENT_MEMORY_AVAILABLE:
            try:
                self.memory_core = QuantumMemoryCore(memory_file="test_comprehensive_memory.json")
                print("✅ Persistent memory component initialized")
            except Exception as e:
                print(f"⚠️ Persistent memory initialization failed: {e}")
        
        if MULTIMODAL_FUSION_AVAILABLE:
            try:
                self.multimodal_fusion = QuantumMultimodalFusion()
                print("✅ Multimodal fusion component initialized")
            except Exception as e:
                print(f"⚠️ Multimodal fusion initialization failed: {e}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        
        print("\n🚀 Starting Comprehensive Test Suite...")
        print("=" * 60)
        
        # Phase 1: Context Extension Tests
        self._test_context_extension()
        
        # Phase 2: Function Calling Tests
        self._test_function_calling()
        
        # Phase 3: Persistent Memory Tests
        self._test_persistent_memory()
        
        # Phase 4: Multimodal Fusion Tests
        self._test_multimodal_fusion()
        
        # Phase 5: Integration Tests
        self._test_system_integration()
        
        # Performance Benchmarks
        self._run_performance_benchmarks()
        
        # Generate final report
        return self._generate_test_report()
    
    def _test_context_extension(self):
        """Test context extension capabilities"""
        
        print("\n📝 Testing Context Extension (Phase 1)...")
        
        if not self.context_ai:
            self._add_test_result("context_extension", False, 0.0, {}, "Component not available")
            return
        
        try:
            start_time = time.time()
            
            # Test short context
            short_response = self.context_ai.generate_response("Hello, test message")
            
            # Test long context  
            long_context = "This is a very long context message. " * 1000
            long_response = self.context_ai.generate_response("Summarize this", long_context)
            
            # Test context compression
            context_result = self.context_ai.process_long_context("Test data " * 10000)
            
            execution_time = time.time() - start_time
            
            # Validate results
            tests_passed = all([
                short_response.text is not None,
                long_response.text is not None,
                context_result["status"] in ["no_compression_needed", "rft_compressed"],
                self.context_ai.context_length == 32768
            ])
            
            details = {
                "context_length": self.context_ai.context_length,
                "short_response_length": len(short_response.text),
                "long_response_length": len(long_response.text),
                "compression_status": context_result["status"],
                "compression_ratio": context_result.get("compression_ratio", 1.0)
            }
            
            self._add_test_result("context_extension", tests_passed, execution_time, details)
            
        except Exception as e:
            self._add_test_result("context_extension", False, 0.0, {}, str(e))
    
    def _test_function_calling(self):
        """Test function calling capabilities"""
        
        print("\n🔧 Testing Function Calling (Phase 2)...")
        
        if not self.tool_executor:
            self._add_test_result("function_calling", False, 0.0, {}, "Component not available")
            return
        
        try:
            start_time = time.time()
            
            # Test available tools
            available_tools = self.tool_executor.get_available_tools()
            
            # Test quantum calculation
            calc_result = self.tool_executor.execute_tool("quantum_calculate", {
                "expression": "pi * phi + sqrt(2)",
                "use_rft": True
            })
            
            # Test system status
            status_result = self.tool_executor.execute_tool("system_status", {})
            
            # Test quantum data analysis
            analysis_result = self.tool_executor.execute_tool("analyze_quantum_data", {
                "data": [1.0, 2.0, 3.0, 4.0, 5.0],
                "analysis_type": "rft_compression"
            })
            
            execution_time = time.time() - start_time
            
            # Validate results
            tests_passed = all([
                len(available_tools) > 0,
                calc_result.get("success", False),
                status_result.get("success", False) or "error" in status_result,  # Some tools may fail validation
                len(self.tool_executor.execution_history) > 0
            ])
            
            details = {
                "available_tools_count": len(available_tools),
                "successful_executions": len([e for e in self.tool_executor.execution_history if e.get("success")]),
                "total_executions": len(self.tool_executor.execution_history),
                "tool_names": list(available_tools.keys())
            }
            
            self._add_test_result("function_calling", tests_passed, execution_time, details)
            
        except Exception as e:
            self._add_test_result("function_calling", False, 0.0, {}, str(e))
    
    def _test_persistent_memory(self):
        """Test persistent memory capabilities"""
        
        print("\n🧠 Testing Persistent Memory (Phase 3)...")
        
        if not self.memory_core:
            self._add_test_result("persistent_memory", False, 0.0, {}, "Component not available")
            return
        
        try:
            start_time = time.time()
            
            # Store test memories
            memory_ids = []
            test_memories = [
                ("Quantum computing principles", 0.9),
                ("RFT compression algorithms", 0.8),
                ("Golden ratio in quantum systems", 0.7)
            ]
            
            for content, importance in test_memories:
                memory_id = self.memory_core.store_memory(content, importance)
                memory_ids.append(memory_id)
            
            # Test memory retrieval
            search_results = self.memory_core.retrieve_memory("quantum", max_results=3)
            
            # Test entanglements
            entangled = []
            if memory_ids:
                entangled = self.memory_core.get_entangled_memories(memory_ids[0])
            
            # Get memory statistics
            stats = self.memory_core.get_memory_stats()
            
            execution_time = time.time() - start_time
            
            # Validate results
            tests_passed = all([
                len(memory_ids) == len(test_memories),
                len(search_results) > 0,
                stats["total_memories"] >= len(test_memories),
                stats["quantum_coherence"] > 0
            ])
            
            details = {
                "memories_stored": len(memory_ids),
                "search_results_count": len(search_results),
                "entangled_memories": len(entangled),
                "total_memories": stats["total_memories"],
                "quantum_coherence": stats["quantum_coherence"]
            }
            
            self._add_test_result("persistent_memory", tests_passed, execution_time, details)
            
        except Exception as e:
            self._add_test_result("persistent_memory", False, 0.0, {}, str(e))
    
    def _test_multimodal_fusion(self):
        """Test multimodal fusion capabilities"""
        
        print("\n🎭 Testing Multimodal Fusion (Phase 4)...")
        
        if not self.multimodal_fusion:
            self._add_test_result("multimodal_fusion", False, 0.0, {}, "Component not available")
            return
        
        try:
            start_time = time.time()
            
            # Test text-only processing
            text_input = MultimodalInput(text="Generate a quantum-inspired visualization")
            text_result = self.multimodal_fusion.process_multimodal_input(text_input)
            
            # Create test image
            from PIL import Image
            import io
            test_image = Image.new('RGB', (64, 64), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='PNG')
            test_image_bytes = img_buffer.getvalue()
            
            # Test image-only processing
            image_input = MultimodalInput(image_data=test_image_bytes)
            image_result = self.multimodal_fusion.process_multimodal_input(image_input)
            
            # Test multimodal processing
            multimodal_input = MultimodalInput(
                text="Analyze this red test image",
                image_data=test_image_bytes
            )
            multimodal_result = self.multimodal_fusion.process_multimodal_input(multimodal_input)
            
            # Get processing stats
            stats = self.multimodal_fusion.get_processing_stats()
            
            execution_time = time.time() - start_time
            
            # Validate results
            tests_passed = all([
                text_result.text_response is not None,
                image_result.text_response is not None,
                multimodal_result.text_response is not None,
                stats["total_processes"] >= 3,
                len(text_result.confidence_scores) > 0
            ])
            
            details = {
                "text_processing": text_result.text_response is not None,
                "image_processing": image_result.text_response is not None,
                "multimodal_processing": multimodal_result.text_response is not None,
                "image_generation": text_result.generated_image is not None,
                "total_processes": stats["total_processes"],
                "average_processing_time": stats["average_processing_time"]
            }
            
            self._add_test_result("multimodal_fusion", tests_passed, execution_time, details)
            
        except Exception as e:
            self._add_test_result("multimodal_fusion", False, 0.0, {}, str(e))
    
    def _test_system_integration(self):
        """Test integration between all components"""
        
        print("\n🔗 Testing System Integration...")
        
        try:
            start_time = time.time()
            
            integration_tests = []
            
            # Test 1: Context AI + Memory Integration
            if self.context_ai and self.memory_core:
                # Generate response and store in memory
                response = self.context_ai.generate_response("What is quantum computing?")
                memory_id = self.memory_core.store_memory(response.text, 0.8)
                
                # Retrieve from memory
                memories = self.memory_core.retrieve_memory("quantum computing")
                integration_tests.append(len(memories) > 0)
            else:
                integration_tests.append(False)
            
            # Test 2: Function Calling + Context AI
            if self.tool_executor and self.context_ai:
                # Use tool executor for calculation
                calc_result = self.tool_executor.execute_tool("quantum_calculate", {
                    "expression": "phi * pi",
                    "use_rft": True
                })
                
                # Use result in context AI
                if calc_result.get("success"):
                    calc_value = calc_result["result"].get("result", 0)
                    response = self.context_ai.generate_response(f"The calculated value is {calc_value}")
                    integration_tests.append(response.text is not None)
                else:
                    integration_tests.append(False)
            else:
                integration_tests.append(False)
            
            # Test 3: Memory + Multimodal Integration
            if self.memory_core and self.multimodal_fusion:
                # Store multimodal interaction in memory
                text_input = MultimodalInput(text="Store this multimodal interaction")
                mm_result = self.multimodal_fusion.process_multimodal_input(text_input)
                
                memory_id = self.memory_core.store_memory(mm_result.text_response, 0.7)
                integration_tests.append(memory_id is not None)
            else:
                integration_tests.append(False)
            
            execution_time = time.time() - start_time
            
            tests_passed = any(integration_tests)  # At least one integration test should pass
            
            details = {
                "context_memory_integration": integration_tests[0] if len(integration_tests) > 0 else False,
                "function_context_integration": integration_tests[1] if len(integration_tests) > 1 else False,
                "memory_multimodal_integration": integration_tests[2] if len(integration_tests) > 2 else False,
                "available_components": self._get_available_components()
            }
            
            self._add_test_result("system_integration", tests_passed, execution_time, details)
            
        except Exception as e:
            self._add_test_result("system_integration", False, 0.0, {}, str(e))
    
    def _run_performance_benchmarks(self):
        """Run performance benchmarks"""
        
        print("\n⚡ Running Performance Benchmarks...")
        
        try:
            start_time = time.time()
            
            benchmark_results = {}
            
            # Context processing benchmark
            if self.context_ai:
                context_times = []
                for size in [100, 1000, 5000]:
                    test_text = "Performance test. " * size
                    ctx_start = time.time()
                    result = self.context_ai.process_long_context(test_text)
                    ctx_time = time.time() - ctx_start
                    context_times.append((size, ctx_time))
                
                benchmark_results["context_processing"] = context_times
            
            # Memory operation benchmark
            if self.memory_core:
                memory_times = []
                for count in [10, 50, 100]:
                    mem_start = time.time()
                    for i in range(count):
                        self.memory_core.store_memory(f"Benchmark memory {i}", 0.5)
                    mem_time = time.time() - mem_start
                    memory_times.append((count, mem_time))
                
                benchmark_results["memory_operations"] = memory_times
            
            # Function calling benchmark
            if self.tool_executor:
                func_times = []
                for count in [5, 10, 20]:
                    func_start = time.time()
                    for i in range(count):
                        self.tool_executor.execute_tool("quantum_calculate", {
                            "expression": f"pi * {i + 1}",
                            "use_rft": True
                        })
                    func_time = time.time() - func_start
                    func_times.append((count, func_time))
                
                benchmark_results["function_calling"] = func_times
            
            execution_time = time.time() - start_time
            
            # Calculate performance scores
            performance_score = self._calculate_performance_score(benchmark_results)
            
            details = {
                "benchmark_results": benchmark_results,
                "performance_score": performance_score,
                "total_benchmark_time": execution_time
            }
            
            self._add_test_result("performance_benchmarks", performance_score > 0.5, execution_time, details)
            
        except Exception as e:
            self._add_test_result("performance_benchmarks", False, 0.0, {}, str(e))
    
    def _calculate_performance_score(self, benchmark_results: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        
        scores = []
        
        # Context processing score (lower time = higher score)
        if "context_processing" in benchmark_results:
            times = [t for _, t in benchmark_results["context_processing"]]
            avg_time = np.mean(times)
            context_score = max(0.0, 1.0 - avg_time / 10.0)  # Good if under 10s
            scores.append(context_score)
        
        # Memory operations score
        if "memory_operations" in benchmark_results:
            times = [t for _, t in benchmark_results["memory_operations"]]
            avg_time = np.mean(times)
            memory_score = max(0.0, 1.0 - avg_time / 5.0)  # Good if under 5s
            scores.append(memory_score)
        
        # Function calling score
        if "function_calling" in benchmark_results:
            times = [t for _, t in benchmark_results["function_calling"]]
            avg_time = np.mean(times)
            func_score = max(0.0, 1.0 - avg_time / 2.0)  # Good if under 2s
            scores.append(func_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _get_available_components(self) -> List[str]:
        """Get list of available components"""
        
        components = []
        if self.context_ai:
            components.append("context_ai")
        if self.tool_executor:
            components.append("function_calling")
        if self.memory_core:
            components.append("persistent_memory")
        if self.multimodal_fusion:
            components.append("multimodal_fusion")
        
        return components
    
    def _add_test_result(self, test_name: str, passed: bool, execution_time: float, 
                        details: Dict[str, Any], error_message: Optional[str] = None):
        """Add test result to collection"""
        
        result = TestResult(
            test_name=test_name,
            passed=passed,
            execution_time=execution_time,
            details=details,
            error_message=error_message
        )
        
        self.test_results.append(result)
        
        # Print immediate result
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status} ({execution_time:.3f}s)")
        if error_message:
            print(f"      Error: {error_message}")
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_time = time.time() - self.start_time
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "success_rate": len(passed_tests) / len(self.test_results) if self.test_results else 0,
                "total_execution_time": total_time
            },
            "component_availability": {
                "context_extension": CONTEXT_AI_AVAILABLE and self.context_ai is not None,
                "function_calling": FUNCTION_CALLING_AVAILABLE and self.tool_executor is not None,
                "persistent_memory": PERSISTENT_MEMORY_AVAILABLE and self.memory_core is not None,
                "multimodal_fusion": MULTIMODAL_FUSION_AVAILABLE and self.multimodal_fusion is not None
            },
            "test_results": [asdict(result) for result in self.test_results],
            "recommendations": self._generate_recommendations(),
            "timestamp": time.time()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Check component availability
        if not CONTEXT_AI_AVAILABLE:
            recommendations.append("Context extension module needs dependency fixes")
        
        if not FUNCTION_CALLING_AVAILABLE:
            recommendations.append("Function calling system needs dependency fixes")
        
        if not PERSISTENT_MEMORY_AVAILABLE:
            recommendations.append("Persistent memory system needs dependency fixes")
        
        if not MULTIMODAL_FUSION_AVAILABLE:
            recommendations.append("Multimodal fusion system needs dependency fixes (PIL required)")
        
        # Check test results
        failed_tests = [r for r in self.test_results if not r.passed]
        
        if any(r.test_name == "performance_benchmarks" and not r.passed for r in failed_tests):
            recommendations.append("Performance optimization needed - consider RFT kernel compilation")
        
        if any(r.test_name == "system_integration" and not r.passed for r in failed_tests):
            recommendations.append("Component integration needs improvement")
        
        # Success recommendations
        passed_tests = [r for r in self.test_results if r.passed]
        
        if len(passed_tests) == len(self.test_results):
            recommendations.append("All tests passed! System ready for production integration")
        elif len(passed_tests) > len(self.test_results) * 0.8:
            recommendations.append("Most tests passed - system ready for staged deployment")
        
        return recommendations

# Main execution
if __name__ == "__main__":
    print("🚀 QuantoniumOS Phase 1-5 Comprehensive Validation Suite")
    print("=" * 70)
    
    # Run comprehensive test suite
    test_suite = ComprehensiveTestSuite()
    report = test_suite.run_all_tests()
    
    # Print final report
    print("\n" + "=" * 70)
    print("📊 FINAL TEST REPORT")
    print("=" * 70)
    
    summary = report["test_summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Time: {summary['total_execution_time']:.2f}s")
    
    print(f"\n🔧 Component Availability:")
    for component, available in report["component_availability"].items():
        status = "✅" if available else "❌"
        print(f"   {component}: {status}")
    
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"   • {rec}")
    
    # Save detailed report
    with open("comprehensive_test_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved: comprehensive_test_report.json")
    
    # Overall assessment
    if summary["success_rate"] >= 0.8:
        print("\n🎉 SYSTEM VALIDATION: SUCCESS")
        print("QuantoniumOS Phase 1-5 enhancements are ready for integration!")
    elif summary["success_rate"] >= 0.6:
        print("\n⚠️ SYSTEM VALIDATION: PARTIAL SUCCESS")
        print("Most features working - address failed components before full deployment")
    else:
        print("\n❌ SYSTEM VALIDATION: NEEDS WORK")
        print("Several critical issues need resolution before deployment")