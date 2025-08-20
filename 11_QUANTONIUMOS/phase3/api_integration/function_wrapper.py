"""
QuantoniumOS Phase 3: API Integration - Function Wrapper Layer
Comprehensive wrapper for existing Python/C++ functions with quantum integration
"""

import sys
import os
import importlib
import inspect
import json
import asyncio
from typing import Any, Dict, List, Callable, Optional, Union
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'kernel'))

class QuantoniumFunctionWrapper:
    """
    Universal wrapper for Python/C++ functions with quantum state management
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.wrapped_functions = {}
        self.execution_history = []
        self.quantum_state_registry = {}
        self.performance_metrics = {}
        
        # Thread pools for different types of operations
        self.thread_executor = ThreadPoolExecutor(max_workers=8)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        
        self._initialize_core_modules()
    
    def _initialize_core_modules(self):
        """Initialize and register core QuantoniumOS modules"""
        try:
            # Try to import core modules
            self.core_modules = {}
            
            # RFT modules
            try:
                from canonical_true_rft_fixed import true_rft_transform
                self.core_modules['rft_transform'] = true_rft_transform
                self.logger.info("RFT transform module loaded")
            except ImportError:
                self.logger.warning("RFT transform module not available")
            
            # Quantum modules
            try:
                from quantum_vertex_kernel import QuantoniumKernel
                self.core_modules['quantum_kernel'] = QuantoniumKernel
                self.logger.info("Quantum kernel module loaded")
            except ImportError:
                self.logger.warning("Quantum kernel module not available")
            
            # Cryptography modules
            try:
                import cryptography
                self.core_modules['crypto'] = cryptography
                self.logger.info("Cryptography module loaded")
            except ImportError:
                self.logger.warning("Cryptography module not available")
                
        except Exception as e:
            self.logger.error(f"Error initializing core modules: {e}")
    
    def register_function(self, 
                         func: Callable, 
                         name: str = None, 
                         category: str = "general",
                         quantum_aware: bool = False,
                         async_capable: bool = False) -> str:
        """
        Register a function for quantum-aware execution
        
        Args:
            func: Function to register
            name: Optional name override
            category: Function category (rft, quantum, crypto, etc.)
            quantum_aware: Whether function can handle quantum states
            async_capable: Whether function supports async execution
            
        Returns:
            Function ID for later reference
        """
        function_name = name or func.__name__
        function_id = f"{category}_{function_name}_{id(func)}"
        
        # Extract function metadata
        signature = inspect.signature(func)
        doc = inspect.getdoc(func) or "No documentation available"
        
        wrapper_info = {
            'function': func,
            'name': function_name,
            'category': category,
            'signature': str(signature),
            'documentation': doc,
            'quantum_aware': quantum_aware,
            'async_capable': async_capable,
            'parameters': list(signature.parameters.keys()),
            'registered_at': datetime.now().isoformat(),
            'call_count': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
        self.wrapped_functions[function_id] = wrapper_info
        self.logger.info(f"Registered function: {function_name} (ID: {function_id})")
        
        return function_id
    
    async def execute_function(self, 
                              function_id: str, 
                              *args, 
                              quantum_context: Dict = None,
                              execution_mode: str = "sync",
                              **kwargs) -> Dict[str, Any]:
        """
        Execute a registered function with quantum context management
        
        Args:
            function_id: ID of registered function
            *args: Function arguments
            quantum_context: Optional quantum state context
            execution_mode: "sync", "async", "thread", or "process"
            **kwargs: Function keyword arguments
            
        Returns:
            Execution result with metadata
        """
        if function_id not in self.wrapped_functions:
            raise ValueError(f"Function ID {function_id} not registered")
        
        wrapper_info = self.wrapped_functions[function_id]
        func = wrapper_info['function']
        
        # Prepare execution context
        execution_context = {
            'function_id': function_id,
            'function_name': wrapper_info['name'],
            'category': wrapper_info['category'],
            'quantum_context': quantum_context or {},
            'execution_mode': execution_mode,
            'started_at': datetime.now().isoformat(),
            'args': args,
            'kwargs': kwargs
        }
        
        try:
            # Pre-execution quantum state management
            if wrapper_info['quantum_aware'] and quantum_context:
                await self._prepare_quantum_context(quantum_context)
            
            start_time = datetime.now()
            
            # Execute based on mode
            if execution_mode == "async" and wrapper_info['async_capable']:
                result = await func(*args, **kwargs)
            elif execution_mode == "thread":
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_executor, func, *args
                )
            elif execution_mode == "process":
                result = await asyncio.get_event_loop().run_in_executor(
                    self.process_executor, func, *args
                )
            else:
                # Synchronous execution
                result = func(*args, **kwargs)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Update performance metrics
            wrapper_info['call_count'] += 1
            wrapper_info['total_execution_time'] += execution_time
            wrapper_info['average_execution_time'] = (
                wrapper_info['total_execution_time'] / wrapper_info['call_count']
            )
            
            # Post-execution quantum state management
            if wrapper_info['quantum_aware'] and quantum_context:
                await self._update_quantum_context(quantum_context, result)
            
            execution_result = {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'context': execution_context,
                'completed_at': end_time.isoformat()
            }
            
            # Log execution
            self.execution_history.append(execution_result)
            self.logger.info(f"Executed {wrapper_info['name']} in {execution_time:.4f}s")
            
            return execution_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'context': execution_context,
                'failed_at': datetime.now().isoformat()
            }
            
            self.execution_history.append(error_result)
            self.logger.error(f"Function execution failed: {e}")
            
            return error_result
    
    async def _prepare_quantum_context(self, quantum_context: Dict):
        """Prepare quantum state for function execution"""
        # Initialize quantum state if needed
        if 'quantum_state' not in quantum_context:
            quantum_context['quantum_state'] = {
                'qubits': quantum_context.get('num_qubits', 10),
                'entangled_pairs': [],
                'measurement_results': []
            }
        
        # Store context in registry
        context_id = f"ctx_{datetime.now().timestamp()}"
        self.quantum_state_registry[context_id] = quantum_context
        quantum_context['context_id'] = context_id
    
    async def _update_quantum_context(self, quantum_context: Dict, result: Any):
        """Update quantum state after function execution"""
        if 'context_id' in quantum_context:
            context_id = quantum_context['context_id']
            if context_id in self.quantum_state_registry:
                # Update stored context with results
                self.quantum_state_registry[context_id]['last_result'] = result
                self.quantum_state_registry[context_id]['updated_at'] = datetime.now().isoformat()
    
    def get_function_info(self, function_id: str = None) -> Dict[str, Any]:
        """Get information about registered functions"""
        if function_id:
            if function_id in self.wrapped_functions:
                return self.wrapped_functions[function_id]
            else:
                return {"error": f"Function ID {function_id} not found"}
        else:
            return {
                'total_functions': len(self.wrapped_functions),
                'functions': {fid: info for fid, info in self.wrapped_functions.items()},
                'categories': list(set(info['category'] for info in self.wrapped_functions.values()))
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all registered functions"""
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len([r for r in self.execution_history if r['success']]),
            'failed_executions': len([r for r in self.execution_history if not r['success']]),
            'function_metrics': {
                fid: {
                    'call_count': info['call_count'],
                    'total_time': info['total_execution_time'],
                    'average_time': info['average_execution_time']
                }
                for fid, info in self.wrapped_functions.items()
            }
        }
    
    def export_api_schema(self) -> Dict[str, Any]:
        """Export API schema for external integration"""
        schema = {
            'api_version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'functions': {}
        }
        
        for function_id, info in self.wrapped_functions.items():
            schema['functions'][function_id] = {
                'name': info['name'],
                'category': info['category'],
                'signature': info['signature'],
                'documentation': info['documentation'],
                'parameters': info['parameters'],
                'quantum_aware': info['quantum_aware'],
                'async_capable': info['async_capable']
            }
        
        return schema

# Global wrapper instance
quantum_wrapper = QuantoniumFunctionWrapper()

def register_quantum_function(category: str = "general", 
                             quantum_aware: bool = False,
                             async_capable: bool = False):
    """Decorator for easy function registration"""
    def decorator(func):
        function_id = quantum_wrapper.register_function(
            func, 
            category=category,
            quantum_aware=quantum_aware,
            async_capable=async_capable
        )
        func._quantum_function_id = function_id
        return func
    return decorator

# Example usage and built-in function registrations
if __name__ == "__main__":
    # Example: Register some basic functions
    
    @register_quantum_function(category="math", quantum_aware=False)
    def fibonacci(n: int) -> int:
        """Calculate Fibonacci number"""
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    @register_quantum_function(category="quantum", quantum_aware=True)
    def quantum_random(num_qubits: int) -> List[int]:
        """Generate quantum random numbers"""
        import random
        return [random.randint(0, 1) for _ in range(num_qubits)]
    
    # Demo execution
    async def demo():
        print("QuantoniumOS Function Wrapper Demo")
        print("=" * 50)
        
        # Execute Fibonacci
        result1 = await quantum_wrapper.execute_function(
            fibonacci._quantum_function_id, 
            10
        )
        print(f"Fibonacci(10): {result1['result']}")
        
        # Execute quantum random with context
        quantum_context = {'num_qubits': 5}
        result2 = await quantum_wrapper.execute_function(
            quantum_random._quantum_function_id,
            5,
            quantum_context=quantum_context
        )
        print(f"Quantum Random: {result2['result']}")
        
        # Show metrics
        metrics = quantum_wrapper.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        print(f"Total Executions: {metrics['total_executions']}")
        print(f"Successful: {metrics['successful_executions']}")
        
        # Export schema
        schema = quantum_wrapper.export_api_schema()
        print(f"\nAPI Schema exported with {len(schema['functions'])} functions")
    
    asyncio.run(demo())
