#!/usr/bin/env python3
"""
Qubit Capability Test - Determine the practical limits of the RFT quantum kernel.
Tests progressively larger qubit counts to find the maximum supported size.
"""

import numpy as np
import time
import psutil
import sys
from unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY, RFT_FLAG_QUANTUM_SAFE


def format_memory(bytes_val):
    """Format memory usage in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process()
    return process.memory_info().rss


def test_qubit_count(qubit_count, verbose=False):
    """Test a specific qubit count and return success/failure with metrics."""
    size = 2 ** qubit_count
    
    print(f"\n🧪 Testing {qubit_count} qubits (size={size:,})")
    
    # Check memory requirements
    complex_size = 16  # 8 bytes real + 8 bytes imag
    basis_memory = size * size * complex_size  # Basis matrix
    eigenval_memory = size * 8  # Eigenvalues (double)
    estimated_memory = basis_memory + eigenval_memory + (size * complex_size * 4)  # Some buffers
    
    print(f"   📊 Estimated memory: {format_memory(estimated_memory)}")
    
    # Check available memory
    available_memory = psutil.virtual_memory().available
    if estimated_memory > available_memory * 0.8:  # Use max 80% of available memory
        print(f"   ⚠️  Insufficient memory (available: {format_memory(available_memory)})")
        return False, None
    
    try:
        # Initialize RFT engine
        start_time = time.time()
        mem_before = get_memory_usage()
        
        rft = UnitaryRFT(size, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE)
        
        init_time = time.time() - start_time
        mem_after = get_memory_usage()
        mem_used = mem_after - mem_before
        
        print(f"   ✅ Initialization: {init_time:.3f}s, Memory used: {format_memory(mem_used)}")
        
        # Initialize quantum basis
        start_time = time.time()
        rft.init_quantum_basis(qubit_count)
        quantum_init_time = time.time() - start_time
        
        print(f"   ✅ Quantum basis: {quantum_init_time:.3f}s")
        
        if verbose:
            # Test a simple quantum state
            start_time = time.time()
            
            # Create a superposition state |0...0⟩ + |1...1⟩
            test_state = np.zeros(size, dtype=np.complex128)
            test_state[0] = 1.0 / np.sqrt(2)      # |0...0⟩
            test_state[size-1] = 1.0 / np.sqrt(2)  # |1...1⟩
            
            # Perform forward transform
            spectrum = rft.forward(test_state)
            
            # Measure entanglement
            entanglement = rft.measure_entanglement(test_state)
            
            operation_time = time.time() - start_time
            
            print(f"   ✅ Operations: {operation_time:.3f}s, Entanglement: {entanglement:.6f}")
        
        metrics = {
            'size': size,
            'init_time': init_time,
            'quantum_init_time': quantum_init_time,
            'memory_used': mem_used,
            'total_time': init_time + quantum_init_time
        }
        
        return True, metrics
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False, None


def find_qubit_limits():
    """Find the practical limits for qubit count."""
    print("🔬 RFT Quantum Kernel - Qubit Capability Analysis")
    print("=" * 60)
    
    # System info
    memory_info = psutil.virtual_memory()
    print(f"💻 System Memory: {format_memory(memory_info.total)} total, {format_memory(memory_info.available)} available")
    print(f"🖥️  CPU Count: {psutil.cpu_count()} cores")
    
    results = []
    max_working_qubits = 0
    
    # Test from 1 qubit up to reasonable limits
    for qubits in range(1, 25):  # Up to 24 qubits (16M states)
        success, metrics = test_qubit_count(qubits, verbose=(qubits <= 10))
        
        if success:
            results.append((qubits, metrics))
            max_working_qubits = qubits
            
            # Stop if initialization takes too long (>30 seconds)
            if metrics['total_time'] > 30.0:
                print(f"   ⏱️  Stopping due to long initialization time ({metrics['total_time']:.1f}s)")
                break
                
            # Stop if using too much memory (>4GB)
            if metrics['memory_used'] > 4 * 1024 * 1024 * 1024:
                print(f"   💾 Stopping due to high memory usage ({format_memory(metrics['memory_used'])})")
                break
        else:
            # Stop on first failure
            break
    
    # Summary
    print(f"\n📈 RESULTS SUMMARY")
    print("=" * 60)
    print(f"🏆 Maximum working qubits: {max_working_qubits}")
    print(f"🧮 Maximum quantum states: {2**max_working_qubits:,}")
    
    if results:
        print(f"\n📊 Performance Summary:")
        print(f"{'Qubits':<8} {'States':<12} {'Init Time':<12} {'Memory':<12} {'Rate':<15}")
        print("-" * 60)
        
        for qubits, metrics in results[-5:]:  # Show last 5 results
            states = metrics['size']
            init_time = metrics['total_time']
            memory = format_memory(metrics['memory_used'])
            rate = f"{states/init_time:.0f} states/s" if init_time > 0 else "N/A"
            
            print(f"{qubits:<8} {states:<12:,} {init_time:<12.3f} {memory:<12} {rate:<15}")
    
    # Practical recommendations
    print(f"\n💡 PRACTICAL RECOMMENDATIONS")
    print("=" * 60)
    
    if max_working_qubits >= 1:
        fast_limit = max(1, max_working_qubits - 3)  # Conservative for fast operations
        print(f"🚀 Fast operations: 1-{fast_limit} qubits")
    
    if max_working_qubits >= 5:
        medium_limit = max(5, max_working_qubits - 1)  # For medium-scale experiments
        print(f"⚡ Medium operations: 5-{medium_limit} qubits")
    
    if max_working_qubits >= 10:
        print(f"🔬 Research scale: {max_working_qubits} qubits (maximum)")
    
    print(f"\n✨ The RFT kernel can handle up to {max_working_qubits} qubits with {2**max_working_qubits:,} quantum states!")
    
    return max_working_qubits, results


if __name__ == "__main__":
    try:
        max_qubits, results = find_qubit_limits()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        sys.exit(1)
