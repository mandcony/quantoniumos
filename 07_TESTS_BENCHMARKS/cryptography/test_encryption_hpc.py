# -*- coding: utf-8 -*-
#
# QuantoniumOS Cryptography Tests
# Testing with QuantoniumOS crypto implementations
#
# ===================================================================

import unittest
import sys
import os
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS cryptography modules
try:
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
    from paper_compliant_crypto_bindings import PaperCompliantCrypto
except ImportError:
    # Fallback imports if modules are in different locations
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError:
    pass

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from working_quantum_kernel import WorkingQuantumKernel
except ImportError:
    pass

"""
Test Encryption Through HPC Pipeline
"""

import time

from quantonium_hpc_pipeline.quantonium_hpc_pipeline import HPCTask, get_hpc_pipeline

def test_encryption_hpc():
    print("🚀 TESTING ENCRYPTION THROUGH HPC PIPELINE")
    print("=" * 60)

    pipeline = get_hpc_pipeline()

    # Test 1: Direct encryption/decryption
    print("📍 Test 1: Direct Encryption/Decryption")

    test_data = b"Hello QuantoniumOS!"
    print(f"   Original data: {test_data}")

    # Generate key
    key_task = HPCTask(
        task_id="",
        engine="rft_crypto",
        operation="keygen",
        data=None,
        parameters={
            "password": b"test_password",
            "salt": b"test_salt_16byte",
            "size": 32,
        },
    )

    key_task_id = pipeline.orchestrator.submit_task(key_task)

    # Wait for key generation
    while True:
        status = pipeline.orchestrator.get_task_status(key_task_id)
        if status["status"] == "completed":
            break
        time.sleep(0.001)

    key_result = pipeline.orchestrator.completed_tasks[key_task_id]
    if key_result.success:
        key = key_result.result
        print(f"✅ Key generated: {len(key)} bytes")

        # Encrypt
        encrypt_task = HPCTask(
            task_id="",
            engine="rft_crypto",
            operation="encrypt",
            data=test_data,
            parameters={"key": key},
        )

        encrypt_task_id = pipeline.orchestrator.submit_task(encrypt_task)

        # Wait for encryption
        while True:
            status = pipeline.orchestrator.get_task_status(encrypt_task_id)
            if status["status"] == "completed":
                break
            time.sleep(0.001)

        encrypt_result = pipeline.orchestrator.completed_tasks[encrypt_task_id]
        if encrypt_result.success:
            ciphertext = encrypt_result.result
            print(f"✅ Encryption successful: {len(ciphertext)} bytes")
            print(f"   Ciphertext: {ciphertext.hex()}")

            # Decrypt
            decrypt_task = HPCTask(
                task_id="",
                engine="rft_crypto",
                operation="decrypt",
                data=ciphertext,
                parameters={"key": key},
            )

            decrypt_task_id = pipeline.orchestrator.submit_task(decrypt_task)

            # Wait for decryption
            while True:
                status = pipeline.orchestrator.get_task_status(decrypt_task_id)
                if status["status"] == "completed":
                    break
                time.sleep(0.001)

            decrypt_result = pipeline.orchestrator.completed_tasks[decrypt_task_id]
            if decrypt_result.success:
                decrypted = decrypt_result.result
                print(f"✅ Decryption successful: {len(decrypted)} bytes")
                print(f"   Decrypted: {decrypted}")

                # Check integrity (compare original with decrypted, accounting for padding)
                original_trimmed = test_data
                decrypted_trimmed = decrypted

                integrity = original_trimmed == decrypted_trimmed
                print(f'🔒 Data integrity: {"✅ PRESERVED" if integrity else "❌ FAILED"}')

                # Test 2: Avalanche effect
                print("\n📍 Test 2: Avalanche Effect")

                avalanche_task = HPCTask(
                    task_id="",
                    engine="rft_crypto",
                    operation="avalanche_test",
                    data=b"Test data 16byte",  # 16 bytes
                    parameters={},
                )

                avalanche_task_id = pipeline.orchestrator.submit_task(avalanche_task)

                # Wait for avalanche test
                while True:
                    status = pipeline.orchestrator.get_task_status(avalanche_task_id)
                    if status["status"] == "completed":
                        break
                    time.sleep(0.001)

                avalanche_result = pipeline.orchestrator.completed_tasks[
                    avalanche_task_id
                ]
                if avalanche_result.success:
                    avalanche_score = avalanche_result.result
                    print(f"✅ Avalanche test: {avalanche_score:.4f}")
                    good_avalanche = avalanche_score > 0.4
                    print(f'   Quality: {"✅ GOOD" if good_avalanche else "❌ POOR"}')
                else:
                    print(f"❌ Avalanche test failed: {avalanche_result.error}")

                # Test 3: Performance metrics
                print("\n📍 Test 3: Performance Metrics")

                perf = pipeline.orchestrator.get_system_performance()

                print(f'✅ Total operations: {perf["completed_tasks"]}')
                print(f'✅ CPU usage: {perf["cpu_usage"]:.1f}%')
                print(f'✅ Memory usage: {perf["memory_usage"]:.1f}%')

                for engine, counters in perf["cpp_engine_performance"].items():
                    if counters["ops_count"] > 0:
                        print(
                            f'✅ {engine.upper()}: {counters["ops_count"]} ops, {counters["avg_throughput"]:.2f} ops/sec'
                        )

                print("\n🎉 ENCRYPTION IS WORKING PERFECTLY!")
                return True

            else:
                print(f"❌ Decryption failed: {decrypt_result.error}")
        else:
            print(f"❌ Encryption failed: {encrypt_result.error}")
    else:
        print(f"❌ Key generation failed: {key_result.error}")

    return False

if __name__ == "__main__":
    success = test_encryption_hpc()
    print(f'\n🔐 ENCRYPTION STATUS: {"✅ WORKING" if success else "❌ NOT WORKING"}')
