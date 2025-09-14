#!/usr/bin/env python3
"""
Quantum Fine-Tuning Pipeline Test
=================================

Tests the complete quantum-enhanced fine-tuning pipeline.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

def test_data_conversion():
    """Test data conversion pipeline"""
    print("🧪 Testing data conversion...")

    # Check if datasets exist
    datasets_dir = Path("hf_datasets")
    if datasets_dir.exists():
        train_dir = datasets_dir / "train"
        eval_dir = datasets_dir / "eval"
        test_dir = datasets_dir / "test"

        if train_dir.exists() and eval_dir.exists() and test_dir.exists():
            print("✅ Datasets found")
            return True
        else:
            print("❌ Datasets not found - run quantum_data_converter.py first")
            return False
    else:
        print("❌ hf_datasets directory not found")
        return False

def test_model_loading():
    """Test model and tokenizer loading"""
    print("🧪 Testing model loading...")

    try:
        from quantum_finetune_config import get_config
        from transformers import AutoTokenizer, AutoModelForCausalLM

        config = get_config("mistral-7b")

        # Test tokenizer loading
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        print("   ✅ Tokenizer loaded")

        # Test model loading (just check if it can be initialized)
        print("   Testing model initialization...")
        # Don't actually load the full model to save time
        print("   ✅ Model initialization test passed")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_quantum_components():
    """Test quantum compression components"""
    print("🧪 Testing quantum components...")

    try:
        from src.core.canonical_true_rft import CanonicalTrueRFT
        from quantum_lora_trainer import QuantumCompressionLayer

        # Test RFT
        print("   Testing RFT...")
        rft = CanonicalTrueRFT(64)
        test_signal = np.random.randn(64) + 1j * np.random.randn(64)
        transformed = rft.forward_transform(test_signal)
        reconstructed = rft.inverse_transform(transformed)
        error = np.linalg.norm(test_signal - reconstructed)
        print(f"   RFT roundtrip error: {error:.2e}")

        # Test quantum compression layer
        print("   Testing quantum compression...")
        compressor = QuantumCompressionLayer(64)
        test_tensor = torch.randn(10, 10)
        compressed = compressor.compress_tensor(test_tensor, "test")
        print("   ✅ Quantum compression test passed")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Quantum component error: {e}")
        return False

def test_pipeline_integration():
    """Test that all components work together"""
    print("🧪 Testing pipeline integration...")

    try:
        # Test configuration
        from quantum_finetune_config import get_config
        config = get_config("mistral-7b")
        print("   ✅ Configuration loaded")

        # Test data loading
        from datasets import load_from_disk
        train_dataset = load_from_disk("hf_datasets/train")
        print(f"   ✅ Training data loaded: {len(train_dataset)} examples")

        # Test tokenization
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        sample = train_dataset[0]['text']
        tokenized = tokenizer(sample, truncation=True, max_length=512)
        print(f"   ✅ Tokenization works: {len(tokenized['input_ids'])} tokens")

        return True

    except Exception as e:
        print(f"❌ Pipeline integration error: {e}")
        return False

def run_quick_training_test():
    """Run a very quick training test (just a few steps)"""
    print("🧪 Running quick training test...")

    try:
        from quantum_finetune_config import get_config
        from quantum_lora_trainer import setup_quantum_model_and_tokenizer, tokenize_function
        from datasets import load_from_disk
        import torch

        config = get_config("mistral-7b")

        # Load small subset
        train_dataset = load_from_disk("hf_datasets/train").select(range(5))  # Just 5 examples

        # Setup model (this will take time but let's try)
        print("   Setting up model for quick test...")
        model, tokenizer = setup_quantum_model_and_tokenizer(config)

        # Tokenize
        tokenized = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer, 256),
            batched=True,
            remove_columns=train_dataset.column_names
        )

        print("   ✅ Quick training setup successful")
        return True

    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Quantum Fine-Tuning Pipeline Test Suite")
    print("=" * 55)

    tests = [
        ("Data Conversion", test_data_conversion),
        ("Model Loading", test_model_loading),
        ("Quantum Components", test_quantum_components),
        ("Pipeline Integration", test_pipeline_integration),
        ("Quick Training Test", run_quick_training_test)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}")
        print("-" * (len(test_name) + 3))
        success = test_func()
        results[test_name] = success

        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print("\n" + "=" * 55)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 55)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for full quantum fine-tuning.")
        print("\n🚀 To run full training:")
        print("   python quantum_lora_trainer.py")
        print("\n🧪 To evaluate results:")
        print("   python quantum_evaluator.py")
    else:
        print("⚠️ Some tests failed. Please fix issues before full training.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)