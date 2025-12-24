# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos

import sys
import os
import json
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.apps.ai_model_wrapper import get_model

def check_weights():
    print("--- Checking Active AI Model Weights ---")
    try:
        model, tokenizer = get_model()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Active Model: {model.config._name_or_path}")
        print(f"Total Parameters (Weights): {total_params:,}")
        
        # Check for Quantum Compressed file
        quantum_path = "ai/models/quantum/tinyllama_real_quantum_compressed.json"
        if os.path.exists(quantum_path):
            print(f"\n--- Quantum Compressed File Found ---")
            with open(quantum_path, 'r') as f:
                data = json.load(f)
                states_count = len(data.get("quantum_states", []))
                print(f"Compressed File: {quantum_path}")
                print(f"Quantum States Count: {states_count:,}")
                if total_params > 0 and states_count > 0:
                    ratio = total_params / states_count
                    print(f"Potential Compression Ratio: {ratio:.1f}:1")
        else:
            print("\nNo Quantum Compressed file found at default location.")

    except Exception as e:
        print(f"Error checking model: {e}")

if __name__ == "__main__":
    check_weights()
