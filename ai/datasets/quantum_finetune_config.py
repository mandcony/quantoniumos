#!/usr/bin/env python3
"""
Quantum-Enhanced Fine-Tuning Configuration
==========================================

Configuration for fine-tuning LLMs with QuantoniumOS quantum compression integration.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class QuantumFineTuneConfig:
    """Configuration for quantum-enhanced fine-tuning"""

    # Base Model Configuration
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    model_short_name: str = "mistral-7b"

    # Training Configuration
    output_dir: str = "./hf_training_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"

    # LoRA Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = None

    # Quantum Compression Configuration
    use_quantum_compression: bool = True
    quantum_compression_ratio: int = 1000
    quantum_memory_reduction: float = 1000.0

    # Data Configuration
    max_seq_length: int = 2048
    train_data_path: str = "./hf_datasets/train"
    eval_data_path: str = "./hf_datasets/eval"
    test_data_path: str = "./hf_datasets/test"

    # Quantization Configuration
    use_4bit_quantization: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    # Logging and Saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    eval_steps: int = 500

    # Quantum Integration
    quantum_rft_size: int = 256  # RFT transform size for compression
    quantum_batch_compression: bool = True

    def __post_init__(self):
        # Set default LoRA target modules based on model
        if self.lora_target_modules is None:
            if "mistral" in self.base_model_name.lower():
                self.lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            elif "llama" in self.base_model_name.lower():
                self.lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            else:
                self.lora_target_modules = ["q_proj", "v_proj"]

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.train_data_path, exist_ok=True)
        os.makedirs(self.eval_data_path, exist_ok=True)
        os.makedirs(self.test_data_path, exist_ok=True)

# Global configuration instance
quantum_config = QuantumFineTuneConfig()

# Model-specific configurations
MODEL_CONFIGS = {
    "mistral-7b": QuantumFineTuneConfig(
        base_model_name="gpt2-medium",  # Publicly available GPT-2 medium model
        model_short_name="gpt2-medium",
        quantum_compression_ratio=1000,
        quantum_memory_reduction=1000.0,
        lora_target_modules=["c_attn", "c_proj", "c_fc"]  # GPT-2 attention modules
    ),

    "llama-3.1-8b": QuantumFineTuneConfig(
        base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_short_name="llama-3.1-8b",
        quantum_compression_ratio=1000,
        quantum_memory_reduction=1000.0
    ),

    "codellama-7b": QuantumFineTuneConfig(
        base_model_name="codellama/CodeLlama-7b-Python-hf",
        model_short_name="codellama-7b",
        quantum_compression_ratio=1000,
        quantum_memory_reduction=1000.0
    )
}

def get_config(model_name: str = "mistral-7b") -> QuantumFineTuneConfig:
    """Get configuration for specified model"""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    else:
        print(f"⚠️ Model {model_name} not found, using default Mistral config")
        return quantum_config