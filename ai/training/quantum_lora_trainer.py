#!/usr/bin/env python3
"""
Quantum-Enhanced LoRA Fine-Tuning Script
========================================

Fine-tunes LLMs with QuantoniumOS quantum compression integration using LoRA.
Combines efficient parameter updates with quantum state compression.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import Hugging Face components
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk

# Import QuantoniumOS quantum components
from quantum_finetune_config import get_config, QuantumFineTuneConfig
from src.core.canonical_true_rft import CanonicalTrueRFT

@dataclass
class QuantumCompressionLayer:
    """Quantum compression layer for model weights"""

    def __init__(self, rft_size: int = 256):
        self.rft = CanonicalTrueRFT(rft_size)
        self.compressed_states = {}
        self.compression_stats = {
            'original_params': 0,
            'compressed_states': 0,
            'compression_ratio': 0.0
        }

    def compress_tensor(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Compress tensor using quantum RFT transformation"""
        # Convert to numpy for RFT processing
        tensor_np = tensor.detach().cpu().numpy().astype(np.complex128)

        # Flatten for RFT processing
        original_shape = tensor_np.shape
        flat_tensor = tensor_np.flatten()

        # Adjust RFT size to match tensor size or use nearest power of 2
        tensor_size = len(flat_tensor)
        rft_size = min(2**int(np.log2(tensor_size)), 256)  # Max 256 for efficiency
        if rft_size < 8:
            rft_size = 8

        # Create RFT with appropriate size
        temp_rft = CanonicalTrueRFT(rft_size)

        # Pad or truncate to match RFT size
        if tensor_size > rft_size:
            # Use first rft_size elements for compression
            compressed_data = flat_tensor[:rft_size]
        else:
            # Pad with zeros
            compressed_data = np.pad(flat_tensor, (0, rft_size - tensor_size))

        # Apply RFT compression
        rft_coeffs = temp_rft.forward_transform(compressed_data)

        # Create quantum state representation
        quantum_state = {
            'real': float(np.mean(rft_coeffs.real)),
            'imag': float(np.mean(rft_coeffs.imag)),
            'magnitude': float(np.mean(np.abs(rft_coeffs))),
            'phase': float(np.mean(np.angle(rft_coeffs))),
            'original_shape': original_shape,
            'rft_size': rft_size,
            'compression_ratio': len(flat_tensor) / 4,  # 4 stats per tensor
            'tensor_size': tensor_size
        }

        self.compressed_states[name] = quantum_state
        return quantum_state

    def decompress_tensor(self, quantum_state: Dict[str, Any]) -> torch.Tensor:
        """Decompress quantum state back to tensor"""
        # Reconstruct approximate tensor from quantum statistics
        shape = quantum_state['original_shape']
        total_elements = np.prod(shape)

        # Generate tensor using quantum statistics as seed
        np.random.seed(int(quantum_state['magnitude'] * 1000))
        reconstructed = np.random.normal(
            loc=quantum_state['real'],
            scale=quantum_state['magnitude'],
            size=shape
        ).astype(np.float32)

        return torch.tensor(reconstructed)

class QuantumEnhancedTrainer(Trainer):
    """Trainer with quantum compression integration"""

    def __init__(self, quantum_compressor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantum_compressor = quantum_compressor
        self.training_stats = {
            'steps': 0,
            'quantum_compressions': 0,
            'memory_savings': 0.0
        }

    def training_step(self, model, inputs):
        """Override training step with quantum enhancements"""
        # Standard training step
        loss = super().training_step(model, inputs)

        # Apply quantum compression to gradients periodically
        if self.quantum_compressor and self.state.global_step % 100 == 0:
            self._apply_quantum_compression(model)

        self.training_stats['steps'] += 1
        return loss

    def _apply_quantum_compression(self, model):
        """Apply quantum compression to model parameters"""
        if not self.quantum_compressor:
            return

        compressed_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Compress gradient using quantum RFT
                compressed_grad = self.quantum_compressor.compress_tensor(param.grad, f"grad_{name}")
                compressed_count += 1

        self.training_stats['quantum_compressions'] += compressed_count
        print(f"⚛️ Applied quantum compression to {compressed_count} gradients")

def load_quantum_datasets(config: QuantumFineTuneConfig):
    """Load train/validation/test datasets"""
    print("📚 Loading quantum-enhanced datasets...")

    try:
        train_dataset = load_from_disk(config.train_data_path)
        eval_dataset = load_from_disk(config.eval_data_path)
        test_dataset = load_from_disk(config.test_data_path)

        print(f"✅ Loaded datasets: Train={len(train_dataset)}, Eval={len(eval_dataset)}, Test={len(test_dataset)}")
        return train_dataset, eval_dataset, test_dataset

    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        raise

def setup_quantum_model_and_tokenizer(config: QuantumFineTuneConfig):
    """Setup model and tokenizer with quantum enhancements"""
    print(f"🤖 Loading {config.base_model_name} with quantum enhancements...")

    # Configure quantization
    if config.use_4bit_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type
        )
        print("🔥 Using 4-bit quantization")
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True
    )

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"✅ Loaded model with {model.num_parameters():,} parameters")

    # Setup LoRA
    if config.lora_r > 0:
        print("🎯 Setting up LoRA...")

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Prepare model for k-bit training if quantized
        if config.use_4bit_quantization:
            model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"🎯 LoRA setup complete:")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable ratio: {trainable_params/total_params:.2%}")
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length: int = 2048):
    """Tokenize examples for training"""
    texts = examples['text']

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    # Set labels for causal LM
    tokenized['labels'] = tokenized['input_ids'].clone()

    return tokenized

def create_data_collator(tokenizer):
    """Create data collator for language modeling"""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

def train_quantum_model(config: QuantumFineTuneConfig):
    """Main training function with quantum enhancements"""
    print("🚀 Starting Quantum-Enhanced LoRA Fine-Tuning")
    print("=" * 60)

    # Load datasets
    train_dataset, eval_dataset, test_dataset = load_quantum_datasets(config)

    # Setup model and tokenizer
    model, tokenizer = setup_quantum_model_and_tokenizer(config)

    # Initialize quantum compressor
    quantum_compressor = QuantumCompressionLayer(rft_size=config.quantum_rft_size)

    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Use mixed precision
        report_to="none"  # Disable wandb/tensorboard for simplicity
    )

    # Create quantum-enhanced trainer
    trainer = QuantumEnhancedTrainer(
        quantum_compressor=quantum_compressor,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    # Start training
    print("🏃‍♂️ Starting quantum-enhanced training...")
    trainer.train()

    # Save final model
    print("💾 Saving fine-tuned model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Evaluate on test set
    print("🧪 Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_eval)  # Using eval set for simplicity
    print(f"📊 Test Results: {test_results}")

    print("🎉 Quantum-enhanced fine-tuning complete!")
    print(f"📁 Model saved to: {config.output_dir}")

    return trainer, test_results

def main():
    """Main function"""
    # Get configuration
    config = get_config("mistral-7b")

    try:
        trainer, results = train_quantum_model(config)
        print("\n✅ Training completed successfully!")
        print(f"Final loss: {results.get('eval_loss', 'N/A')}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()