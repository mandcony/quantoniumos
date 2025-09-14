#!/usr/bin/env python3
"""
Quantum AI Fine-tuning - Fixed Version
Handles API compatibility issues and provides reliable fine-tuning
"""

import torch
from transformers import __version__ as transformers_version
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import json
import os
import sys
from pathlib import Path

def check_requirements():
    """Check if we have the necessary components"""
    print(f"Transformers version: {transformers_version}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")

    return cuda_available

def create_minimal_dataset():
    """Create a minimal working dataset for testing"""
    data = [
        {
            "instruction": "Explain quantum computing",
            "output": "Quantum computing uses quantum mechanics principles like superposition and entanglement to perform computations. Unlike classical bits, quantum bits (qubits) can exist in multiple states simultaneously, enabling massive parallelism for certain problems."
        },
        {
            "instruction": "What is machine learning?",
            "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions."
        },
        {
            "instruction": "Explain the Schrödinger equation",
            "output": "The Schrödinger equation is the fundamental equation of quantum mechanics that describes how quantum states evolve over time. For a single particle, it's iℏ ∂ψ/∂t = - (ℏ²/2m) ∇²ψ + Vψ, where ψ is the wave function, V is the potential, and ℏ is the reduced Planck constant."
        }
    ]

    # Save to file
    output_file = "hf_training_output/minimal_dataset.jsonl"
    os.makedirs("hf_training_output", exist_ok=True)

    with open(output_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

    print(f"Created minimal dataset with {len(data)} examples")
    return output_file

def main():
    print("🔧 Quantum AI Fine-tuning (Fixed Version)")
    print("=" * 45)

    # Check requirements
    cuda_available = check_requirements()

    # Use CPU-compatible model if no CUDA
    if cuda_available:
        model_name = "microsoft/DialoGPT-medium"
        device = "cuda"
    else:
        model_name = "microsoft/DialoGPT-small"  # Even smaller for CPU
        device = "cpu"

    print(f"Using model: {model_name} on {device}")

    # Create or use existing dataset
    dataset_path = "hf_training_output/quantum_finetuning_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print("Creating minimal dataset...")
        dataset_path = create_minimal_dataset()

    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset("json", data_files=dataset_path)
        train_dataset = dataset["train"]
        print(f"Loaded {len(train_dataset)} training examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load model and tokenizer
    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Move to device
        if device == "cuda":
            model = model.to(device)

        print("✅ Model loaded successfully")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Preprocessing function
    def preprocess_function(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            # Simple format for DialoGPT
            text = f"Instruction: {instruction} Response: {output}"
            texts.append(text)

        # Tokenize
        result = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,  # Shorter for DialoGPT
            return_tensors="pt"
        )

        # Set labels for language modeling loss
        result["labels"] = result["input_ids"].clone()

        return result

    print("Preprocessing dataset...")
    try:
        tokenized_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        print("✅ Dataset preprocessed")
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return

    # Training arguments - compatible with current transformers
    output_dir = "hf_training_output/fixed_fine_tuned_model"

    # Check transformers version for correct API
    try:
        # Try newer API first
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,  # Just 1 epoch for testing
            per_device_train_batch_size=1,  # Very small batch
            learning_rate=1e-5,  # Very conservative
            logging_steps=1,
            save_steps=5,
            save_total_limit=1,
            report_to="none",
            # Use newer API if available
            eval_strategy="no",
        )
        print("Using newer transformers API")
    except TypeError:
        # Fallback to older API
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                learning_rate=1e-5,
                logging_steps=1,
                save_steps=5,
                save_total_limit=1,
                report_to="none",
                evaluation_strategy="no",
            )
            print("Using older transformers API")
        except Exception as e:
            print(f"TrainingArguments error: {e}")
            return

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train
    print("🚀 Starting training...")
    try:
        trainer.train()
        print("✅ Training completed!")

        # Save model
        print("💾 Saving model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save config
        config = {
            "model_name": model_name,
            "fine_tuned_path": output_dir,
            "training_samples": len(train_dataset),
            "transformers_version": transformers_version,
            "device": device
        }

        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"🎉 Fine-tuning complete! Model saved to {output_dir}")

        return output_dir

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n✅ Success! Fine-tuned model available at: {result}")
    else:
        print("\n❌ Fine-tuning failed. Check the errors above.")