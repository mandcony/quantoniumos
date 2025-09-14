#!/usr/bin/env python3
"""
Quantum AI Fine-tuning Pipeline
Trains models on comprehensive PhD-level knowledge across multiple domains
"""

import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumFineTuner:
    """Fine-tunes language models on quantum-enhanced datasets"""

    def __init__(self, base_model_name: str = None, device: str = "auto"):
        self.base_model_name = base_model_name or self._auto_select_model()
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.training_output_dir = Path("hf_training_output")
        self.training_output_dir.mkdir(exist_ok=True)

    def _auto_select_model(self) -> str:
        """Auto-select model based on hardware"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory >= 16:
                    return "mistralai/Mistral-7B-Instruct-v0.1"
                elif gpu_memory >= 8:
                    return "microsoft/DialoGPT-large"
                else:
                    return "microsoft/DialoGPT-medium"
            else:
                return "microsoft/DialoGPT-medium"
        except:
            return "microsoft/DialoGPT-medium"

    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load and preprocess the training dataset"""
        logger.info(f"Loading dataset from {dataset_path}")

        # Load JSONL dataset
        dataset = load_dataset("json", data_files=str(dataset_path))

        # Split into train/validation if needed
        if "train" not in dataset:
            dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        else:
            train_dataset = dataset["train"]
            eval_dataset = dataset["validation"] if "validation" in dataset else None

        logger.info(f"Dataset loaded: {len(train_dataset)} training examples")
        if eval_dataset:
            logger.info(f"Evaluation set: {len(eval_dataset)} examples")

        return {"train": train_dataset, "eval": eval_dataset}

    def setup_tokenizer_and_model(self):
        """Initialize tokenizer and model with LoRA configuration"""
        logger.info(f"Setting up {self.base_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # Set padding token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Load model
        if "mistral" in self.base_model_name.lower():
            # Use 4-bit quantization for Mistral if possible
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("✅ Loaded Mistral with 4-bit quantization")
            except ImportError:
                logger.warning("BitsAndBytes not available, loading without quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
        else:
            # Standard loading for other models
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

        # Prepare for LoRA training
        if self.device == "cuda":
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA - different target modules for different model architectures
        if "mistral" in self.base_model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gpt" in self.base_model_name.lower() or "dialogpt" in self.base_model_name.lower():
            # DialoGPT uses different attention module names
            target_modules = ["c_attn", "c_proj"] if hasattr(self.model.config, 'n_embd') else ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"]
        else:
            # Generic fallback
            target_modules = ["query", "key", "value", "dense"]

        # Configure LoRA or full fine-tuning based on model size and compatibility
        model_size_gb = self._estimate_model_size()

        try:
            lora_config = LoraConfig(
                r=16,  # Rank
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            logger.info("✅ Using LoRA fine-tuning")

        except ValueError as e:
            if "not found in the base model" in str(e):
                logger.warning(f"LoRA targets not found, falling back to full fine-tuning: {e}")
                # Continue with full fine-tuning
            else:
                raise

        self.model.print_trainable_parameters()

        logger.info("✅ Model and tokenizer ready for training")

    def _estimate_model_size(self) -> float:
        """Estimate model size in GB"""
        if not self.model:
            return 0.0

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_gb = (param_size + buffer_size) / (1024**3)
        return size_gb

    def preprocess_function(self, examples):
        """Preprocess examples for training"""
        # Format as instruction-response pairs
        formatted_texts = []

        for instruction, output in zip(examples["instruction"], examples["output"]):
            if "mistral" in self.base_model_name.lower():
                # Mistral chat format
                formatted_text = f"<s>[INST] {instruction} [/INST] {output}</s>"
            else:
                # Generic format
                formatted_text = f"Instruction: {instruction}\nResponse: {output}\n"

            formatted_texts.append(formatted_text)

        # Tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone()  # For causal LM, labels = input_ids
        }

    def train(self, dataset_path: str, output_dir: str = "quantum_fine_tuned_model"):
        """Run the fine-tuning process"""
        logger.info("🚀 Starting Quantum AI Fine-tuning")
        logger.info("=" * 50)

        # Load dataset
        datasets = self.load_dataset(dataset_path)

        # Setup model and tokenizer
        self.setup_tokenizer_and_model()

        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        train_dataset = datasets["train"].map(
            self.preprocess_function,
            batched=True,
            remove_columns=datasets["train"].column_names
        )

        eval_dataset = None
        if datasets["eval"]:
            eval_dataset = datasets["eval"].map(
                self.preprocess_function,
                batched=True,
                remove_columns=datasets["eval"].column_names
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.training_output_dir / output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=10,
            logging_steps=10,
            save_steps=50,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            fp16=self.device == "cuda",
            dataloader_pin_memory=False,
            report_to="none"  # Disable wandb/tensorboard
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("🎯 Starting training...")
        trainer.train()

        # Save the fine-tuned model
        final_output_dir = self.training_output_dir / output_dir / "final_model"
        trainer.save_model(str(final_output_dir))
        self.tokenizer.save_pretrained(str(final_output_dir))

        logger.info(f"✅ Fine-tuning complete! Model saved to {final_output_dir}")

        # Save training stats
        training_stats = {
            "base_model": self.base_model_name,
            "training_samples": len(train_dataset),
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "final_output_dir": str(final_output_dir)
        }

        with open(self.training_output_dir / "training_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2)

        return str(final_output_dir)

def main():
    """Run the fine-tuning pipeline"""
    print("🧠 Quantum AI Fine-tuning Pipeline")
    print("=" * 40)

    # Initialize fine-tuner
    fine_tuner = QuantumFineTuner()

    # Dataset path
    dataset_path = "hf_training_output/quantum_finetuning_dataset.jsonl"

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Run quantum_finetuning_dataset.py first")
        return

    # Train the model
    output_dir = fine_tuner.train(dataset_path)

    print(f"🎉 Fine-tuning complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Training stats: hf_training_output/training_stats.json")

    return output_dir

if __name__ == "__main__":
    main()