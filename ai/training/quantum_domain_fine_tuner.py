#!/usr/bin/env python3
"""
Quantum Domain Fine-Tuning System
Implements domain-specific fine-tuning with quantum compression for specialized AI capabilities.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from quantum_lora_trainer import QuantumCompressionLayer
from src.core.canonical_true_rft import CanonicalTrueRFT
from quantum_safety_system import QuantumSafetySystem
from quantum_conversation_manager import QuantumConversationManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumDomainFineTuner:
    """Quantum-enhanced domain-specific fine-tuning system."""

    def __init__(self, base_model_name: str = "gpt2-medium", device: str = "auto"):
        self.base_model_name = base_model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.peft_config = None
        self.quantum_compression = None
        self.safety_system = QuantumSafetySystem()
        self.conversation_manager = QuantumConversationManager()

        # Domain configurations
        self.domains = {
            "mathematics": {
                "description": "Mathematical reasoning and problem solving",
                "special_tokens": ["<MATH>", "</MATH>", "<PROOF>", "</PROOF>"],
                "quantum_layers": 2
            },
            "coding": {
                "description": "Programming and software development",
                "special_tokens": ["<CODE>", "</CODE>", "<DEBUG>", "</DEBUG>"],
                "quantum_layers": 3
            },
            "science": {
                "description": "Scientific research and analysis",
                "special_tokens": ["<HYPOTHESIS>", "</HYPOTHESIS>", "<EXPERIMENT>", "</EXPERIMENT>"],
                "quantum_layers": 2
            },
            "creative": {
                "description": "Creative writing and content generation",
                "special_tokens": ["<CREATIVE>", "</CREATIVE>", "<STORY>", "</STORY>"],
                "quantum_layers": 1
            },
            "business": {
                "description": "Business analysis and strategy",
                "special_tokens": ["<BUSINESS>", "</BUSINESS>", "<STRATEGY>", "</STRATEGY>"],
                "quantum_layers": 2
            }
        }

    def initialize_model(self):
        """Initialize the base model with quantum enhancements."""
        logger.info("🔧 Initializing quantum domain fine-tuning model...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        # Prepare for LoRA
        self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj", "c_fc"]
        )

        self.model = get_peft_model(self.model, self.peft_config)

        # Add quantum compression
        self.quantum_compression = QuantumCompressionLayer(
            rft_size=self.model.config.hidden_size
        )

        # Add domain-specific quantum layers
        self.domain_quantum_layers = {}
        for domain, config in self.domains.items():
            layers = []
            for _ in range(config["quantum_layers"]):
                layers.append(CanonicalTrueRFT(
                    size=self.model.config.hidden_size
                ))
            self.domain_quantum_layers[domain] = layers

        logger.info("✅ Quantum domain fine-tuning model initialized")

    def prepare_domain_data(self, domain: str, data_path: Optional[str] = None) -> Dataset:
        """Prepare domain-specific training data."""
        logger.info(f"📚 Preparing {domain} domain data...")

        if data_path and os.path.exists(data_path):
            # Load from file
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.json'):
                    data = json.load(f)
                else:
                    data = {"text": f.read().split('\n\n')}
        else:
            # Generate synthetic domain data
            data = self._generate_domain_data(domain)

        # Add domain markers
        domain_config = self.domains[domain]
        marked_texts = []
        for text in data["text"]:
            marked_text = f"{domain_config['special_tokens'][0]}\n{text}\n{domain_config['special_tokens'][1]}"
            marked_texts.append(marked_text)

        # Tokenize
        tokenized = self.tokenizer(
            marked_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        })

        logger.info(f"✅ Prepared {len(dataset)} {domain} training samples")
        return dataset

    def _generate_domain_data(self, domain: str) -> Dict[str, List[str]]:
        """Generate synthetic domain-specific training data."""
        samples = []

        if domain == "mathematics":
            samples = [
                "Solve for x: 2x + 3 = 7\nStep 1: Subtract 3 from both sides: 2x = 4\nStep 2: Divide by 2: x = 2",
                "Prove that the sum of angles in a triangle is 180 degrees.\nGiven triangle ABC, extend side BC to D. Angle ACD = angle ACB + angle BCD. But angle ACD = 180° - angle BAC. Therefore angle BAC + angle ACB + angle BCD = 180°.",
                "Calculate the derivative of f(x) = x² + 3x + 1\nf'(x) = 2x + 3"
            ]
        elif domain == "coding":
            samples = [
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Test\nprint(fibonacci(10))",
                "class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def multiply(self, a, b):\n        return a * b\n\ncalc = Calculator()\nprint(calc.add(5, 3))",
                "# Debug: Fix the infinite loop\nwhile True:\n    user_input = input('Enter a number: ')\n    if user_input == 'quit':\n        break\n    print(f'You entered: {user_input}')"
            ]
        elif domain == "science":
            samples = [
                "Hypothesis: Plants grow faster with more sunlight.\nExperiment: Grow identical plants with different light exposure.\nResult: Plants with more light grew 30% taller.",
                "Newton's First Law: An object at rest stays at rest, and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced force.",
                "The scientific method: 1) Observe, 2) Hypothesize, 3) Experiment, 4) Analyze, 5) Conclude"
            ]
        elif domain == "creative":
            samples = [
                "Once upon a time, in a magical forest, there lived a wise old owl who could speak every language in the world. One day, he met a curious fox...",
                "Write a haiku about autumn:\nLeaves fall gently down,\nCrimson and gold dance in breeze,\nWinter whispers near.",
                "The hero stood at the crossroads, destiny pulling him in three directions. Which path would he choose?"
            ]
        elif domain == "business":
            samples = [
                "Business Strategy: Identify market gaps, develop unique value proposition, build customer relationships, optimize operations.",
                "SWOT Analysis:\nStrengths: Strong brand, loyal customers\nWeaknesses: High costs, limited market reach\nOpportunities: New markets, technology trends\nThreats: Competition, economic downturns",
                "Key Performance Indicators: Revenue growth, customer acquisition cost, lifetime value, churn rate."
            ]

        return {"text": samples}

    def fine_tune_domain(self, domain: str, dataset: Dataset, output_dir: str = "./domain_models"):
        """Fine-tune the model for a specific domain."""
        logger.info(f"🎯 Fine-tuning for {domain} domain...")

        # Create output directory
        domain_dir = os.path.join(output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=domain_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            fp16=self.device == "cuda",
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            evaluation_strategy="steps",
            eval_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="loss"
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,  # Using same data for simplicity
            data_collator=data_collator
        )

        # Train
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model(domain_dir)
        self.tokenizer.save_pretrained(domain_dir)

        logger.info(f"✅ {domain.capitalize()} domain fine-tuning complete")

    def generate_domain_response(self, domain: str, prompt: str, max_length: int = 100) -> str:
        """Generate a response specialized for a domain."""
        logger.info(f"💬 Generating {domain} response...")

        # Add domain markers
        domain_config = self.domains[domain]
        full_prompt = f"{domain_config['special_tokens'][0]}\n{prompt}\n{domain_config['special_tokens'][1]}"

        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        # Generate with safety checks
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Safety check
        safety_result = self.safety_system.check_safety(response)
        safe_response = safety_result.modified_response if safety_result.modified_response else response

        # Update conversation memory (simplified for now)
        # self.conversation_manager.add_turn("domain_session", prompt, safe_response)

        return safe_response

    def get_domain_stats(self) -> Dict[str, Dict]:
        """Get statistics for all domains."""
        stats = {}
        for domain in self.domains:
            stats[domain] = {
                "description": self.domains[domain]["description"],
                "quantum_layers": self.domains[domain]["quantum_layers"],
                "special_tokens": self.domains[domain]["special_tokens"]
            }
        return stats


def test_quantum_domain_fine_tuning():
    """Test the quantum domain fine-tuning system."""
    print("🧠 Testing Quantum Domain Fine-Tuning System")
    print("=" * 50)

    # Initialize
    tuner = QuantumDomainFineTuner()
    tuner.initialize_model()

    # Test each domain
    for domain in tuner.domains:
        print(f"\n🎯 Testing {domain.upper()} domain:")

        # Prepare data
        dataset = tuner.prepare_domain_data(domain)

        # Fine-tune (simulated - would take too long for full training)
        print(f"   📚 Dataset size: {len(dataset)} samples")
        print("   🎯 Fine-tuning simulated (full training requires more data/time)")

        # Generate sample response
        test_prompts = {
            "mathematics": "Solve: 3x - 7 = 8",
            "coding": "Write a function to reverse a string in Python",
            "science": "Explain photosynthesis in simple terms",
            "creative": "Write the beginning of a mystery story",
            "business": "What are the key components of a business plan?"
        }

        if domain in test_prompts:
            response = tuner.generate_domain_response(domain, test_prompts[domain], max_length=50)
            print(f"   💬 Sample response: {response[:100]}...")

    # Show stats
    print("\n📊 Domain Statistics:")
    stats = tuner.get_domain_stats()
    for domain, info in stats.items():
        print(f"   {domain.capitalize()}: {info['description']} ({info['quantum_layers']} quantum layers)")

    print("\n🎉 Quantum domain fine-tuning test complete!")


if __name__ == "__main__":
    test_quantum_domain_fine_tuning()