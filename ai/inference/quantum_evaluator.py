#!/usr/bin/env python3
"""
Quantum Model Evaluator
=======================

Evaluates fine-tuned models with quantum enhancements and standard metrics.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
import numpy as np
from typing import Dict, List, Any
import json
from pathlib import Path

class QuantumModelEvaluator:
    """Evaluate quantum-enhanced models"""

    def __init__(self, model_path: str, test_dataset_path: str):
        self.model_path = Path(model_path)
        self.test_dataset_path = Path(test_dataset_path)
        self.model = None
        self.tokenizer = None
        self.test_dataset = None

    def load_model(self):
        """Load the fine-tuned model"""
        print(f"🤖 Loading model from {self.model_path}...")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Load base model
            base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Should match training
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Load LoRA weights
            self.model = PeftModel.from_pretrained(self.model, self.model_path)

            print("✅ Model loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def load_test_data(self):
        """Load test dataset"""
        print(f"📚 Loading test dataset from {self.test_dataset_path}...")

        try:
            self.test_dataset = load_from_disk(self.test_dataset_path)
            print(f"✅ Loaded {len(self.test_dataset)} test examples")
            return True

        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return False

    def generate_response(self, instruction: str, max_length: int = 256) -> str:
        """Generate response for instruction"""
        if not self.model or not self.tokenizer:
            return ""

        # Format instruction
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()

        return response

    def evaluate_generation_quality(self, num_samples: int = 10) -> Dict[str, Any]:
        """Evaluate generation quality on test set"""
        print(f"🧪 Evaluating generation quality on {num_samples} samples...")

        results = {
            'responses': [],
            'avg_response_length': 0,
            'quantum_context_usage': 0
        }

        total_length = 0

        # Sample from test set
        indices = np.random.choice(len(self.test_dataset), min(num_samples, len(self.test_dataset)), replace=False)

        for i, idx in enumerate(indices):
            example = self.test_dataset[idx]
            instruction = example['instruction']
            expected_output = example['output']

            print(f"\n📝 Sample {i+1}:")
            print(f"Instruction: {instruction[:100]}...")
            print(f"Expected: {expected_output[:100]}...")

            # Generate response
            generated = self.generate_response(instruction)
            print(f"Generated: {generated[:100]}...")

            results['responses'].append({
                'instruction': instruction,
                'expected': expected_output,
                'generated': generated
            })

            total_length += len(generated)

            # Check for quantum context
            if "[QUANTUM" in generated:
                results['quantum_context_usage'] += 1

        results['avg_response_length'] = total_length / len(results['responses'])
        results['quantum_context_usage'] /= len(results['responses'])

        print("
📊 Generation Quality Results:"        print(f"   Average response length: {results['avg_response_length']:.1f} characters")
        print(f"   Quantum context usage: {results['quantum_context_usage']:.2%}")

        return results

    def calculate_perplexity(self) -> float:
        """Calculate perplexity on test set"""
        print("🧮 Calculating perplexity...")

        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for example in self.test_dataset:
                text = example['text']

                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # Get loss
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

                total_loss += loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        print(f"📊 Perplexity: {perplexity:.2f}")
        return perplexity

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        print("🚀 Starting Quantum Model Evaluation")
        print("=" * 50)

        if not self.load_model():
            return {"error": "Failed to load model"}

        if not self.load_test_data():
            return {"error": "Failed to load test data"}

        results = {
            'perplexity': self.calculate_perplexity(),
            'generation_quality': self.evaluate_generation_quality(),
            'model_info': {
                'path': str(self.model_path),
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }

        # Save results
        results_file = self.model_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON
            json_results = json.dumps(results, indent=2, default=str)
            f.write(json_results)

        print(f"💾 Results saved to {results_file}")
        print("🎉 Evaluation complete!")

        return results

def main():
    """Main evaluation function"""
    # Default paths - adjust as needed
    model_path = "./hf_training_output"
    test_dataset_path = "./hf_datasets/test"

    evaluator = QuantumModelEvaluator(model_path, test_dataset_path)
    results = evaluator.run_full_evaluation()

    print("\n📊 Final Results Summary:")
    if 'error' not in results:
        print(f"   Perplexity: {results['perplexity']:.2f}")
        print(f"   Avg Response Length: {results['generation_quality']['avg_response_length']:.1f}")
        print(f"   Quantum Context Usage: {results['generation_quality']['quantum_context_usage']:.2%}")
    else:
        print(f"   Error: {results['error']}")

if __name__ == "__main__":
    main()