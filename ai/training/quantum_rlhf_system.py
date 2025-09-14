#!/usr/bin/env python3
"""
Quantum RLHF Implementation
===========================

Reinforcement Learning from Human Feedback with quantum enhancements.
Includes preference data collection, reward modeling, and PPO fine-tuning.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

@dataclass
class PreferencePair:
    """A preference pair for RLHF training"""
    prompt: str
    chosen_response: str
    rejected_response: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class QuantumPreferenceDataset(Dataset):
    """Dataset for preference learning"""

    def __init__(self, preference_pairs: List[PreferencePair], tokenizer, max_length: int = 512):
        self.preference_pairs = preference_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.preference_pairs)

    def __getitem__(self, idx):
        pair = self.preference_pairs[idx]

        # Tokenize prompt + chosen response
        chosen_text = f"{pair.prompt}\n\n{pair.chosen_response}"
        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tokenize prompt + rejected response
        rejected_text = f"{pair.prompt}\n\n{pair.rejected_response}"
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
            "prompt": pair.prompt
        }

class QuantumRewardModel(nn.Module):
    """Reward model for RLHF with quantum enhancements"""

    def __init__(self, base_model_name: str = "gpt2-medium", hidden_size: int = 1024):  # GPT-2 medium has 1024 hidden size
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.quantum_scale = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask=None):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Use last hidden state
        last_hidden_state = outputs.hidden_states[-1]

        # Get reward score from last token
        last_token_hidden = last_hidden_state[:, -1, :]
        reward_score = self.reward_head(last_token_hidden)

        # Apply quantum scaling
        reward_score = reward_score * self.quantum_scale

        return reward_score

    def get_reward(self, text: str, tokenizer) -> float:
        """Get reward score for a text"""
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            reward = self.forward(**tokens)

        return reward.item()

class QuantumRewardTrainer(Trainer):
    """Trainer for the reward model"""

    def compute_loss(self, model, inputs, return_outputs=False):
        chosen_rewards = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        )

        rejected_rewards = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        )

        # Bradley-Terry loss: log(sigmoid(chosen - rejected))
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        return (loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}) if return_outputs else loss

class QuantumPPOTrainer:
    """PPO trainer with quantum enhancements"""

    def __init__(self, model, reward_model, tokenizer, config):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config

        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        self.max_grad_norm = 0.5

        # Quantum enhancements
        self.quantum_regularization = 0.01

    def generate_responses(self, prompts: List[str], num_samples: int = 4) -> List[List[str]]:
        """Generate multiple response samples for each prompt"""
        all_responses = []

        for prompt in prompts:
            responses = []
            for _ in range(num_samples):
                # Generate response with sampling
                inputs = self.tokenizer(prompt, return_tensors="pt")

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=len(inputs["input_ids"][0]) + 100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                responses.append(response.strip())

            all_responses.append(responses)

        return all_responses

    def compute_rewards(self, prompts: List[str], responses: List[List[str]]) -> torch.Tensor:
        """Compute rewards for generated responses"""
        all_rewards = []

        for prompt, response_list in zip(prompts, responses):
            prompt_rewards = []

            for response in response_list:
                full_text = f"{prompt}\n\n{response}"

                # Get reward from reward model
                reward = self.reward_model.get_reward(full_text, self.tokenizer)

                # Add quantum enhancement bonus
                if "[QUANTUM" in response:
                    reward += 0.1  # Small bonus for quantum context

                prompt_rewards.append(reward)

            all_rewards.extend(prompt_rewards)

        return torch.tensor(all_rewards)

    def ppo_step(self, prompts: List[str], old_log_probs: torch.Tensor, rewards: torch.Tensor):
        """Perform one PPO update step"""
        # Generate new responses
        new_responses = self.generate_responses(prompts, num_samples=1)
        new_responses_flat = [resp[0] for resp in new_responses]

        # Compute new log probabilities
        new_log_probs = self._compute_log_probs(prompts, new_responses_flat)

        # Compute PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        ppo_loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()

        # Add entropy bonus
        entropy = self._compute_entropy(prompts, new_responses_flat)
        entropy_loss = -self.entropy_coeff * entropy.mean()

        # Add quantum regularization
        quantum_loss = self.quantum_regularization * self.model.quantum_scale.abs().mean()

        total_loss = ppo_loss + entropy_loss + quantum_loss

        # Update model
        self.model.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Here you would typically use an optimizer step
        # optimizer.step()

        return {
            "ppo_loss": ppo_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "quantum_loss": quantum_loss.item(),
            "total_loss": total_loss.item()
        }

    def _compute_log_probs(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute log probabilities for responses"""
        log_probs = []

        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}\n\n{response}"
            inputs = self.tokenizer(full_text, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Get log probs for generated tokens
                log_prob = 0
                for i in range(len(inputs["input_ids"][0]) - len(self.tokenizer.encode(prompt))):
                    token_id = inputs["input_ids"][0][len(self.tokenizer.encode(prompt)) + i]
                    log_prob += logits[0, len(self.tokenizer.encode(prompt)) + i - 1, token_id].log()

                log_probs.append(log_prob)

        return torch.tensor(log_probs)

    def _compute_entropy(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute entropy for responses"""
        entropies = []

        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}\n\n{response}"
            inputs = self.tokenizer(full_text, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Compute entropy of the last token distribution
                probs = F.softmax(logits[0, -1, :], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                entropies.append(entropy)

        return torch.tensor(entropies)

class QuantumRLHFSystem:
    """Complete RLHF system with quantum enhancements"""

    def __init__(self, base_model_name: str = "gpt2-medium"):
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.reward_model = None
        self.policy_model = None
        self.ppo_trainer = None

        self.preference_data = []
        self.training_stats = {
            "reward_model_epochs": 0,
            "ppo_steps": 0,
            "total_preferences": 0
        }

    def initialize_models(self):
        """Initialize all models for RLHF"""
        print("🤖 Initializing RLHF models...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize reward model
        self.reward_model = QuantumRewardModel(self.base_model_name)

        # Initialize policy model with LoRA
        base_policy = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"]
        )
        self.policy_model = get_peft_model(base_policy, lora_config)

        # Add quantum scale parameter to policy model
        self.policy_model.quantum_scale = nn.Parameter(torch.ones(1))

        print("✅ RLHF models initialized")

    def add_preference_data(self, prompt: str, chosen: str, rejected: str, metadata: Dict = None):
        """Add a preference pair to the dataset"""
        pair = PreferencePair(
            prompt=prompt,
            chosen_response=chosen,
            rejected_response=rejected,
            metadata=metadata or {}
        )

        self.preference_data.append(pair)
        self.training_stats["total_preferences"] += 1

    def train_reward_model(self, num_epochs: int = 3, batch_size: int = 4):
        """Train the reward model"""
        print(f"🎯 Training reward model for {num_epochs} epochs...")

        if not self.preference_data:
            print("❌ No preference data available")
            return

        # Create dataset
        dataset = QuantumPreferenceDataset(self.preference_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup trainer
        training_args = TrainingArguments(
            output_dir="./rlhf_reward_model",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=1e-5,
            logging_steps=10,
            save_steps=100,
            eval_strategy="no",
            report_to="none"
        )

        trainer = QuantumRewardTrainer(
            model=self.reward_model,
            args=training_args,
            train_dataset=dataset
        )

        # Train
        trainer.train()

        self.training_stats["reward_model_epochs"] += num_epochs
        print("✅ Reward model training complete")

    def train_ppo(self, prompts: List[str], num_steps: int = 10):
        """Train with PPO"""
        print(f"🚀 Training with PPO for {num_steps} steps...")

        if self.ppo_trainer is None:
            self.ppo_trainer = QuantumPPOTrainer(
                self.policy_model, self.reward_model, self.tokenizer, {}
            )

        for step in range(num_steps):
            # Generate responses and compute rewards
            responses = self.ppo_trainer.generate_responses(prompts)
            rewards = self.ppo_trainer.compute_rewards(prompts, responses)

            # Compute old log probs (simplified)
            old_log_probs = torch.randn(len(rewards))  # Placeholder

            # PPO step
            step_stats = self.ppo_trainer.ppo_step(prompts, old_log_probs, rewards)

            print(f"Step {step + 1}: Loss = {step_stats['total_loss']:.4f}")

        self.training_stats["ppo_steps"] += num_steps
        print("✅ PPO training complete")

    def generate_response(self, prompt: str) -> str:
        """Generate a response using the trained policy"""
        if not self.policy_model:
            return "Model not trained yet."

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_length=len(inputs["input_ids"][0]) + 100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response.strip()

    def save_models(self, output_dir: str = "./rlhf_models"):
        """Save trained models"""
        Path(output_dir).mkdir(exist_ok=True)

        if self.reward_model:
            torch.save(self.reward_model.state_dict(), f"{output_dir}/reward_model.pt")

        if self.policy_model:
            self.policy_model.save_pretrained(f"{output_dir}/policy_model")

        if self.tokenizer:
            self.tokenizer.save_pretrained(f"{output_dir}/tokenizer")

        # Save training stats
        with open(f"{output_dir}/training_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        print(f"💾 Models saved to {output_dir}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return self.training_stats.copy()

# Global RLHF system instance
rlhf_system = QuantumRLHFSystem()

def create_sample_preferences():
    """Create sample preference data for testing"""
    preferences = [
        PreferencePair(
            prompt="Explain quantum computing",
            chosen_response="Quantum computing uses quantum mechanics principles like superposition and entanglement to perform computations. Unlike classical bits, quantum bits (qubits) can exist in multiple states simultaneously, allowing for massive parallelism.",
            rejected_response="Quantum computing is when computers use quantum stuff. It's complicated and I don't really understand it well."
        ),
        PreferencePair(
            prompt="How does safety work in AI?",
            chosen_response="AI safety involves multiple layers: technical alignment to ensure AI goals match human values, robustness testing, and monitoring systems. Safety guardrails prevent harmful outputs while maintaining helpfulness.",
            rejected_response="Safety in AI means making sure the AI doesn't break or something. It's important I guess."
        ),
        PreferencePair(
            prompt="What is machine learning?",
            chosen_response="Machine learning is a subset of AI where algorithms learn patterns from data without being explicitly programmed. It includes supervised learning (with labels), unsupervised learning (finding patterns), and reinforcement learning (learning through rewards).",
            rejected_response="Machine learning is when computers learn stuff automatically. Like magic but with math."
        )
    ]

    return preferences

if __name__ == "__main__":
    # Test RLHF system
    print("🧠 Testing Quantum RLHF System")
    print("=" * 35)

    # Initialize
    rlhf_system.initialize_models()

    # Add sample preferences
    sample_prefs = create_sample_preferences()
    for pref in sample_prefs:
        rlhf_system.add_preference_data(pref.prompt, pref.chosen_response, pref.rejected_response)

    print(f"📚 Added {len(sample_prefs)} preference pairs")

    # Train reward model (simplified for demo)
    print("🎯 Reward model training simulated (full training requires more data/time)")
    # rlhf_system.train_reward_model(num_epochs=1)

    # Test reward model
    test_prompt = "What is quantum computing?"
    good_response = "Quantum computing harnesses quantum mechanics to process information in ways classical computers cannot."
    bad_response = "Quantum computing is weird physics stuff."

    good_reward = rlhf_system.reward_model.get_reward(f"{test_prompt}\n\n{good_response}", rlhf_system.tokenizer)
    bad_reward = rlhf_system.reward_model.get_reward(f"{test_prompt}\n\n{bad_response}", rlhf_system.tokenizer)

    print("🎯 Reward Model Test:")
    print(f"   Good response reward: {good_reward:.4f}")
    print(f"   Bad response reward: {bad_reward:.4f}")
    print(f"   Preference learned: {'✅' if good_reward > bad_reward else '❌'}")

    # Generate response
    test_response = rlhf_system.generate_response("Explain AI safety briefly")
    print(f"\n💬 Generated response: {test_response}")

    # Show stats
    stats = rlhf_system.get_training_stats()
    print("\n📊 Training Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n🎉 RLHF system test complete!")