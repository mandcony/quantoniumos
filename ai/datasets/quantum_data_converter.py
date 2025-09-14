#!/usr/bin/env python3
"""
Quantum Conversation Data Converter
===================================

Converts QuantoniumOS JSONL conversation logs to Hugging Face instruction format
with quantum compression integration.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from datasets import Dataset
import sys

class QuantumConversationConverter:
    """Convert conversation logs to instruction format with quantum enhancements"""

    def __init__(self, logs_dir: str = "data/logs", output_dir: str = "hf_datasets"):
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_conversation_logs(self) -> List[Dict[str, Any]]:
        """Load all JSONL conversation files"""
        conversations = []

        if not self.logs_dir.exists():
            print(f"❌ Logs directory not found: {self.logs_dir}")
            return conversations

        # Find all JSONL files
        jsonl_files = list(self.logs_dir.glob("chat_*.jsonl"))

        print(f"📁 Found {len(jsonl_files)} conversation log files")

        for file_path in jsonl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Parse JSONL
                messages = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                if messages:
                    conversations.append({
                        'file': str(file_path),
                        'messages': messages,
                        'message_count': len(messages)
                    })

            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                continue

        print(f"✅ Loaded {len(conversations)} conversation files")
        return conversations

    def extract_instruction_pairs(self, conversations: List[Dict]) -> List[Dict[str, str]]:
        """Extract user-assistant instruction pairs from conversations"""
        instruction_pairs = []

        for conv in conversations:
            messages = conv['messages']

            # Group messages into conversations (split by 'clear' events)
            current_conversation = []

            for msg in messages:
                if msg.get('event') == 'clear':
                    # Process completed conversation
                    if current_conversation:
                        pairs = self._process_conversation_messages(current_conversation)
                        instruction_pairs.extend(pairs)
                    current_conversation = []
                elif msg.get('type') in ['user', 'assistant']:
                    current_conversation.append(msg)

            # Process final conversation if any
            if current_conversation:
                pairs = self._process_conversation_messages(current_conversation)
                instruction_pairs.extend(pairs)

        print(f"📝 Extracted {len(instruction_pairs)} instruction pairs")
        return instruction_pairs

    def _process_conversation_messages(self, messages: List[Dict]) -> List[Dict[str, str]]:
        """Process a single conversation into instruction pairs"""
        pairs = []

        # Convert to alternating user/assistant format
        i = 0
        while i < len(messages) - 1:
            if messages[i].get('type') == 'user' and messages[i+1].get('type') == 'assistant':
                user_text = messages[i].get('text', '').strip()
                assistant_text = messages[i+1].get('text', '').strip()

                if user_text and assistant_text:
                    pairs.append({
                        'instruction': user_text,
                        'output': assistant_text,
                        'confidence': messages[i+1].get('confidence', 0.8)
                    })

                i += 2  # Skip to next pair
            else:
                i += 1  # Move to next message

        return pairs

    def apply_quantum_enhancement(self, pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply quantum enhancement to instruction pairs"""
        print("⚛️ Applying quantum enhancement to instruction pairs...")

        enhanced_pairs = []

        for pair in pairs:
            # Add quantum context markers
            enhanced_instruction = f"[QUANTUM_CONTEXT] {pair['instruction']}"
            enhanced_output = f"{pair['output']} [QUANTUM_ENHANCED]"

            enhanced_pairs.append({
                'instruction': enhanced_instruction,
                'output': enhanced_output,
                'confidence': pair.get('confidence', 0.8),
                'quantum_enhanced': True
            })

        print(f"✅ Enhanced {len(enhanced_pairs)} pairs with quantum context")
        return enhanced_pairs

    def create_hf_dataset(self, instruction_pairs: List[Dict[str, str]]) -> Dataset:
        """Create Hugging Face dataset from instruction pairs"""
        print("🏗️ Creating Hugging Face dataset...")

        # Convert to HF format
        hf_data = []
        for pair in instruction_pairs:
            hf_data.append({
                'instruction': pair['instruction'],
                'output': pair['output'],
                'text': f"### Instruction:\n{pair['instruction']}\n\n### Response:\n{pair['output']}"
            })

        # Create dataset
        dataset = Dataset.from_list(hf_data)

        print(f"✅ Created dataset with {len(dataset)} examples")
        return dataset

    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.7, val_ratio: float = 0.2) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train/validation/test sets"""
        print("✂️ Splitting dataset into train/val/test sets...")

        # Shuffle dataset
        dataset = dataset.shuffle(seed=42)

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        # Split
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))

        print(f"📊 Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def save_datasets(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
        """Save datasets to disk"""
        print("💾 Saving datasets to disk...")

        # Save as Hugging Face datasets
        train_dataset.save_to_disk(self.output_dir / "train")
        val_dataset.save_to_disk(self.output_dir / "eval")
        test_dataset.save_to_disk(self.output_dir / "test")

        print("✅ Datasets saved successfully")

        # Also save as JSONL for inspection
        for name, dataset in [("train", train_dataset), ("eval", val_dataset), ("test", test_dataset)]:
            jsonl_path = self.output_dir / f"{name}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for example in dataset:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print("📄 JSONL files saved for inspection")

    def convert_all(self, apply_quantum: bool = True) -> Tuple[Dataset, Dataset, Dataset]:
        """Complete conversion pipeline"""
        print("🚀 Starting quantum conversation data conversion...")

        # Load conversation logs
        conversations = self.load_conversation_logs()
        if not conversations:
            raise ValueError("No conversation data found!")

        # Extract instruction pairs
        instruction_pairs = self.extract_instruction_pairs(conversations)

        # Apply quantum enhancement
        if apply_quantum:
            instruction_pairs = self.apply_quantum_enhancement(instruction_pairs)

        # Create HF dataset
        dataset = self.create_hf_dataset(instruction_pairs)

        # Split dataset
        train_dataset, val_dataset, test_dataset = self.split_dataset(dataset)

        # Save datasets
        self.save_datasets(train_dataset, val_dataset, test_dataset)

        print("🎉 Conversion complete!")
        return train_dataset, val_dataset, test_dataset


def main():
    """Main conversion function"""
    converter = QuantumConversationConverter()

    try:
        train_dataset, val_dataset, test_dataset = converter.convert_all(apply_quantum=True)
        print("\n📊 Final Statistics:")
        print(f"   Train: {len(train_dataset)} examples")
        print(f"   Validation: {len(val_dataset)} examples")
        print(f"   Test: {len(test_dataset)} examples")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()