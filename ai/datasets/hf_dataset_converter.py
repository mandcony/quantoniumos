#!/usr/bin/env python3
"""
QUANTONIUMOS → HUGGING FACE DATASET CONVERTER
==============================================

Converts your JSONL conversation logs into HuggingFace dataset format
with proper train/validation/test splits and quality filtering.

Author: QuantoniumOS AI Pipeline
Date: 2025-01-09
Version: 1.0
"""

import json
import os
import glob
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import pandas as pd
from datasets import Dataset, DatasetDict

class JSONLToHFConverter:
    """Convert QuantoniumOS JSONL logs to HuggingFace datasets"""
    
    def __init__(self, logs_dir: str = "data/logs", output_dir: str = "hf_datasets"):
        self.logs_dir = logs_dir
        self.output_dir = output_dir
        self.min_confidence = 0.8
        self.min_messages = 4  # Minimum messages per conversation
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'valid_conversations': 0,
            'total_messages': 0,
            'filtered_messages': 0,
            'average_confidence': 0.0,
            'confidence_distribution': defaultdict(int)
        }
    
    def load_jsonl_files(self) -> List[Dict[str, Any]]:
        """Load all JSONL conversation files"""
        print("📂 Loading JSONL files...")
        
        # Find all chat JSONL files
        jsonl_files = glob.glob(os.path.join(self.logs_dir, "chat_*.jsonl"))
        self.stats['total_files'] = len(jsonl_files)
        
        all_conversations = []
        
        for file_path in jsonl_files:
            print(f"   Processing: {os.path.basename(file_path)}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    messages = []
                    for line in f:
                        if line.strip():
                            msg = json.loads(line.strip())
                            messages.append(msg)
                    
                    if len(messages) >= self.min_messages:
                        conversation_id = os.path.splitext(os.path.basename(file_path))[0]
                        
                        # Extract metadata from filename
                        timestamp = conversation_id.replace('chat_', '').replace('_', '')
                        
                        conversation = {
                            'id': conversation_id,
                            'messages': messages,
                            'timestamp': timestamp,
                            'source_file': file_path,
                            'message_count': len(messages)
                        }
                        
                        all_conversations.append(conversation)
                        self.stats['valid_conversations'] += 1
                        self.stats['total_messages'] += len(messages)
                        
            except Exception as e:
                print(f"   ⚠️ Error processing {file_path}: {e}")
        
        print(f"✅ Loaded {len(all_conversations)} valid conversations from {len(jsonl_files)} files")
        return all_conversations
    
    def calculate_conversation_quality(self, conversation: Dict[str, Any]) -> float:
        """Calculate overall quality score for a conversation"""
        messages = conversation['messages']
        confidences = []
        
        for msg in messages:
            if 'confidence' in msg and msg['confidence'] is not None:
                confidences.append(float(msg['confidence']))
            elif msg.get('role') == 'assistant':
                # Default confidence for assistant messages without scores
                confidences.append(0.7)
        
        if not confidences:
            return 0.5  # Default quality for conversations without confidence scores
            
        avg_confidence = sum(confidences) / len(confidences)
        
        # Bonus for longer conversations
        length_bonus = min(0.1, len(messages) / 20)
        
        # Bonus for balanced conversations (user/assistant ratio)
        user_msgs = sum(1 for msg in messages if msg.get('role') == 'user')
        assistant_msgs = sum(1 for msg in messages if msg.get('role') == 'assistant')
        
        if user_msgs > 0 and assistant_msgs > 0:
            balance_ratio = min(user_msgs, assistant_msgs) / max(user_msgs, assistant_msgs)
            balance_bonus = balance_ratio * 0.1
        else:
            balance_bonus = 0
        
        final_quality = avg_confidence + length_bonus + balance_bonus
        return min(1.0, final_quality)  # Cap at 1.0
    
    def filter_and_clean_conversations(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter conversations by quality and clean message format"""
        print("🧹 Filtering and cleaning conversations...")
        
        cleaned_conversations = []
        total_confidence = 0
        confidence_count = 0
        
        for conv in conversations:
            quality_score = self.calculate_conversation_quality(conv)
            
            if quality_score >= self.min_confidence:
                # Clean and standardize messages
                cleaned_messages = []
                
                for msg in conv['messages']:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    confidence = msg.get('confidence')
                    timestamp = msg.get('timestamp', '')
                    
                    # Skip empty messages or system messages
                    if not content or role == 'system':
                        continue
                    
                    # Standardize role names
                    if role in ['human', 'user']:
                        role = 'user'
                    elif role in ['assistant', 'ai', 'bot']:
                        role = 'assistant'
                    
                    cleaned_msg = {
                        'role': role,
                        'content': content.strip(),
                        'timestamp': timestamp
                    }
                    
                    if confidence is not None:
                        cleaned_msg['confidence'] = float(confidence)
                        total_confidence += float(confidence)
                        confidence_count += 1
                    
                    cleaned_messages.append(cleaned_msg)
                
                if len(cleaned_messages) >= self.min_messages:
                    conv['messages'] = cleaned_messages
                    conv['quality_score'] = quality_score
                    cleaned_conversations.append(conv)
                    
                    # Update confidence distribution
                    conf_bucket = int(quality_score * 10) / 10
                    self.stats['confidence_distribution'][conf_bucket] += 1
        
        self.stats['filtered_messages'] = sum(len(conv['messages']) for conv in cleaned_conversations)
        
        if confidence_count > 0:
            self.stats['average_confidence'] = total_confidence / confidence_count
        
        print(f"✅ Filtered to {len(cleaned_conversations)} high-quality conversations")
        print(f"   Average confidence: {self.stats['average_confidence']:.3f}")
        print(f"   Total messages after filtering: {self.stats['filtered_messages']}")
        
        return cleaned_conversations
    
    def create_train_val_test_splits(self, conversations: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """Create train/validation/test splits maintaining conversation integrity"""
        print("🔀 Creating train/validation/test splits...")
        
        # Sort by quality score (descending) then by message count (descending)
        conversations.sort(key=lambda x: (x['quality_score'], x['message_count']), reverse=True)
        
        total_conversations = len(conversations)
        train_size = int(0.8 * total_conversations)
        val_size = int(0.15 * total_conversations)
        test_size = total_conversations - train_size - val_size
        
        # Ensure minimum sizes
        if val_size == 0 and total_conversations > 2:
            val_size = 1
            train_size -= 1
        if test_size == 0 and total_conversations > 4:
            test_size = 1
            train_size -= 1
        
        train_conversations = conversations[:train_size]
        val_conversations = conversations[train_size:train_size + val_size]
        test_conversations = conversations[train_size + val_size:]
        
        print(f"   Train: {len(train_conversations)} conversations")
        print(f"   Validation: {len(val_conversations)} conversations")
        print(f"   Test: {len(test_conversations)} conversations")
        
        return train_conversations, val_conversations, test_conversations
    
    def format_for_huggingface(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format conversations for HuggingFace dataset structure"""
        formatted_data = {
            'conversation_id': [],
            'messages': [],
            'quality_score': [],
            'message_count': [],
            'timestamp': [],
            'instruction': [],  # First user message
            'response': [],     # First assistant response
            'full_conversation': []  # Complete conversation as string
        }
        
        for conv in conversations:
            messages = conv['messages']
            
            # Extract instruction/response pairs
            instruction = ""
            response = ""
            
            for i, msg in enumerate(messages):
                if msg['role'] == 'user' and not instruction:
                    instruction = msg['content']
                elif msg['role'] == 'assistant' and instruction and not response:
                    response = msg['content']
                    break
            
            # Format full conversation as string
            full_conv = ""
            for msg in messages:
                role = msg['role'].title()
                content = msg['content']
                full_conv += f"{role}: {content}\n\n"
            
            formatted_data['conversation_id'].append(conv['id'])
            formatted_data['messages'].append(messages)
            formatted_data['quality_score'].append(conv['quality_score'])
            formatted_data['message_count'].append(conv['message_count'])
            formatted_data['timestamp'].append(conv['timestamp'])
            formatted_data['instruction'].append(instruction)
            formatted_data['response'].append(response)
            formatted_data['full_conversation'].append(full_conv.strip())
        
        return formatted_data
    
    def create_huggingface_dataset(self, train_convs: List, val_convs: List, test_convs: List) -> DatasetDict:
        """Create HuggingFace DatasetDict from conversation splits"""
        print("🤗 Creating HuggingFace dataset format...")
        
        # Format each split
        train_data = self.format_for_huggingface(train_convs)
        val_data = self.format_for_huggingface(val_convs) if val_convs else train_data
        test_data = self.format_for_huggingface(test_convs) if test_convs else train_data
        
        # Create datasets
        dataset_dict = DatasetDict({
            'train': Dataset.from_dict(train_data),
            'validation': Dataset.from_dict(val_data),
            'test': Dataset.from_dict(test_data)
        })
        
        print(f"✅ Created HuggingFace dataset:")
        print(f"   Train: {len(dataset_dict['train'])} examples")
        print(f"   Validation: {len(dataset_dict['validation'])} examples")
        print(f"   Test: {len(dataset_dict['test'])} examples")
        
        return dataset_dict
    
    def save_dataset(self, dataset: DatasetDict, name: str = "quantoniumos_conversations"):
        """Save dataset to disk and generate metadata"""
        print(f"💾 Saving dataset as '{name}'...")
        
        # Save dataset
        dataset_path = os.path.join(self.output_dir, name)
        dataset.save_to_disk(dataset_path)
        
        # Generate dataset card
        card_content = f"""---
license: mit
language:
- en
task_categories:
- conversational
- text-generation
tags:
- quantoniumos
- quantum-enhanced
- conversations
- ai-training
size_categories:
- n<1K
dataset_info:
  features:
  - name: conversation_id
    dtype: string
  - name: messages
    sequence:
    - name: role
      dtype: string
    - name: content
      dtype: string
    - name: timestamp
      dtype: string
    - name: confidence
      dtype: float64
  - name: quality_score
    dtype: float64
  - name: message_count
    dtype: int64
  - name: timestamp
    dtype: string
  - name: instruction
    dtype: string
  - name: response
    dtype: string
  - name: full_conversation
    dtype: string
  splits:
  - name: train
    num_examples: {len(dataset['train'])}
  - name: validation
    num_examples: {len(dataset['validation'])}
  - name: test
    num_examples: {len(dataset['test'])}
---

# QuantoniumOS Conversation Dataset

This dataset contains high-quality conversational data from the QuantoniumOS AI system, prepared for fine-tuning and evaluation.

## Dataset Statistics

- **Total Conversations**: {self.stats['valid_conversations']}
- **Total Messages**: {self.stats['filtered_messages']}
- **Average Confidence**: {self.stats['average_confidence']:.3f}
- **Quality Threshold**: {self.min_confidence}

## Data Format

Each example contains:
- `conversation_id`: Unique identifier for the conversation
- `messages`: Full conversation with role, content, timestamp, and confidence
- `quality_score`: Overall conversation quality (0.0-1.0)
- `instruction`: First user message in the conversation
- `response`: First assistant response
- `full_conversation`: Complete conversation formatted as text

## Usage

```python
from datasets import load_from_disk
dataset = load_from_disk("{name}")

# Access training data
train_data = dataset['train']
for example in train_data:
    print(f"Instruction: {{example['instruction']}}")
    print(f"Response: {{example['response']}}")
```

## Quality Assurance

- Conversations filtered by confidence score >= {self.min_confidence}
- Minimum {self.min_messages} messages per conversation
- Balanced user/assistant message ratios preferred
- Content cleaned and standardized

Generated by QuantoniumOS HF Dataset Converter v1.0
"""
        
        # Save dataset card
        with open(os.path.join(dataset_path, "README.md"), 'w', encoding='utf-8') as f:
            f.write(card_content)
        
        # Save conversion statistics
        stats_path = os.path.join(dataset_path, "conversion_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✅ Dataset saved to: {dataset_path}")
        return dataset_path
    
    def convert(self, dataset_name: str = "quantoniumos_conversations") -> str:
        """Main conversion pipeline"""
        print("🚀 QUANTONIUMOS → HUGGING FACE DATASET CONVERSION")
        print("=" * 50)
        
        # Load JSONL files
        conversations = self.load_jsonl_files()
        
        if not conversations:
            print("❌ No valid conversations found!")
            return None
        
        # Filter and clean
        cleaned_conversations = self.filter_and_clean_conversations(conversations)
        
        if not cleaned_conversations:
            print("❌ No conversations passed quality filter!")
            return None
        
        # Create splits
        train_convs, val_convs, test_convs = self.create_train_val_test_splits(cleaned_conversations)
        
        # Create HF dataset
        dataset = self.create_huggingface_dataset(train_convs, val_convs, test_convs)
        
        # Save dataset
        dataset_path = self.save_dataset(dataset, dataset_name)
        
        print("\n🎯 CONVERSION SUMMARY")
        print("=" * 30)
        print(f"Input Files: {self.stats['total_files']}")
        print(f"Valid Conversations: {self.stats['valid_conversations']}")
        print(f"Quality Filtered: {len(cleaned_conversations)}")
        print(f"Average Confidence: {self.stats['average_confidence']:.3f}")
        print(f"Output Dataset: {dataset_path}")
        
        print("\n📊 CONFIDENCE DISTRIBUTION")
        for conf, count in sorted(self.stats['confidence_distribution'].items()):
            print(f"   {conf:.1f}: {count} conversations")
        
        print(f"\n✅ Dataset ready for HuggingFace fine-tuning!")
        return dataset_path

def main():
    """Convert QuantoniumOS JSONL logs to HuggingFace dataset format"""
    converter = JSONLToHFConverter()
    
    # Convert with default settings
    dataset_path = converter.convert("quantoniumos_conversations")
    
    if dataset_path:
        print(f"\n🎉 SUCCESS! Dataset available at: {dataset_path}")
        print("\nNext Steps:")
        print("1. pip install transformers datasets accelerate")
        print("2. Load dataset: dataset = load_from_disk('hf_datasets/quantoniumos_conversations')")
        print("3. Start fine-tuning with your quantum-enhanced trainers!")
    else:
        print("❌ Conversion failed. Check your JSONL files.")

if __name__ == "__main__":
    main()
