# tests/benchmarks/downstream_performance.py
"""
Evaluates the actual impact of RFT compression on a real downstream task.

This script replaces the previous simulation with a scientifically-grounded
benchmark. It measures the performance degradation of a fine-tuned language
model after a full cycle of RFT compression and decompression.

**Methodology:**
1.  **Model and Task Selection**: A pre-trained BERT model fine-tuned for the
    Microsoft Research Paraphrase Corpus (MRPC) task from the GLUE benchmark is used.
    This is a standard binary classification task (are two sentences paraphrases?).
2.  **Baseline Performance**: The accuracy of the original, uncompressed model is
    measured on the MRPC validation set to establish a baseline.
3.  **Full Compression/Decompression Cycle**:
    a. The state dictionary of the original model is copied.
    b. Each tensor (weight/parameter) in the new state dictionary undergoes a
       full encode -> decode cycle using the `rft_hybrid_codec`.
    c. A new model instance is loaded with this compressed-and-decoded state dict.
4.  **Compressed Performance**: The accuracy of the reconstructed model is evaluated
    on the same MRPC validation set.
5.  **Comparison**: The accuracy scores are compared to quantify the real-world
    performance drop caused by the lossy compression.
"""

import torch
import time
import numpy as np
from datetime import datetime
import evaluate
import datasets
from copy import deepcopy

from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Import the actual RFT codec for a full encode/decode cycle
from algorithms.compression.hybrid.rft_hybrid_codec import encode_tensor_hybrid, decode_tensor_hybrid

# --- Configuration ---
# Using a model fine-tuned on MRPC, a standard GLUE task.
MODEL_ID = "textattack/bert-base-uncased-MRPC"
DATASET = "glue"
TASK = "mrpc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Evaluation Function ---
def evaluate_model_accuracy(model, tokenizer, dataset):
    """
    Evaluates the model's accuracy on the given GLUE task dataset.
    """
    print(f"Evaluating model accuracy on {TASK}...")
    metric = evaluate.load(DATASET, TASK)
    model.eval()
    
    for batch in dataset:
        # The tokenizer needs to handle the sentence pairs correctly
        inputs = tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["label"])
        
    results = metric.compute()
    print(f"Evaluation complete. Accuracy: {results['accuracy']:.4f}")
    return results['accuracy']

# --- Main Benchmark Execution ---

def run_benchmark():
    """
    Executes the benchmark and prints a comparative report.
    """
    print("\n" + "=" * 80)
    print(f"** Benchmark Report: Downstream Performance ({datetime.now().isoformat()}) **")
    print(f"** Model: {MODEL_ID}, Task: GLUE/{TASK} on {DEVICE} **")
    print("=" * 80)

    # 1. Load Model, Tokenizer, and Dataset
    print(f"Loading model, tokenizer, and dataset...")
    original_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load and prepare a subset of the validation data for faster testing
    raw_dataset = datasets.load_dataset(DATASET, TASK, split='validation').shuffle(seed=42).select(range(200))

    # --- Baseline Evaluation ---
    print("\n--- Evaluating Baseline Performance (Original Model) ---")
    baseline_accuracy = evaluate_model_accuracy(original_model, tokenizer, raw_dataset)
    print("-" * 60)

    # --- RFT Encode/Decode Cycle ---
    print("\n--- Performing Full RFT Encode/Decode Cycle ---")
    start_time = time.time()
    
    # Create a new state dict to hold the reconstructed weights
    reconstructed_state_dict = deepcopy(original_model.state_dict())
    
    for name, param in reconstructed_state_dict.items():
        # Only compress floating-point tensors
        if param.dtype.is_floating_point:
            print(f"  Processing tensor: {name} ({param.shape})")
            tensor_np = param.cpu().numpy()
            
            # 1. Encode
            encoded_container = encode_tensor_hybrid(tensor_np, tensor_name=name).container
            
            # 2. Decode
            decoded_np = decode_tensor_hybrid(encoded_container)
            
            # 3. Update the tensor in the new state dict
            reconstructed_state_dict[name] = torch.from_numpy(decoded_np).to(param.dtype)

    # Create a new model and load the reconstructed weights
    reconstructed_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    reconstructed_model.load_state_dict(reconstructed_state_dict)
    reconstructed_model.to(DEVICE)
    
    cycle_time = time.time() - start_time
    print(f"Full model encode/decode cycle finished in {cycle_time:.2f}s.")
    print("-" * 60)

    # --- Compressed Model Evaluation ---
    print("\n--- Evaluating Performance (Reconstructed Model) ---")
    reconstructed_accuracy = evaluate_model_accuracy(reconstructed_model, tokenizer, raw_dataset)
    print("-" * 60)

    # --- Final Report ---
    performance_drop = ((baseline_accuracy - reconstructed_accuracy) / baseline_accuracy) * 100

    print("\n" + "=" * 80)
    print(f"** Final Benchmark Results **")
    print("=" * 80)
    print(f"{'State':<25} | {'MRPC Accuracy':>20} | {'Performance Drop':>20}")
    print("-" * 80)
    print(f"{'Original (FP32)':<25} | {baseline_accuracy:>20.4f} | {'N/A':>20}")
    print(f"{'RFT Compressed/Decompressed':<25} | {reconstructed_accuracy:>20.4f} | {f'{performance_drop:.2f}%':>20}")
    print("=" * 80)
    print("\nThis test uses the actual RFT codec and a real GLUE benchmark task.")
    print("It provides a concrete measure of performance degradation after compression.")

if __name__ == "__main__":
    run_benchmark()
