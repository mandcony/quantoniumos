# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Train a small LoRA adapter on your local chat transcripts.

This is the simplest practical way to make the local model "smarter" for *your* domain
without retraining a full LLM.

Input
- Chatbox JSONL logs written by src/apps/qshll_chatbox.py in ./logs/chat_*.jsonl

Output
- A LoRA adapter directory (default: ai/training/models/local_chat_lora)

Usage
- python src/apps/train_local_chat_lora.py --model distilgpt2 --out ai/training/models/local_chat_lora

After training, run chatbox with:
- QUANTONIUM_LORA_PATH=ai/training/models/local_chat_lora QUANTONIUM_MODEL_ID=distilgpt2 python src/apps/qshll_chatbox.py

Notes
- Training is local and can be slow on CPU.
- Keep prompts non-agentic; this script does not add tool use.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def load_chat_pairs(log_glob: str, max_pairs: int = 2000) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for path in sorted(glob.glob(log_glob)):
        last_user = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("type")
                if t == "user":
                    last_user = obj.get("text", "")
                elif t == "assistant" and last_user is not None:
                    pairs.append((last_user, obj.get("text", "")))
                    last_user = None
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def format_example(user_text: str, assistant_text: str) -> str:
    # Match the wrapper formatting.
    return f"User: {user_text.strip()}\nAssistant: {assistant_text.strip()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("QUANTONIUM_MODEL_ID", "distilgpt2"))
    ap.add_argument("--logs", default="logs/chat_*.jsonl")
    ap.add_argument("--corpus", default=None, help="Path to a JSONL corpus with {'text': ...} rows")
    ap.add_argument("--out", default="ai/training/models/local_chat_lora")
    ap.add_argument("--max_pairs", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    args = ap.parse_args()

    if args.corpus:
        # Train on a multi-domain corpus (plain causal LM text)
        if not os.path.exists(args.corpus):
            raise SystemExit(f"Corpus not found: {args.corpus}")
        ds = Dataset.from_json(args.corpus)
        if "text" not in ds.column_names:
            raise SystemExit("Corpus JSONL must contain a 'text' field")
        print(f"Loaded corpus: {args.corpus} rows={len(ds)}")
    else:
        # Train on your chat logs as instruction-style pairs
        pairs = load_chat_pairs(args.logs, max_pairs=args.max_pairs)
        if not pairs:
            raise SystemExit(f"No training pairs found in {args.logs}. Chat a bit first to create logs/chat_*.jsonl")
        texts = [format_example(u, a) for (u, a) in pairs]
        ds = Dataset.from_dict({"text": texts})
        print(f"Loaded chat pairs: {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    # LoRA config: gpt2-like modules commonly use c_attn/c_proj
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"],
    )

    model = get_peft_model(model, lora)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=bool(torch.cuda.is_available()),
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=512,
        packing=True,
        args=training_args,
    )

    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print(f"Saved LoRA adapter to: {args.out}")


if __name__ == "__main__":
    main()
