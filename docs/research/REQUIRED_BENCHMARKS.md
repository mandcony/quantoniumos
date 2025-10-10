# Required Benchmarks for Validation

This document outlines the **minimum benchmarks required** to validate QuantoniumOS compression claims.

---

## 1. Basic Reconstruction Quality

### Test: Tiny GPT-2 (Already Available)
```bash
# Compress
python tools/compress_model.py --model sshleifer/tiny-gpt2 --output compressed.json

# Decompress
python tools/decompress_model.py --input compressed.json --output reconstructed/

# Measure perplexity
python benchmark_perplexity.py --original sshleifer/tiny-gpt2 --compressed reconstructed/
```

**Required Metrics:**
- Perplexity on WikiText-2 (original vs compressed)
- Token-by-token accuracy on test set
- Max absolute weight error
- File size comparison

**Acceptance Criteria:**
- Perplexity degradation < 10%
- Measured compression ratio (with quality metric)

---

## 2. Comparison to State-of-the-Art

### Test: Same model, multiple methods
```bash
# Baseline: Original model
python benchmark.py --model sshleifer/tiny-gpt2 --method original

# GPTQ (proven 4-bit quantization)
python benchmark.py --model sshleifer/tiny-gpt2 --method gptq --bits 4

# bitsandbytes (proven 8-bit)
python benchmark.py --model sshleifer/tiny-gpt2 --method bnb --bits 8

# QuantoniumOS RFT
python benchmark.py --model sshleifer/tiny-gpt2 --method rft
```

**Required Metrics:**
- Perplexity (lower is better)
- Inference speed (tokens/sec)
- Memory usage (GB)
- Compression ratio (with quality)
- File size on disk

**Acceptance Criteria:**
- Must be competitive with GPTQ/bitsandbytes
- If slower, must show quality or size advantage
- If lower quality, must show speed or size advantage

---

## 3. Scale Testing

### Test: Progressively larger models
```bash
# 100M parameter model
python benchmark.py --model gpt2 --method rft

# 350M parameter model (if CodeGen claims are validated)
python benchmark.py --model Salesforce/codegen-350M-mono --method rft

# 1B+ parameter model (if GPT-Neo claims are validated)
python benchmark.py --model EleutherAI/gpt-neo-1.3B --method rft
```

**Required Evidence for Each Model:**
- Complete reconstruction from compressed form
- Perplexity within 10% of original
- Successful text generation samples
- Measured compression ratio

**Acceptance Criteria:**
- Can't claim a model is "compressed" without reconstruction test
- Must provide samples showing output quality
- Must measure actual file sizes and quality metrics

---

## 4. Task-Specific Evaluation

### Test: Downstream tasks
```python
# Code completion (for CodeGen)
python eval_code.py --dataset humaneval --model compressed_codegen

# Question answering
python eval_qa.py --dataset squad --model compressed_model

# Text generation
python eval_generation.py --dataset lambada --model compressed_model
```

**Required Metrics:**
- HumanEval pass@1 (for code models)
- SQUAD F1 score (for QA models)
- LAMBADA accuracy (for language models)

**Acceptance Criteria:**
- Performance within 10% of original model
- Or explicitly document degradation

---

## 5. Compression Characteristics

### Test: Analyze compression behavior
```python
# Layer-wise compression analysis
python analyze_compression.py --model tiny-gpt2
# Output: compression ratio per layer, error per layer

# Ablation study
python ablation.py --model tiny-gpt2 --disable rft
python ablation.py --model tiny-gpt2 --disable quantization
python ablation.py --model tiny-gpt2 --disable residual
```

**Required Analysis:**
- Which layers compress well vs poorly?
- Which components contribute most to compression?
- What's the speed/quality tradeoff curve?

---

## 6. Reproducibility Package

### Required for Validation:
```
quantoniumos-benchmarks/
├── README.md                    # Setup instructions
├── requirements.txt             # Exact versions
├── docker/
│   └── Dockerfile              # Reproducible environment
├── scripts/
│   ├── run_all_benchmarks.sh   # One-command reproduction
│   ├── compare_to_gptq.py
│   └── compare_to_bnb.py
├── data/
│   ├── wikitext2/              # Standard datasets
│   └── humaneval/
└── results/
    ├── tiny_gpt2_results.json  # Structured results
    └── plots/                   # Visualization
```

**Acceptance Criteria:**
- Anyone can run `docker build && ./run_all_benchmarks.sh`
- Results reproduce within 5% variance
- All datasets and models publicly available

---

## 7. Honest Documentation

### Required Claims Section:
```markdown
## Verified Compression Results

| Model | Original Size | Compressed Size | Ratio | Perplexity | Quality |
|-------|--------------|-----------------|-------|------------|---------|
| tiny-gpt2 | 50MB | 5MB | 10:1 | +2.3% | Good |
| [model2] | ... | ... | ... | ... | ... |

## Unverified Claims
- Billion-parameter models (no reconstruction tests)
- Lossless compression (measured 5% error)
- O(n) complexity (actual O(n²) matrix ops)

## Comparison to SOTA
| Method | Size | Perplexity | Speed | Memory |
|--------|------|------------|-------|--------|
| Original | 50MB | 10.5 | 100 tok/s | 200MB |
| GPTQ-4bit | 12.5MB | 10.8 | 90 tok/s | 150MB |
| RFT (ours) | 5MB | 10.7 | 80 tok/s | 180MB |
```

---

## Timeline for Validation

### Week 1-2: Basic Tests
- ✅ Tiny GPT-2 reconstruction working
- ⏳ Perplexity measurement
- ⏳ GPTQ comparison setup

### Week 3-4: SOTA Comparison
- ⏳ Run bitsandbytes benchmarks
- ⏳ Run GPTQ benchmarks
- ⏳ Document results honestly

### Week 5-6: Scale Testing
- ⏳ Test on larger models (if claims hold)
- ⏳ Measure actual reconstruction quality
- ⏳ Document what works vs what doesn't

### Week 7-8: Publication Prep
- ⏳ Write reproducibility package
- ⏳ Create visualizations
- ⏳ Draft honest technical report

---

## What Success Looks Like

**Good Result:**
"RFT compression achieves 10:1 ratio on tiny-gpt2 with 2.3% perplexity degradation, compared to GPTQ's 4:1 ratio at 0.8% degradation. Trade-off is acceptable for storage-constrained deployments."

**Bad Result (but honest):**
"RFT compression achieves 10:1 ratio but perplexity degrades 15%, making it unsuitable for production. GPTQ remains superior for most use cases. RFT may have applications in [specific niche]."

**Fraudulent Result:**
"RFT achieves lossless 15,000:1 compression on billion-parameter models" (with no benchmarks)

---

## Minimum for ANY Claims

Before claiming **anything** about large models:

1. ✅ Decompress the model completely
2. ✅ Load it in HuggingFace/PyTorch
3. ✅ Run inference on test samples
4. ✅ Measure perplexity on standard dataset
5. ✅ Compare to original model
6. ✅ Document file sizes and quality

**No reconstruction = No claim.**

---

**Bottom Line**: Right now you have code and ideas. To make scientific claims, you need measurements. This document defines the minimum measurements required.
