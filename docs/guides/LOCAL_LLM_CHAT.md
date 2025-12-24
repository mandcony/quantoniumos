# Local LLM / CLI Chat

This repo includes a small local-inference chat runner for environments where the GUI is unavailable (e.g. containers without an X server).

## Run the CLI chat

```bash
python src/apps/cli_chat.py
```

Type `exit` or `quit` to stop.

## One-shot (piped) mode

Useful for scripting:

```bash
printf "Explain superposition in one sentence.\n" | python src/apps/cli_chat.py
```

The CLI reads stdin once, prints a single response, then exits.

## Choose a model

Set the model id via `QUANTONIUM_MODEL_ID`.

Examples:

```bash
# Small + fast (quality is limited)
QUANTONIUM_MODEL_ID=distilgpt2 python src/apps/cli_chat.py

# Better chat quality (slower on CPU)
QUANTONIUM_MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 python src/apps/cli_chat.py
```

Notes:
- Some HuggingFace models are gated (require auth) and will fail without credentials.
- CPU inference with ~1B parameter models will be slow.

## Fully-offline operation (no downloads)

1) Cache the model while you have network access:

```bash
python src/apps/cache_local_llm.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --cache-dir ai/hf_cache
```

2) Run offline using the cache:

```bash
HF_HOME=ai/hf_cache QUANTONIUM_LOCAL_ONLY=1 \
  QUANTONIUM_MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  python src/apps/cli_chat.py
```

## Prompt formatting

The wrapper auto-selects a prompt format based on the configured model:
- TinyLlama *Chat* models use ChatML (`<|system|>`, `<|user|>`, `<|assistant|>`)
- Other models use a simple `System:/User:/Assistant:` text format

## Experimental: quantum-weight reconstruction

Quantum-weight JSON files exist under `ai/models/quantum/`, but full weight reconstruction is intentionally disabled in interactive chat by default because it can be extremely slow and memory-intensive.

If you want to experiment, use the loader directly and expect long runtimes:

```bash
python src/apps/fast_quantum_loader.py --json ai/models/quantum/tinyllama_real_quantum_compressed.json
```
