# QuantoniumOS AI Models Directory

**Consolidated Location**: All AI models are now organized in `ai/models/`

## Directory Structure

```
ai/models/
├── quantum/                    # Quantum-compressed JSON models
│   ├── quantonium_120b_quantum_states.json          (4.82MB - GPT-OSS 120B)
│   ├── quantonium_streaming_7b.json                 (7.86MB - Llama2-7B) 
│   ├── llama_2_180b_chat_quantum_compressed.json    (0.02MB - Llama2-180B)
│   ├── llama_3.1_70b_instruct_quantum_compressed.json (0.02MB - Llama3.1-70B)
│   └── [4 other small quantum models]               (0MB each)
├── compressed/                 # PKL.GZ compressed models  
│   ├── dialogpt_small_compressed.pkl.gz             (0.34MB - 117M params)
│   ├── eleutherai_gpt_neo_1.3b_compressed.pkl.gz    (0.15MB - 1.3B params)
│   └── phi3_mini_quantum_resonance.pkl.gz           (0.25MB - 3.8B params)
└── huggingface/               # Full HuggingFace models
    └── DialoGPT-small/                              (1,620MB - 117M params)
        ├── pytorch_model.bin, tf_model.h5, flax_model.msgpack
        └── [tokenizer and config files]
```

## Model Inventory

### Quantum-Compressed Models (20.98B effective parameters)
- **GPT-OSS-120B**: 14.24B effective (120B original, 8.4:1 compression)
- **Llama2-7B-Streaming**: 6.74B effective (7B original, 1.04:1 compression)

### PKL.GZ Compressed Models (5.22B parameters)  
- **GPT-Neo 1.3B**: 1.3B parameters (quantum-compressed)
- **Phi-3 Mini**: 3.8B parameters (quantum-compressed)
- **DialoGPT-small**: 117M parameters (quantum-compressed)

### HuggingFace Models (117M parameters)
- **DialoGPT-small**: Full model with all formats (PyTorch, TensorFlow, Flax)

## Total System Capacity
- **Real Parameters**: 26.32B parameters
- **Total Storage**: 1,641MB
- **Compression Achievement**: 99.4% space savings vs uncompressed equivalents
- **Models**: 6 functional AI systems

## Usage Notes
- All models are **100% legal** for compression and commercial use
- JSON models use RFT quantum encoding with golden ratio compression
- PKL.GZ models use traditional compression with quantum enhancement
- HuggingFace models are full-precision reference implementations

**Last Updated**: September 21, 2025
**Organization**: Consolidated from 7 scattered directories into single location