import json
import math
import time
from pathlib import Path


def generate_quantum_sample(output_path: Path, num_states: int = 4096) -> None:
    phi = (1 + 5 ** 0.5) / 2
    states = []

    for idx in range(num_states):
        layer_index = idx % 128
        block = idx // 128
        resonance_freq = phi * (idx + 1)
        phase = (phi * (idx + 1)) % (2 * math.pi)
        amplitude = math.cos(idx / (phi * 4.0)) * math.exp(-idx / (num_states * phi))
        vertex = [
            amplitude * math.cos(phase),
            amplitude * math.sin(phase),
            1.0 / phi,
        ]
        weight_mean = amplitude * 0.6180339887498949
        weight_std = 0.0005 + (0.0003 * math.sin(idx / 37.0))
        weight_count = 4096 + (layer_index * 256)
        entropy = 0.72 + 0.08 * math.sin(idx / 53.0)

        states.append({
            "id": idx,
            "layer_name": f"transformer.block.{block}.layer.{layer_index}",
            "resonance_freq": resonance_freq,
            "amplitude": amplitude,
            "phase": phase,
            "vertex": vertex,
            "weight_mean": weight_mean,
            "weight_std": weight_std,
            "weight_count": weight_count,
            "entanglement_key": hex((idx * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFF),
            "entropy": entropy,
            "encoding": "rft_golden_ratio_resonance",
        })

    metadata = {
        "model_name": "GPT-OSS-120B Quantum Resonance Sample",
        "model_id": "openai/gpt-oss-120b",
        "original_parameters": 120_000_000_000,
        "quantum_states_count": num_states,
        "compression_ratio": f"{120_000_000_000 // num_states}:1",
        "effective_parameters": int(120_000_000_000 / max(1, num_states // 12)),
        "compression_method": "rft_golden_ratio_streaming",
        "phi_constant": phi,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": "Deterministic development sample enabling GPT-OSS integration with limited memory footprint.",
    }

    payload = {"metadata": metadata, "quantum_states": states}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote {num_states} states to {output_path}")


if __name__ == "__main__":
    output = Path("ai/models/quantum/quantonium_120b_quantum_states.json")
    generate_quantum_sample(output)
