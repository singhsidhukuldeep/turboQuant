# turboQuant

Weight quantization for Large Language Models, adapted from the [TurboQuant algorithm](https://arxiv.org/abs/2504.19874) by Google Research (ICLR 2026).

Quantize any HuggingFace model to 2-4 bits per weight with minimal quality loss. No calibration data required.

## Features

- **2/3/4-bit weight quantization** using TurboQuant's random-rotation + Lloyd-Max pipeline
- **No calibration data needed** -- uses mathematical properties of random rotations, not model-specific tuning
- **Residual quantization** -- optional second pass (e.g., 4+4 = 8 bit) for near-lossless compression
- **Wide HuggingFace model support** -- works with Llama, Mistral, Qwen, Gemma, Phi, and other models using `nn.Linear` (CausalLM, Seq2Seq, classification via `--model-class`)
- **KV cache compression** -- runtime cache compression with proper bit-packing for longer contexts
- **CLI included** -- quantize models from the command line
- **Pure PyTorch** -- no CUDA/Triton dependency required

## Installation

```bash
# Core (PyTorch only)
pip install turbo-quant

# With HuggingFace support (recommended)
pip install turbo-quant[transformers]

# Development
pip install turbo-quant[dev]
```

Or install from source:
```bash
git clone https://github.com/singhsidhukuldeep/turboQuant.git
cd turboQuant
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantConfig, quantize_model, save_quantized, load_quantized

# Load a model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Quantize to 4-bit
config = TurboQuantConfig(bit_width=4, group_size=128)
model = quantize_model(model, config)

# Save
save_quantized(model, config, "./qwen-0.5b-tq4", save_tokenizer=True, tokenizer=tokenizer)

# Load later
model = load_quantized("Qwen/Qwen2.5-0.5B", "./qwen-0.5b-tq4")

# Generate
inputs = tokenizer("The meaning of life is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Higher Quality with Residual Quantization

```python
# 4+4 = 8-bit total, near-lossless
config = TurboQuantConfig(bit_width=4, residual_bit_width=4, group_size=128)
model = quantize_model(model, config)
```

### KV Cache Compression

```python
from turboquant import TurboQuantKVCache

cache = TurboQuantKVCache(key_bits=4, value_bits=4, residual_window=128)

# During generation, compress and store:
# cache.update(layer_idx, key_states, value_states)
# keys, values = cache.get(layer_idx)
```

### Command Line

```bash
# Quantize a model
turboquant quantize \
    --model Qwen/Qwen2.5-0.5B \
    --output ./quantized \
    --bits 4 \
    --group-size 128

# Estimate compression ratio
turboquant estimate --model Qwen/Qwen2.5-0.5B --bits 4

# Generate text with a quantized model
turboquant generate \
    --model Qwen/Qwen2.5-0.5B \
    --quantized ./quantized \
    --prompt "Hello, world!"

# Inspect a quantized model
turboquant info ./quantized
```

## How It Works

This library adapts the TurboQuant vector quantization algorithm for model weight compression:

1. **Normalize**: Extract per-group norms (stored in float32)
2. **Rotate**: Apply a random orthogonal transform (Walsh-Hadamard or Haar). After rotation, each coordinate follows a known Beta((d-1)/2, (d-1)/2) distribution regardless of the original weight values
3. **Quantize**: Apply a precomputed Lloyd-Max optimal scalar quantizer tailored to this Beta distribution
4. **Pack**: Bit-pack the quantization indices for compact storage

The key insight from the TurboQuant paper is that random rotation makes the statistical properties of rotated coordinates **predictable and universal**, enabling an optimal quantizer without any calibration data.

### Scope and Relationship to the Paper

The TurboQuant paper (Zandieh et al., 2025) focuses on KV cache compression and nearest-neighbor search. This library applies the paper's core technique (Algorithm 1: random rotation + Lloyd-Max scalar quantization) to **weight quantization**, which is a community-driven adaptation not covered in the original paper.

The paper also describes a two-stage approach (Algorithm 2: MSE + 1-bit QJL correction) for unbiased inner product estimation. Community testing has shown that QJL correction hurts in practice for softmax attention and weight reconstruction, so this library uses the MSE-only quantizer (Algorithm 1) and offers an optional multi-bit residual pass instead.

### Why No Calibration?

Traditional quantization methods (GPTQ, AWQ) require calibration data to determine optimal quantization parameters per-layer. TurboQuant sidesteps this: after rotation, the coordinate distribution is determined by **dimensionality alone**. The optimal codebook is precomputed once for each (dimension, bit-width) pair.

## Supported Configurations

Compression ratios account for per-group float32 norms and remainder column overhead at group_size=128:

| Config | Total Bits | Approx. Compression | Quality |
|--------|-----------|---------------------|---------|
| 4-bit | 4 | ~3.7x | Good for most tasks |
| 3-bit | 3 | ~4.8x | Acceptable for large models (7B+) |
| 2-bit | 2 | ~6.6x | Aggressive, some quality loss |
| 4+4 residual | 8 | ~1.9x | Near-lossless |
| 4+2 residual | 6 | ~2.5x | Balanced |

## Citations

Adapted from research by Google:

- **TurboQuant** ([Zandieh et al., 2025](https://arxiv.org/abs/2504.19874)) -- Online vector quantization with near-optimal distortion rate. ICLR 2026.
- **QJL** ([Zandieh et al., 2024](https://arxiv.org/abs/2406.03482)) -- 1-bit quantized JL transform for KV cache quantization with zero overhead. AAAI 2025.
- **PolarQuant** ([Han et al., 2025](https://arxiv.org/abs/2502.02617)) -- Quantizing KV caches with polar transformation. AISTATS 2026.
