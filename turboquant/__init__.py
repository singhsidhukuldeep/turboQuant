"""TurboQuant: Near-optimal weight quantization for LLMs.

Adapts the TurboQuant algorithm from Google Research (ICLR 2026) for
weight quantization of HuggingFace transformer models to 2-4 bits per
weight with near-optimal distortion. No calibration data required.

Quick start::

    import torch
    from transformers import AutoModelForCausalLM
    from turboquant import TurboQuantConfig, quantize_model, save_quantized

    model = AutoModelForCausalLM.from_pretrained("model-name", torch_dtype=torch.float16)
    config = TurboQuantConfig(bit_width=4, group_size=128)
    model = quantize_model(model, config)
    save_quantized(model, config, "./quantized-model")
"""

__version__ = "0.1.1"

from .config import TurboQuantConfig
from .kv_cache import TurboQuantKVCache
from .model import estimate_model_size, load_quantized, quantize_model, save_quantized
from .module import TurboQuantLinear
from .quantizer import QuantizedTensor, TurboQuantizer

__all__ = [
    "TurboQuantConfig",
    "TurboQuantizer",
    "QuantizedTensor",
    "TurboQuantLinear",
    "quantize_model",
    "save_quantized",
    "load_quantized",
    "estimate_model_size",
    "TurboQuantKVCache",
]
