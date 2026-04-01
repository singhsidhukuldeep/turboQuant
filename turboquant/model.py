"""Model-level quantization: quantize, save, and load HuggingFace models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import TurboQuantConfig
from .module import TurboQuantLinear

logger = logging.getLogger(__name__)


def _assign_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]):
    """Load a state dict by direct assignment, handling size mismatches.

    PyTorch's load_state_dict rejects shape mismatches even with assign=True.
    This function bypasses that by setting each parameter/buffer via setattr,
    which is necessary when loading quantized weights into freshly initialized
    modules (whose buffers start as empty tensors).
    """
    model_state = dict(model.named_parameters())
    model_state.update(dict(model.named_buffers()))

    param_names = {n for n, _ in model.named_parameters()}

    loaded = set()
    for key, val in state_dict.items():
        if key in model_state:
            parts = key.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            # Parameters must be wrapped in nn.Parameter; buffers are plain tensors
            if key in param_names:
                val = nn.Parameter(val)
            setattr(obj, parts[-1], val)
            loaded.add(key)

    missing = set(model_state) - loaded
    if missing:
        logger.warning(
            f"Missing keys when loading: {sorted(missing)[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )


def _should_skip(name: str, modules_to_not_convert: list[str]) -> bool:
    """Check if a module should be skipped during quantization."""
    return any(pattern in name for pattern in modules_to_not_convert)


def _find_parent_and_attr(
    module_map: dict[str, nn.Module], name: str, root: nn.Module
) -> tuple[nn.Module, str]:
    """Find the parent module and attribute name for a named module."""
    parts = name.rsplit(".", 1)
    if len(parts) == 1:
        return root, parts[0]
    return module_map[parts[0]], parts[1]


def quantize_model(
    model: nn.Module,
    config: TurboQuantConfig,
    verbose: bool = True,
) -> nn.Module:
    """Quantize all eligible nn.Linear layers in a model using TurboQuant.

    Replaces nn.Linear layers with TurboQuantLinear in-place.

    Args:
        model: The model to quantize (e.g., from AutoModelForCausalLM.from_pretrained).
        config: TurboQuant configuration.
        verbose: Log progress.

    Returns:
        The model with quantized linear layers (modified in-place).
    """
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Quantizing model ({total_params:,} parameters) "
            f"with {config.total_bits}-bit TurboQuant"
        )

    # Build module map once (not per-layer)
    module_map = dict(model.named_modules())

    layers_quantized = 0
    layers_skipped = 0
    replacements: list[tuple[nn.Module, str, TurboQuantLinear]] = []

    for name, module in module_map.items():
        if not isinstance(module, nn.Linear):
            continue

        if _should_skip(name, config.modules_to_not_convert):
            if verbose:
                logger.info(f"  Skipping {name}")
            layers_skipped += 1
            continue

        if verbose:
            logger.info(
                f"  Quantizing {name} "
                f"({module.weight.shape[0]}x{module.weight.shape[1]})"
            )

        quantized = TurboQuantLinear.from_linear(module, config)
        layers_quantized += 1

        parent, attr = _find_parent_and_attr(module_map, name, model)
        replacements.append((parent, attr, quantized))

    for parent, attr, quantized in replacements:
        setattr(parent, attr, quantized)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        logger.info(
            f"Quantized {layers_quantized} layers, skipped {layers_skipped}"
        )

    return model


def save_quantized(
    model: nn.Module,
    config: TurboQuantConfig,
    output_dir: str | Path,
    save_tokenizer: bool = False,
    tokenizer: Any = None,
) -> None:
    """Save a quantized model to disk.

    Creates a directory with:
        - turboquant_config.json: Quantization configuration
        - quantized_weights.pt: All quantized parameters

    Args:
        model: Quantized model.
        config: TurboQuant configuration.
        output_dir: Output directory path.
        save_tokenizer: Whether to save the tokenizer.
        tokenizer: HuggingFace tokenizer to save alongside the model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config.save(output_dir / "turboquant_config.json")

    module_map = dict(model.named_modules())
    state_dict: dict[str, Tensor] = {}

    # Save quantized and skipped linear layers
    for name, module in module_map.items():
        if isinstance(module, TurboQuantLinear):
            prefix = name + "."
            for buf_name, buf in module.named_buffers(recurse=False):
                state_dict[prefix + buf_name] = buf.cpu()
            if module.bias is not None:
                state_dict[prefix + "bias"] = module.bias.data.cpu()
        elif isinstance(module, nn.Linear):
            prefix = name + "."
            state_dict[prefix + "weight"] = module.weight.data.cpu()
            if module.bias is not None:
                state_dict[prefix + "bias"] = module.bias.data.cpu()

    # Save non-linear parameters (embeddings, norms, etc.)
    linear_module_names = {
        name
        for name, m in module_map.items()
        if isinstance(m, (nn.Linear, TurboQuantLinear))
    }
    for name, param in model.named_parameters():
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        if parent_name not in linear_module_names and name not in state_dict:
            state_dict[name] = param.data.cpu()

    for name, buf in model.named_buffers():
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        if parent_name not in linear_module_names and name not in state_dict:
            state_dict[name] = buf.cpu()

    torch.save(state_dict, output_dir / "quantized_weights.pt")

    if hasattr(model, "config"):
        model.config.save_pretrained(output_dir)

    if save_tokenizer and tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    logger.info(f"Saved quantized model to {output_dir}")


def _get_model_class(model_name_or_path: str, task_hint: Optional[str] = None):
    """Import and return the appropriate HuggingFace Auto model class.

    Args:
        model_name_or_path: Model identifier (used for error messages).
        task_hint: Optional hint like "causal-lm", "seq2seq", "classification".
            Defaults to "causal-lm".
    """
    try:
        import transformers
    except ImportError:
        raise ImportError(
            "transformers is required for loading models. "
            "Install with: pip install turboquant-hf[transformers]"
        )

    hint_map = {
        "causal-lm": "AutoModelForCausalLM",
        "seq2seq": "AutoModelForSeq2SeqLM",
        "classification": "AutoModelForSequenceClassification",
        "token-classification": "AutoModelForTokenClassification",
        "question-answering": "AutoModelForQuestionAnswering",
        "auto": "AutoModel",
    }

    if task_hint and task_hint in hint_map:
        class_name = hint_map[task_hint]
        return getattr(transformers, class_name)

    # Default: try CausalLM first, fall back to AutoModel
    return getattr(transformers, "AutoModelForCausalLM")


def load_quantized(
    model_name_or_path: str,
    quantized_path: str | Path,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    model_class: Optional[str] = None,
) -> nn.Module:
    """Load a TurboQuant quantized model.

    Args:
        model_name_or_path: HuggingFace model name or local path for the base
            architecture. If quantized_path contains a model config, it's used instead.
        quantized_path: Path to the directory saved by save_quantized().
        device_map: Device map for model loading (e.g., "auto", "cpu", "cuda").
        torch_dtype: Data type for non-quantized parameters.
        trust_remote_code: Whether to trust remote code when loading the model.
        model_class: Optional model class hint: "causal-lm", "seq2seq",
            "classification", "auto". Defaults to "causal-lm".

    Returns:
        Model with quantized weights loaded.
    """
    quantized_path = Path(quantized_path)

    config = TurboQuantConfig.load(quantized_path / "turboquant_config.json")
    ModelClass = _get_model_class(model_name_or_path, task_hint=model_class)

    # Prefer model config from quantized path if available
    config_path = quantized_path / "config.json"
    model_source = str(quantized_path) if config_path.exists() else model_name_or_path

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        model_source, trust_remote_code=trust_remote_code
    )

    # Initialize model on meta device to avoid loading full weights
    with torch.device("meta"):
        model = ModelClass.from_config(
            model_config, trust_remote_code=trust_remote_code
        )

    # Replace linear layers with TurboQuantLinear
    module_map = dict(model.named_modules())
    replacements = []

    for name, module in module_map.items():
        if not isinstance(module, nn.Linear):
            continue
        if _should_skip(name, config.modules_to_not_convert):
            continue

        tql = TurboQuantLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            config=config,
        )
        parent, attr = _find_parent_and_attr(module_map, name, model)
        replacements.append((parent, attr, tql))

    for parent, attr, tql in replacements:
        setattr(parent, attr, tql)

    # Load quantized state dict
    state_dict = torch.load(
        quantized_path / "quantized_weights.pt",
        map_location="cpu",
        weights_only=True,
    )

    # Direct assignment to handle size-mismatched buffers (empty init vs
    # populated checkpoint). PyTorch's load_state_dict rejects shape
    # mismatches even with assign=True.
    _assign_state_dict(model, state_dict)

    # Move to device
    if device_map is not None:
        if device_map == "auto":
            try:
                from accelerate import dispatch_model, infer_auto_device_map

                device_map_computed = infer_auto_device_map(model)
                model = dispatch_model(model, device_map_computed)
            except ImportError:
                model = model.to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            model = model.to(device_map)

    if torch_dtype is not None:
        linear_names = {
            name
            for name, m in model.named_modules()
            if isinstance(m, TurboQuantLinear)
        }
        for name, param in model.named_parameters():
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            if parent_name not in linear_names:
                param.data = param.data.to(torch_dtype)

    return model


def estimate_model_size(
    model: nn.Module,
    config: TurboQuantConfig,
) -> dict[str, float]:
    """Estimate the size of a model before and after quantization.

    Should be called on the original (unquantized) model. Accounts for
    packed index storage, per-group float32 norms, and remainder columns.

    Args:
        model: The model to analyze.
        config: TurboQuant configuration.

    Returns:
        Dictionary with size estimates in MB.
    """
    original_size_bytes = 0
    quantized_size_bytes = 0
    module_map = dict(model.named_modules())
    linear_names: set[str] = set()

    for name, module in module_map.items():
        if not isinstance(module, nn.Linear):
            continue
        linear_names.add(name)

        rows, cols = module.weight.shape
        param_bytes = rows * cols * module.weight.element_size()
        original_size_bytes += param_bytes

        if module.bias is not None:
            bias_bytes = rows * module.bias.element_size()
            original_size_bytes += bias_bytes
            quantized_size_bytes += rows * 2  # bias stored as float16

        if _should_skip(name, config.modules_to_not_convert):
            quantized_size_bytes += param_bytes
        else:
            n_full_groups = cols // config.group_size
            remainder = cols % config.group_size

            # Packed indices for full groups
            indices_bits = rows * n_full_groups * config.group_size * config.bit_width
            packed_bytes = (indices_bits + 7) // 8

            # Float32 norms per group
            norms_bytes = rows * n_full_groups * 4

            # Remainder stored as float16
            remainder_bytes = rows * remainder * 2

            q_bytes = packed_bytes + norms_bytes + remainder_bytes

            if config.residual_bit_width > 0:
                rgs = config.residual_group_size or config.group_size
                r_full = cols // rgs
                r_rem = cols % rgs
                r_bits = rows * r_full * rgs * config.residual_bit_width
                q_bytes += (r_bits + 7) // 8 + rows * r_full * 4 + rows * r_rem * 2

            quantized_size_bytes += q_bytes

    # Non-linear params (embeddings, layer norms, etc.)
    for name, param in model.named_parameters():
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        if parent_name not in linear_names:
            p_bytes = param.numel() * param.element_size()
            original_size_bytes += p_bytes
            quantized_size_bytes += p_bytes

    return {
        "original_mb": original_size_bytes / (1024 * 1024),
        "quantized_mb": quantized_size_bytes / (1024 * 1024),
        "compression_ratio": original_size_bytes / max(quantized_size_bytes, 1),
    }
