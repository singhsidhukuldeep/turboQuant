"""Command-line interface for TurboQuant."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("turboquant")


def cmd_quantize(args):
    """Quantize a HuggingFace model."""
    import torch

    from .config import TurboQuantConfig
    from .model import _get_model_class, quantize_model, save_quantized

    logger.info(f"Loading model: {args.model}")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error(
            "transformers is required. Install with: pip install turbo-quant[transformers]"
        )
        sys.exit(1)

    ModelClass = _get_model_class(args.model, task_hint=args.model_class)
    dtype = getattr(torch, args.dtype)
    model = ModelClass.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device if args.device != "cpu" else None,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    config = TurboQuantConfig(
        bit_width=args.bits,
        group_size=args.group_size,
        residual_bit_width=args.residual_bits,
        rotation_method=args.rotation,
        rotation_seed=args.seed,
    )

    logger.info(
        f"Quantizing with config: {config.total_bits}-bit "
        f"(primary={config.bit_width}, residual={config.residual_bit_width})"
    )
    model = quantize_model(model, config)

    logger.info(f"Saving to {args.output}")
    save_quantized(
        model, config, args.output, save_tokenizer=True, tokenizer=tokenizer
    )
    logger.info("Done!")


def cmd_estimate(args):
    """Estimate model size before and after quantization."""
    import torch

    from .config import TurboQuantConfig
    from .model import _get_model_class, estimate_model_size

    try:
        import transformers  # noqa: F401
    except ImportError:
        logger.error(
            "transformers is required. Install with: pip install turbo-quant[transformers]"
        )
        sys.exit(1)

    logger.info(f"Loading model: {args.model}")

    ModelClass = _get_model_class(args.model, task_hint=args.model_class)
    dtype = getattr(torch, args.dtype)
    model = ModelClass.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    )

    config = TurboQuantConfig(
        bit_width=args.bits,
        group_size=args.group_size,
        residual_bit_width=args.residual_bits,
    )

    sizes = estimate_model_size(model, config)
    print(f"\nModel: {args.model}")
    print(
        f"Quantization: {config.total_bits}-bit TurboQuant "
        f"(group_size={config.group_size})"
    )
    print(f"  Original size:     {sizes['original_mb']:.1f} MB")
    print(f"  Quantized size:    {sizes['quantized_mb']:.1f} MB")
    print(f"  Compression ratio: {sizes['compression_ratio']:.2f}x")


def cmd_generate(args):
    """Generate text with a quantized model."""
    import torch

    from .model import load_quantized

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error(
            "transformers is required. Install with: pip install turbo-quant[transformers]"
        )
        sys.exit(1)

    quantized_path = Path(args.quantized)
    logger.info(f"Loading quantized model from {quantized_path}")
    model = load_quantized(
        args.model,
        quantized_path,
        device_map=args.device if args.device != "cpu" else None,
        trust_remote_code=args.trust_remote_code,
        model_class=args.model_class,
    )
    model.eval()

    tokenizer_path = (
        quantized_path
        if (quantized_path / "tokenizer_config.json").exists()
        else Path(args.model)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=args.trust_remote_code
    )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


def cmd_info(args):
    """Show information about a quantized model."""
    from .config import TurboQuantConfig

    path = Path(args.path)

    config = TurboQuantConfig.load(path / "turboquant_config.json")
    print(f"\nTurboQuant Model: {path}")
    print(f"  Bit width:      {config.bit_width}")
    print(f"  Residual bits:  {config.residual_bit_width}")
    print(f"  Total bits:     {config.total_bits}")
    print(f"  Group size:     {config.group_size}")
    print(f"  Rotation:       {config.rotation_method} (seed={config.rotation_seed})")
    print(f"  Skip modules:   {config.modules_to_not_convert}")

    weights_path = path / "quantized_weights.pt"
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"  Weights file:   {size_mb:.1f} MB")


def main():
    # Configure logging only when running as CLI, not at import time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="turboquant",
        description="TurboQuant: Near-optimal weight quantization for LLMs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- quantize ---
    p_quant = subparsers.add_parser("quantize", help="Quantize a HuggingFace model")
    p_quant.add_argument(
        "--model", "-m", required=True, help="HuggingFace model name or path"
    )
    p_quant.add_argument("--output", "-o", required=True, help="Output directory")
    p_quant.add_argument(
        "--bits", "-b", type=int, default=4, choices=[2, 3, 4],
        help="Bit width (default: 4)",
    )
    p_quant.add_argument(
        "--residual-bits", type=int, default=0, choices=[0, 2, 3, 4],
        help="Residual bit width (default: 0)",
    )
    p_quant.add_argument(
        "--group-size", "-g", type=int, default=128,
        help="Group size (default: 128)",
    )
    p_quant.add_argument(
        "--rotation", type=str, default="hadamard", choices=["hadamard", "qr"],
    )
    p_quant.add_argument("--seed", type=int, default=42)
    p_quant.add_argument(
        "--dtype", type=str, default="float16", help="Model loading dtype"
    )
    p_quant.add_argument(
        "--device", type=str, default="cpu", help="Device for quantization"
    )
    p_quant.add_argument("--trust-remote-code", action="store_true")
    p_quant.add_argument(
        "--model-class", type=str, default=None,
        choices=["causal-lm", "seq2seq", "classification", "auto"],
        help="Model class hint (default: causal-lm)",
    )
    p_quant.set_defaults(func=cmd_quantize)

    # --- estimate ---
    p_est = subparsers.add_parser(
        "estimate", help="Estimate model size after quantization"
    )
    p_est.add_argument(
        "--model", "-m", required=True, help="HuggingFace model name or path"
    )
    p_est.add_argument("--bits", "-b", type=int, default=4, choices=[2, 3, 4])
    p_est.add_argument(
        "--residual-bits", type=int, default=0, choices=[0, 2, 3, 4]
    )
    p_est.add_argument("--group-size", "-g", type=int, default=128)
    p_est.add_argument("--dtype", type=str, default="float16")
    p_est.add_argument("--trust-remote-code", action="store_true")
    p_est.add_argument(
        "--model-class", type=str, default=None,
        choices=["causal-lm", "seq2seq", "classification", "auto"],
        help="Model class hint (default: causal-lm)",
    )
    p_est.set_defaults(func=cmd_estimate)

    # --- generate ---
    p_gen = subparsers.add_parser(
        "generate", help="Generate text with a quantized model"
    )
    p_gen.add_argument(
        "--model", "-m", required=True, help="Base model name for architecture"
    )
    p_gen.add_argument(
        "--quantized", "-q", required=True, help="Path to quantized model"
    )
    p_gen.add_argument("--prompt", "-p", required=True, help="Input prompt")
    p_gen.add_argument("--max-tokens", type=int, default=100)
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_gen.add_argument("--device", type=str, default="cpu")
    p_gen.add_argument("--trust-remote-code", action="store_true")
    p_gen.add_argument(
        "--model-class", type=str, default=None,
        choices=["causal-lm", "seq2seq", "classification", "auto"],
        help="Model class hint (default: causal-lm)",
    )
    p_gen.set_defaults(func=cmd_generate)

    # --- info ---
    p_info = subparsers.add_parser("info", help="Show info about a quantized model")
    p_info.add_argument("path", help="Path to quantized model directory")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
