# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Convert a RETFound-DINOv2 checkpoint (Meta DINOv2 naming) to rfdetr's HuggingFace DINOv2 layout.

Usage:
    python my_utils/convert_retfound.py RETFound_dinov2_meh.pth retfound_dinov2_hf.pth

The output holds only the encoder tensors, keyed as ``DinoV2(size="large").encoder`` expects, with the position
embeddings kept at the checkpoint's own grid; ``my_backbones.retfound`` resamples them to the training resolution
when it loads the file. The conversion is checked by a strict load into a freshly built ViT-L/14.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rfdetr.models.backbone.dinov2 import DinoV2

#: Wrapper keys a DINOv2-style training checkpoint may nest its weights under, most preferred first. The teacher is
#: the EMA model DINOv2 evaluates and releases.
_WRAPPER_KEYS = ("teacher", "model", "state_dict", "student")
#: Prefixes added by DDP / the DINOv2 meta-architecture around the ViT itself.
_STRIPPED_PREFIXES = ("module.", "backbone.")
#: Self-supervised projection heads that are not part of the encoder.
_HEAD_PREFIXES = ("dino_head.", "ibot_head.", "head.")

_TOP_LEVEL = {
    "cls_token": "embeddings.cls_token",
    "mask_token": "embeddings.mask_token",
    "pos_embed": "embeddings.position_embeddings",
    "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
    "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
    "norm.weight": "layernorm.weight",
    "norm.bias": "layernorm.bias",
}
_BLOCK = {
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "attn.proj.weight": "attention.output.dense.weight",
    "attn.proj.bias": "attention.output.dense.bias",
    "ls1.gamma": "layer_scale1.lambda1",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "mlp.fc1.weight": "mlp.fc1.weight",
    "mlp.fc1.bias": "mlp.fc1.bias",
    "mlp.fc2.weight": "mlp.fc2.weight",
    "mlp.fc2.bias": "mlp.fc2.bias",
    "ls2.gamma": "layer_scale2.lambda1",
}
# ``blocks.<i>.<param>`` or, for FSDP block chunks, ``blocks.<chunk>.<i>.<param>``; in both the last index is the
# global block index (Meta's BlockChunk pads earlier positions with nn.Identity).
_BLOCK_KEY = re.compile(r"^blocks\.(?:\d+\.)?(\d+)\.(.+)$")


def unwrap_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Tensor]:
    """Return the ViT tensors of *checkpoint*, without wrappers, DDP/meta-arch prefixes or SSL heads.

    Args:
        checkpoint: A loaded checkpoint, either a bare state dict or one nested under ``teacher``/``model``/....

    Returns:
        Meta-DINOv2-named encoder tensors.

    Examples:
        >>> ckpt = {"teacher": {"backbone.cls_token": torch.zeros(1), "dino_head.w": torch.zeros(1)}}
        >>> list(unwrap_checkpoint(ckpt))
        ['cls_token']
    """
    state: dict[str, Any] = checkpoint
    for key in _WRAPPER_KEYS:
        if isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break
    unwrapped: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(value, Tensor):
            continue
        changed = True
        while changed:
            changed = False
            for prefix in _STRIPPED_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
        if name.startswith(_HEAD_PREFIXES):
            continue
        unwrapped[name] = value
    return unwrapped


def convert_meta_dinov2_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename Meta-DINOv2 ViT tensors to the HuggingFace DINOv2 layout, splitting each fused ``qkv``.

    Args:
        state: Encoder tensors as returned by :func:`unwrap_checkpoint`.

    Returns:
        Tensors keyed for ``WindowedDinov2WithRegistersBackbone`` (``DinoV2(...).encoder``).

    Raises:
        ValueError: If *state* has register tokens (needs a registers encoder) or a key this mapping does not know.

    Examples:
        >>> qkv = torch.arange(6.0).reshape(6, 1)
        >>> out = convert_meta_dinov2_state_dict({"blocks.0.attn.qkv.weight": qkv, "norm.bias": torch.zeros(2)})
        >>> sorted(out)  # doctest: +NORMALIZE_WHITESPACE
        ['encoder.layer.0.attention.attention.key.weight', 'encoder.layer.0.attention.attention.query.weight',
         'encoder.layer.0.attention.attention.value.weight', 'layernorm.bias']
        >>> out["encoder.layer.0.attention.attention.key.weight"].flatten().tolist()
        [2.0, 3.0]
    """
    if "register_tokens" in state:
        raise ValueError(
            "Checkpoint has register tokens; build the encoder with use_registers=True and map "
            "'register_tokens' to 'embeddings.register_tokens' before using it."
        )
    converted: dict[str, Tensor] = {}
    unknown: list[str] = []
    for name, value in state.items():
        if name in _TOP_LEVEL:
            converted[_TOP_LEVEL[name]] = value
            continue
        match = _BLOCK_KEY.match(name)
        if match is None:
            unknown.append(name)
            continue
        index, param = match.groups()
        prefix = f"encoder.layer.{index}."
        if param in ("attn.qkv.weight", "attn.qkv.bias"):
            kind = param.rsplit(".", 1)[1]
            query, key, value_ = value.chunk(3, dim=0)
            converted[f"{prefix}attention.attention.query.{kind}"] = query
            converted[f"{prefix}attention.attention.key.{kind}"] = key
            converted[f"{prefix}attention.attention.value.{kind}"] = value_
        elif param in _BLOCK:
            converted[prefix + _BLOCK[param]] = value
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"Unrecognized checkpoint keys (first 10): {unknown[:10]}")
    return converted


def verify_against_vit_large(state: dict[str, Tensor]) -> None:
    """Strictly load *state* into a ViT-L/14 built at the checkpoint's own position-embedding grid.

    Args:
        state: Output of :func:`convert_meta_dinov2_state_dict`.

    Raises:
        RuntimeError: On any missing, unexpected or wrongly shaped tensor.

    Examples:
        >>> template = DinoV2(size="large", use_registers=False, load_dinov2_weights=False, patch_size=14,
        ...     num_windows=1, positional_encoding_size=1, shape=(14, 14), out_feature_indexes=[24])
        >>> state = template.encoder.state_dict()
        >>> verify_against_vit_large(state)
        >>> verify_against_vit_large({**state, "encoder.layer.24.norm1.weight": state["layernorm.weight"]})
        Traceback (most recent call last):
        ...
        RuntimeError: Error(s) in loading state_dict ...
    """
    grid = math.isqrt(state["embeddings.position_embeddings"].shape[1] - 1)
    patch_size = state["embeddings.patch_embeddings.projection.weight"].shape[-1]
    encoder = DinoV2(
        size="large",
        use_registers=False,
        load_dinov2_weights=False,
        patch_size=patch_size,
        num_windows=1,
        positional_encoding_size=grid,
        shape=(grid * patch_size, grid * patch_size),
        out_feature_indexes=[24],
    )
    encoder.encoder.load_state_dict(state, strict=True)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="RETFound-DINOv2 checkpoint (.pth)")
    parser.add_argument("output", type=Path, help="Converted encoder weights (.pth)")
    parser.add_argument(
        "--trust",
        action="store_true",
        help="Allow full pickle loading if the checkpoint holds non-tensor objects (only for trusted files).",
    )
    args = parser.parse_args()

    try:
        checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    except Exception as err:
        if not args.trust:
            raise SystemExit(f"Safe load failed ({err}). Re-run with --trust if the file is from a trusted source.")
        checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)

    state = unwrap_checkpoint(checkpoint)
    print("source tensors:", len(state))
    for name in ("pos_embed", "patch_embed.proj.weight", "cls_token"):
        if name in state:
            print(f"  {name}: {tuple(state[name].shape)}")
    converted = convert_meta_dinov2_state_dict(state)
    verify_against_vit_large(converted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print(f"strict load OK; wrote {len(converted)} tensors to {args.output}")


if __name__ == "__main__":
    main()
