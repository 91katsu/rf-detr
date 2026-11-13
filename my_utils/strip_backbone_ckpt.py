# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Drop the backbone (encoder + projector) from an RF-DETR checkpoint, keeping the decoder and heads.

Usage:
    python my_utils/strip_backbone_ckpt.py ~/.roboflow/models/rf-detr-small.pth rf-detr-small-nobackbone.pth

Use the output as ``pretrain_weights`` of a model with another encoder: its projector input width differs (384 for
DINOv2-small, 1024 for ViT-L), so the original backbone tensors would fail ``load_state_dict`` on shape. The recorded
``args.patch_size`` is dropped too, because rfdetr refuses a checkpoint whose patch size differs from the model's
even when no backbone tensor is left to disagree.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from rfdetr.utilities.io import _safe_torch_load

_BACKBONE_PREFIX = "backbone."
#: ``args`` entries that describe the removed backbone and would be checked against the new one.
_BACKBONE_ARGS = ("patch_size",)


def strip_backbone(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Return a copy of *checkpoint* without backbone tensors or backbone-describing ``args``.

    ``args`` is written as a plain ``dict`` so the result loads with ``torch.load(weights_only=True)``.

    Args:
        checkpoint: A loaded RF-DETR checkpoint with a ``model`` state dict.

    Returns:
        The stripped checkpoint and the number of tensors removed.

    Examples:
        >>> ckpt = {"model": {"backbone.0.x": torch.zeros(1), "class_embed.bias": torch.zeros(3)},
        ...         "args": Namespace(patch_size=16, num_queries=300)}
        >>> stripped, removed = strip_backbone(ckpt)
        >>> list(stripped["model"]), removed, stripped["args"]
        (['class_embed.bias'], 1, {'num_queries': 300})
    """
    model = {k: v for k, v in checkpoint["model"].items() if not k.startswith(_BACKBONE_PREFIX)}
    stripped = {**checkpoint, "model": model}
    args = checkpoint.get("args")
    if args is not None:
        args_dict = dict(vars(args)) if isinstance(args, Namespace) else dict(args)
        for name in _BACKBONE_ARGS:
            args_dict.pop(name, None)
        stripped["args"] = args_dict
    return stripped, len(checkpoint["model"]) - len(model)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="RF-DETR checkpoint, e.g. rf-detr-small.pth")
    parser.add_argument("output", type=Path, help="Stripped checkpoint to write")
    args = parser.parse_args()

    checkpoint = _safe_torch_load(args.source)
    stripped, removed = strip_backbone(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, args.output)
    print(f"removed {removed} backbone tensors, kept {len(stripped['model'])}; wrote {args.output}")


if __name__ == "__main__":
    main()
