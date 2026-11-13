# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RETFound-DINOv2 (ViT-L/14) as the RF-DETR encoder.

RETFound-DINOv2 shares the DINOv2 ViT-L/14 architecture, so the encoder is rfdetr's own windowed HuggingFace DINOv2
(``DinoV2(size="large")``) with the RETFound weights loaded into it. Windowed attention, gradient checkpointing, the
drop-path schedule, layer-wise lr decay and LoRA therefore work exactly as for the built-in encoders.

Workflow:

1. Convert the released RETFound checkpoint to the HuggingFace layout with ``my_utils/convert_retfound.py``.
2. Strip the DINOv2-small backbone out of ``rf-detr-small.pth`` with ``my_utils/strip_backbone_ckpt.py`` so the
   decoder and heads can still start from the COCO-pretrained weights.
3. Point ``RETFOUND_WEIGHTS`` at the converted file and train with ``RFDETRRETFound`` (Python) or
   ``configs/rfdetr_retfound_fundus.yaml`` (CLI). Importing this module registers the encoder.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import torch
from torch import nn

from rfdetr.config import PathLikeStr, RFDETRSmallConfig
from rfdetr.models.backbone import register_backbone
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.weights import interpolate_position_embeddings
from rfdetr.utilities.logger import get_logger
from rfdetr.variants import RFDETRSmall

logger = get_logger()

__all__ = ["ENCODER_NAME", "WEIGHTS_ENV_VAR", "RETFoundBackbone", "RFDETRRETFound", "RFDETRRETFoundConfig"]

#: ``ModelConfig.encoder`` value that selects :class:`RETFoundBackbone`.
ENCODER_NAME = "retfound_dinov2_large"
#: Environment variable holding the path of the converted (HuggingFace-layout) RETFound encoder weights.
WEIGHTS_ENV_VAR = "RETFOUND_WEIGHTS"


def load_retfound_encoder_weights(encoder: DinoV2, path: str | os.PathLike[str], positional_encoding_size: int) -> None:
    """Load converted RETFound weights into *encoder*, resampling the position embeddings to its grid.

    Args:
        encoder: The ``DinoV2(size="large")`` wrapper built by :class:`RETFoundBackbone`.
        path: File written by ``my_utils/convert_retfound.py``.
        positional_encoding_size: Side length, in patches, of the encoder's position-embedding grid.

    Raises:
        RuntimeError: If the file does not match the ViT-L/14 layout (``strict=True``), e.g. a checkpoint with
            register tokens or one that was not converted.

    Examples:
        >>> import tempfile
        >>> encoder = DinoV2(size="large", use_registers=False, load_dinov2_weights=False, patch_size=14,
        ...     num_windows=1, positional_encoding_size=2, shape=(28, 28), out_feature_indexes=[24])
        >>> state = {k: torch.zeros_like(v) for k, v in encoder.encoder.state_dict().items()}
        >>> state["embeddings.position_embeddings"] = torch.zeros(1, 1 + 4 * 4, 1024)  # a 4x4 source grid
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     torch.save(state, f"{tmp}/w.pth")
        ...     load_retfound_encoder_weights(encoder, f"{tmp}/w.pth", positional_encoding_size=2)
        >>> tuple(encoder.encoder.embeddings.position_embeddings.shape)
        (1, 5, 1024)
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    interpolate_position_embeddings(state, positional_encoding_size)
    encoder.encoder.load_state_dict(state, strict=True)


class RETFoundBackbone(Backbone):
    """RF-DETR backbone whose encoder is RETFound-DINOv2 ViT-L/14."""

    def _build_encoder(
        self,
        name: str,
        *,
        out_feature_indexes: list[int] | None,
        target_shape: tuple[int, int],
        gradient_checkpointing: bool,
        patch_size: int,
        num_windows: int,
        positional_encoding_size: int,
        drop_path: float,
        window_block_indexes: list[int] | None,
        **kwargs: Any,
    ) -> nn.Module:
        """Build a windowed DINOv2-large encoder and load the RETFound weights named by ``RETFOUND_WEIGHTS``.

        The upstream-pretrained flag is ignored: RF-DETR clears it whenever ``pretrain_weights`` is set, which is the
        case here (the stripped COCO checkpoint), yet the encoder still needs RETFound. Loading a fine-tuned RF-DETR
        checkpoint afterwards overwrites the encoder anyway, so ``RETFOUND_WEIGHTS`` may be left unset for inference.

        Args:
            name: Encoder identifier (``ModelConfig.encoder``).
            out_feature_indexes: Encoder layers whose outputs are returned.
            target_shape: Native ``(height, width)`` input resolution.
            gradient_checkpointing: Whether to checkpoint encoder activations.
            patch_size: Patch size of the encoder's patch embedding (RETFound-DINOv2 uses 14).
            num_windows: Windows per side for windowed attention.
            positional_encoding_size: Side length, in patches, of the position-embedding grid.
            drop_path: Stochastic-depth rate.
            window_block_indexes: Layers using windowed attention, or ``None`` to derive them.
            **kwargs: Further hook arguments, unused.

        Returns:
            The encoder module.
        """
        encoder = DinoV2(
            size="large",
            use_registers=False,
            use_windowed_attn=True,
            load_dinov2_weights=False,
            shape=target_shape,
            out_feature_indexes=out_feature_indexes,
            gradient_checkpointing=gradient_checkpointing,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            drop_path_rate=drop_path,
            window_block_indexes=window_block_indexes,
        )
        weights_path = os.environ.get(WEIGHTS_ENV_VAR)
        if weights_path:
            load_retfound_encoder_weights(encoder, weights_path, positional_encoding_size)
            logger.info("Loaded RETFound encoder weights from %s.", weights_path)
        else:
            logger.warning(
                "%s is not set: the RETFound encoder starts from random weights. That is only fine when a fine-tuned "
                "RF-DETR checkpoint is loaded next (inference / resume).",
                WEIGHTS_ENV_VAR,
            )
        return encoder


register_backbone(ENCODER_NAME, RETFoundBackbone)


class RFDETRRETFoundConfig(RFDETRSmallConfig):
    """RF-DETR Small decoder on a RETFound-DINOv2 ViT-L/14 encoder.

    Resolution must be a multiple of ``patch_size * num_windows`` = 28; changing ``resolution`` updates
    ``positional_encoding_size`` (= resolution / 14) automatically in Python, but must be set by hand on the CLI.
    ``projector_scale`` stays ``["P4"]`` so the decoder of the stripped ``rf-detr-small`` checkpoint still fits.
    """

    encoder: Literal["retfound_dinov2_large"] = ENCODER_NAME  # type: ignore[assignment]
    patch_size: int = 14
    num_windows: int = 2
    resolution: int = 504
    positional_encoding_size: int = 36
    out_feature_indexes: list[int] = [6, 12, 18, 24]
    pretrain_weights: PathLikeStr | None = None


class RFDETRRETFound(RFDETRSmall):
    """Python-API wrapper for :class:`RFDETRRETFoundConfig`.

    Examples:
        >>> RFDETRRETFound._model_config_class is RFDETRRETFoundConfig
        True
    """

    size = None
    _model_config_class = RFDETRRETFoundConfig
