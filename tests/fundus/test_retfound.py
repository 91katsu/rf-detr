# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the RETFound-DINOv2 encoder plug-in and its checkpoint tools."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from my_backbones.retfound import ENCODER_NAME, WEIGHTS_ENV_VAR, RETFoundBackbone, RFDETRRETFoundConfig
from my_utils.convert_retfound import convert_meta_dinov2_state_dict, unwrap_checkpoint
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.utilities.tensors import NestedTensor

_DIM = 1024
_DEPTH = 24
_HEADS = 16
_PATCH = 14


def random_meta_vit_large_state(grid: int, seed: int = 0) -> dict[str, Tensor]:
    """Random ViT-L/14 weights in Meta DINOv2 naming, with a ``grid`` x ``grid`` position-embedding table.

    Args:
        grid: Side length, in patches, of the position-embedding grid.
        seed: Seed of the random generator.

    Returns:
        Meta-named tensors, as a released DINOv2 checkpoint holds them.

    Examples:
        >>> state = random_meta_vit_large_state(grid=2)
        >>> tuple(state["pos_embed"].shape), tuple(state["blocks.23.attn.qkv.weight"].shape)
        ((1, 5, 1024), (3072, 1024))
    """
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int, scale: float = 0.02) -> Tensor:
        return torch.randn(*shape, generator=generator) * scale

    state = {
        "cls_token": rand(1, 1, _DIM),
        "mask_token": rand(1, _DIM),
        "pos_embed": rand(1, 1 + grid * grid, _DIM),
        "patch_embed.proj.weight": rand(_DIM, 3, _PATCH, _PATCH),
        "patch_embed.proj.bias": rand(_DIM),
        "norm.weight": 1 + rand(_DIM),
        "norm.bias": rand(_DIM),
    }
    for i in range(_DEPTH):
        prefix = f"blocks.{i}."
        state |= {
            prefix + "norm1.weight": 1 + rand(_DIM),
            prefix + "norm1.bias": rand(_DIM),
            prefix + "attn.qkv.weight": rand(3 * _DIM, _DIM),
            prefix + "attn.qkv.bias": rand(3 * _DIM),
            prefix + "attn.proj.weight": rand(_DIM, _DIM),
            prefix + "attn.proj.bias": rand(_DIM),
            prefix + "ls1.gamma": rand(_DIM, scale=0.5),
            prefix + "norm2.weight": 1 + rand(_DIM),
            prefix + "norm2.bias": rand(_DIM),
            prefix + "mlp.fc1.weight": rand(4 * _DIM, _DIM),
            prefix + "mlp.fc1.bias": rand(4 * _DIM),
            prefix + "mlp.fc2.weight": rand(_DIM, 4 * _DIM),
            prefix + "mlp.fc2.bias": rand(_DIM),
            prefix + "ls2.gamma": rand(_DIM, scale=0.5),
        }
    return state


def reference_meta_vit_forward(state: dict[str, Tensor], image: Tensor) -> Tensor:
    """Final normed patch tokens of a Meta-layout DINOv2 ViT, computed directly from its tensors.

    Written against the Meta DINOv2 ``Block`` (pre-norm attention and MLP, each scaled by LayerScale), independent
    of the HuggingFace implementation the converted weights are loaded into.

    Args:
        state: Meta-named tensors, e.g. from :func:`random_meta_vit_large_state`.
        image: ``(B, 3, H, W)`` input whose patch grid equals the position-embedding grid.

    Returns:
        ``(B, C, H / 14, W / 14)`` patch features after the final norm.

    Examples:
        >>> state = random_meta_vit_large_state(grid=1)
        >>> tuple(reference_meta_vit_forward(state, torch.zeros(1, 3, 14, 14)).shape)
        (1, 1024, 1, 1)
    """
    batch, _, height, width = image.shape
    x = F.conv2d(image, state["patch_embed.proj.weight"], state["patch_embed.proj.bias"], stride=_PATCH)
    x = x.flatten(2).transpose(1, 2)
    x = torch.cat([state["cls_token"].expand(batch, -1, -1), x], dim=1) + state["pos_embed"]
    head_dim = _DIM // _HEADS
    for i in range(_DEPTH):
        p = f"blocks.{i}."
        h = F.layer_norm(x, (_DIM,), state[p + "norm1.weight"], state[p + "norm1.bias"], eps=1e-6)
        qkv = F.linear(h, state[p + "attn.qkv.weight"], state[p + "attn.qkv.bias"])
        q, k, v = qkv.reshape(batch, -1, 3, _HEADS, head_dim).permute(2, 0, 3, 1, 4)
        attn = torch.softmax(q @ k.transpose(-2, -1) / head_dim**0.5, dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(batch, -1, _DIM)
        h = F.linear(h, state[p + "attn.proj.weight"], state[p + "attn.proj.bias"])
        x = x + state[p + "ls1.gamma"] * h
        h = F.layer_norm(x, (_DIM,), state[p + "norm2.weight"], state[p + "norm2.bias"], eps=1e-6)
        h = F.linear(
            F.gelu(F.linear(h, state[p + "mlp.fc1.weight"], state[p + "mlp.fc1.bias"])),
            state[p + "mlp.fc2.weight"],
            state[p + "mlp.fc2.bias"],
        )
        x = x + state[p + "ls2.gamma"] * h
    x = F.layer_norm(x, (_DIM,), state["norm.weight"], state["norm.bias"], eps=1e-6)
    return x[:, 1:].transpose(1, 2).reshape(batch, _DIM, height // _PATCH, width // _PATCH)


def build_retfound_backbone(grid: int) -> RETFoundBackbone:
    """Build a :class:`RETFoundBackbone` at ``grid * 28`` pixels with the RFDETRRETFoundConfig defaults.

    Args:
        grid: Side length of the input, in units of ``patch_size * num_windows`` = 28 pixels.

    Returns:
        The backbone, in eval mode.

    Examples:
        >>> build_retfound_backbone(grid=1).encoder._out_feature_channels
        [1024, 1024, 1024, 1024]
    """
    config = RFDETRRETFoundConfig(num_classes=3)
    patches = grid * config.num_windows
    backbone = RETFoundBackbone(
        config.encoder,
        out_feature_indexes=config.out_feature_indexes,
        projector_scale=config.projector_scale,
        layer_norm=config.layer_norm,
        out_channels=config.hidden_dim,
        target_shape=(patches * config.patch_size, patches * config.patch_size),
        patch_size=config.patch_size,
        num_windows=config.num_windows,
        positional_encoding_size=patches,
    )
    return backbone.eval()


class TestConvertRetfound:
    def test_converted_weights_match_meta_reference(self) -> None:
        state = random_meta_vit_large_state(grid=2)
        encoder = DinoV2(
            size="large",
            use_registers=False,
            load_dinov2_weights=False,
            patch_size=_PATCH,
            num_windows=1,
            positional_encoding_size=2,
            shape=(28, 28),
            out_feature_indexes=[24],
        ).eval()
        encoder.encoder.load_state_dict(convert_meta_dinov2_state_dict(state), strict=True)
        image = torch.randn(1, 3, 28, 28, generator=torch.Generator().manual_seed(1))

        with torch.no_grad():
            converted = encoder(image)[0]
            expected = reference_meta_vit_forward(state, image)

        torch.testing.assert_close(converted, expected, atol=1e-4, rtol=1e-4)

    def test_unwrap_strips_teacher_backbone_prefix_and_heads(self) -> None:
        checkpoint = {
            "teacher": {"backbone.blocks.0.ls1.gamma": torch.ones(2), "dino_head.mlp.0.weight": torch.ones(1)}
        }

        assert list(unwrap_checkpoint(checkpoint)) == ["blocks.0.ls1.gamma"]

    def test_chunked_block_keys_use_global_index(self) -> None:
        converted = convert_meta_dinov2_state_dict({"blocks.1.7.ls2.gamma": torch.ones(2)})

        assert list(converted) == ["encoder.layer.7.layer_scale2.lambda1"]

    def test_register_tokens_are_refused(self) -> None:
        with pytest.raises(ValueError, match="register tokens"):
            convert_meta_dinov2_state_dict({"register_tokens": torch.zeros(1, 4, _DIM)})

    def test_unknown_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unrecognized"):
            convert_meta_dinov2_state_dict({"blocks.0.attn.rel_pos": torch.zeros(1)})


class TestRetfoundBackbone:
    def test_projector_outputs_hidden_dim_at_patch_grid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(WEIGHTS_ENV_VAR, raising=False)
        backbone = build_retfound_backbone(grid=1)
        image = torch.randn(1, 3, 28, 28)
        tensors = NestedTensor(image, torch.zeros(1, 28, 28, dtype=torch.bool))

        with torch.no_grad():
            features, _ = backbone(tensors)

        assert [tuple(f.tensors.shape) for f in features] == [(1, 256, 2, 2)]

    def test_weights_from_env_var_are_loaded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        state = convert_meta_dinov2_state_dict(random_meta_vit_large_state(grid=4))
        weights = tmp_path / "retfound_hf.pth"
        torch.save(state, weights)
        monkeypatch.setenv(WEIGHTS_ENV_VAR, str(weights))

        backbone = build_retfound_backbone(grid=1)

        loaded = backbone.encoder.encoder.state_dict()["encoder.layer.23.mlp.fc2.weight"]
        torch.testing.assert_close(loaded, state["encoder.layer.23.mlp.fc2.weight"])

    def test_layer_decay_gives_first_layer_a_lower_lr_than_last(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(WEIGHTS_ENV_VAR, raising=False)
        backbone = build_retfound_backbone(grid=1)
        args = SimpleNamespace(
            out_feature_indexes=RFDETRRETFoundConfig(num_classes=3).out_feature_indexes,
            lr_encoder=1e-4,
            lr_vit_layer_decay=0.9,
            lr_component_decay=1.0,
            weight_decay=1e-4,
        )

        groups = backbone.get_named_param_lr_pairs(args)

        first = groups["backbone.0.encoder.encoder.encoder.layer.0.mlp.fc1.weight"]["lr"]
        last = groups["backbone.0.encoder.encoder.encoder.layer.23.mlp.fc1.weight"]["lr"]
        assert first == pytest.approx(last * 0.9**23)


def test_config_selects_registered_encoder() -> None:
    config = RFDETRRETFoundConfig(num_classes=3, resolution=560)

    assert (config.encoder, config.positional_encoding_size) == (ENCODER_NAME, 40)
