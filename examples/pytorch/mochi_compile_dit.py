#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi DiT compile example — torch.compile(backend="tt", fullgraph=True) with
4-chip tensor parallelism via XLA SPMD.

Loads the 10B MochiTransformer3DModel, applies patches for single-graph
compilation, shards weights across devices, and runs at 480p/5s resolution.

Usage:
    # Compile-only (no device execution)
    python3 mochi_compile_dit.py --compile-only

    # Full forward (compile + execute)
    python3 mochi_compile_dit.py

    # Custom resolution
    python3 mochi_compile_dit.py --height 480 --width 848 --num-frames 121
"""

import argparse
import os
import time
import types

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


# =============================================================================
# PATCHES — required for single-graph compilation (fullgraph=True)
#
# Without these, torch.compile produces multiple graphs or fails entirely.
# Each patch addresses a specific dynamo graph break.
# =============================================================================


def patch_unflatten_for_dynamo():
    """
    PATCH 1: Fix Tensor.unflatten() graph break.

    Problem: torch._tensor.Tensor.unflatten() calls super().unflatten(),
    which dynamo can't trace through TT's TorchFunctionOverride.

    Fix: Replace with .view()-based equivalent that dynamo handles natively.
    """
    def _unflatten_via_view(self, dim, sizes):
        if dim < 0:
            dim = self.ndim + dim
        shape = list(self.shape)
        new_shape = shape[:dim] + list(sizes) + shape[dim + 1:]
        return self.view(new_shape)

    torch.Tensor.unflatten = _unflatten_via_view
    print("  [PATCH 1] Tensor.unflatten → view-based (super() graph break)")


def patch_attention_processors(model):
    """
    PATCH 2: Replace MochiAttnProcessor2_0 with static attention.

    Problem: MochiAttnProcessor2_0 uses torch.nonzero() for dynamic masking,
    which is a dynamic-shape op that breaks fullgraph compilation.

    Fix: Use MochiAttnProcessorTT which does static joint attention
    (concatenates video + text tokens, no dynamic indexing).
    """
    from mochi_tt_compat import MochiAttnProcessorTT

    count = 0
    for block in model.transformer_blocks:
        if hasattr(block, "attn1") and hasattr(block.attn1, "processor"):
            block.attn1.processor = MochiAttnProcessorTT()
            count += 1
    print(f"  [PATCH 2] {count} attention processors → MochiAttnProcessorTT (torch.nonzero)")
    return count


def patch_rope_autocast(model):
    """
    PATCH 3: Remove torch.autocast context manager from _create_rope.

    Problem: The original _create_rope wraps einsum in torch.autocast(),
    which is a context manager that causes a dynamo graph break.

    Fix: Inline the computation in float32 without the context manager.
    """
    def _create_rope_no_autocast(self, freqs, pos):
        freqs = torch.einsum("nd,dhf->nhf", pos.to(torch.float32), freqs.to(torch.float32))
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return freqs_cos, freqs_sin

    model._create_rope = types.MethodType(_create_rope_no_autocast, model)
    print("  [PATCH 3] _create_rope → removed autocast context manager")


# =============================================================================
# TENSOR PARALLELISM — Megatron-style sharding for Mochi DiT
#
# Column-parallel: Q/K/V projections, FFN up/gate (split output dim across devices)
# Row-parallel: output projections, FFN down (split input dim, all-reduce after)
# =============================================================================


def apply_tp_sharding(model, mesh):
    """
    Apply Megatron-style tensor parallel sharding to MochiTransformer3DModel.

    Shards all 48 transformer blocks using the same strategy as mochi_t2v_tp.py:
    - Attention Q/K/V: column-parallel (split heads across devices)
    - Attention output: row-parallel (partial results, all-reduce)
    - FFN up/gate: column-parallel
    - FFN down: row-parallel

    Both video and context streams are sharded identically.
    """
    shard_specs = {}

    for i, block in enumerate(model.transformer_blocks):
        # --- Joint Attention (attn1) ---
        # Video stream Q/K/V — column-parallel
        shard_specs[block.attn1.to_q.weight] = ("model", None)
        shard_specs[block.attn1.to_k.weight] = ("model", None)
        shard_specs[block.attn1.to_v.weight] = ("model", None)

        # Context stream Q/K/V — column-parallel
        if hasattr(block.attn1, 'add_q_proj') and block.attn1.add_q_proj is not None:
            shard_specs[block.attn1.add_q_proj.weight] = ("model", None)
        if hasattr(block.attn1, 'add_k_proj') and block.attn1.add_k_proj is not None:
            shard_specs[block.attn1.add_k_proj.weight] = ("model", None)
        if hasattr(block.attn1, 'add_v_proj') and block.attn1.add_v_proj is not None:
            shard_specs[block.attn1.add_v_proj.weight] = ("model", None)

        # Output projection — row-parallel
        shard_specs[block.attn1.to_out[0].weight] = (None, "model")
        if block.attn1.to_out[0].bias is not None:
            shard_specs[block.attn1.to_out[0].bias] = (None,)

        # Context output projection (all blocks except last)
        if hasattr(block.attn1, 'to_add_out') and block.attn1.to_add_out is not None:
            shard_specs[block.attn1.to_add_out.weight] = (None, "model")
            if block.attn1.to_add_out.bias is not None:
                shard_specs[block.attn1.to_add_out.bias] = (None,)

        # --- Video FFN (SwiGLU) ---
        shard_specs[block.ff.net[0].proj.weight] = ("model", None)  # up — column-parallel
        if hasattr(block.ff.net[0].proj, 'bias') and block.ff.net[0].proj.bias is not None:
            shard_specs[block.ff.net[0].proj.bias] = ("model",)
        shard_specs[block.ff.net[2].weight] = (None, "model")  # down — row-parallel
        if hasattr(block.ff.net[2], 'bias') and block.ff.net[2].bias is not None:
            shard_specs[block.ff.net[2].bias] = (None,)

        # --- Context FFN (all blocks except last) ---
        if block.ff_context is not None:
            shard_specs[block.ff_context.net[0].proj.weight] = ("model", None)
            if hasattr(block.ff_context.net[0].proj, 'bias') and block.ff_context.net[0].proj.bias is not None:
                shard_specs[block.ff_context.net[0].proj.bias] = ("model",)
            shard_specs[block.ff_context.net[2].weight] = (None, "model")
            if hasattr(block.ff_context.net[2], 'bias') and block.ff_context.net[2].bias is not None:
                shard_specs[block.ff_context.net[2].bias] = (None,)

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    print(f"  Applied TP sharding: {len(model.transformer_blocks)} blocks, {len(shard_specs)} tensors")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Mochi DiT compile with TP")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num-frames", type=int, default=121,
                        help="(n-1) must be divisible by 6. 121 = 5s at 24fps")
    parser.add_argument("--optimization-level", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile, skip forward execution")
    args = parser.parse_args()

    assert (args.num_frames - 1) % 6 == 0, f"(num_frames-1) must be divisible by 6"

    # --- SPMD + Device setup ---
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    print(f"SPMD enabled: {num_devices} devices")

    torch_xla.set_custom_compile_options({
        "optimization_level": args.optimization_level,
    })
    device = torch_xla.device()

    # Create mesh for tensor parallelism
    mesh = Mesh(
        np.array(range(num_devices)),
        (1, num_devices),
        ("batch", "model"),
    )
    print(f"Mesh: {mesh.mesh_shape} ({num_devices}-way TP)")

    # --- Apply patches for single-graph compilation ---
    print("\nApplying patches for fullgraph compilation:")
    patch_unflatten_for_dynamo()

    # --- Load model ---
    from diffusers.models import MochiTransformer3DModel

    print("\nLoading MochiTransformer3DModel (~10B params)...", flush=True)
    t0 = time.time()
    model = MochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview", subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).eval()

    patch_attention_processors(model)
    patch_rope_autocast(model)

    model = model.to(device)

    # --- Apply tensor parallel sharding ---
    print("\nApplying tensor parallel sharding:")
    apply_tp_sharding(model, mesh)

    print(f"Loaded + sharded + moved to XLA in {time.time() - t0:.1f}s", flush=True)

    # --- Compile (single graph) ---
    print(f"\ntorch.compile(backend='tt', fullgraph=True)...", flush=True)
    compiled = torch.compile(model, backend="tt", fullgraph=True)

    # --- Prepare inputs ---
    h, w, nf = args.height, args.width, args.num_frames
    latent_frames = (nf - 1) // 6 + 1
    text_len = 256

    print(f"\nConfig: {h}x{w}, {nf} frames ({(nf-1)/24:.1f}s at 24fps)")
    print(f"  Latent: [1, 12, {latent_frames}, {h//8}, {w//8}]")
    print(f"  Text:   [1, {text_len}, 4096]")
    print(f"  TP:     {num_devices}-way, opt_level={args.optimization_level}")

    hidden_states = torch.randn(
        1, 12, latent_frames, h // 8, w // 8,
        dtype=torch.bfloat16, device=device)
    encoder_hidden_states = torch.randn(
        1, text_len, 4096, dtype=torch.bfloat16, device=device)
    timestep = torch.tensor([500], dtype=torch.long, device=device)
    encoder_attention_mask = torch.ones(
        1, text_len, dtype=torch.bfloat16, device=device)

    # --- Forward ---
    if args.compile_only:
        print("\nTracing + compiling (no execution)...", flush=True)
        t0 = time.time()

        with torch.no_grad():
            output = compiled(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
            )

        torch_xla.sync()
        elapsed = time.time() - t0
        print(f"\nCompile done in {elapsed:.1f}s")
    else:
        print("\nForward pass (compile + execute)...", flush=True)
        t0 = time.time()

        with torch.no_grad():
            output = compiled(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
            )
            result = output.sample.to("cpu")

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s")
        print(f"Output: {result.shape}")
        print(f"Range:  [{result.min():.4f}, {result.max():.4f}]")


if __name__ == "__main__":
    main()
