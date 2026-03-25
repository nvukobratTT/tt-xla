#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Accuracy comparison: Mochi DiT — CPU reference vs TT compiled.

Runs the same forward pass on CPU and TT, compares outputs.
Uses small resolution to keep CPU reference fast.
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


def setup_tt():
    """Initialize TT device with SPMD."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()

    mesh = Mesh(
        np.array(range(num_devices)),
        (1, num_devices),
        ("batch", "model"),
    )
    return device, mesh, num_devices


def patch_model(model):
    """Apply all patches for torch.compile compatibility."""
    # Patch 1: unflatten
    def _unflatten_via_view(self, dim, sizes):
        if dim < 0:
            dim = self.ndim + dim
        shape = list(self.shape)
        return self.view(shape[:dim] + list(sizes) + shape[dim + 1:])
    torch.Tensor.unflatten = _unflatten_via_view

    # Patch 2: attention processors
    from mochi_tt_compat import MochiAttnProcessorTT
    for block in model.transformer_blocks:
        if hasattr(block, "attn1") and hasattr(block.attn1, "processor"):
            block.attn1.processor = MochiAttnProcessorTT()

    # Patch 3: rope autocast
    def _create_rope_no_autocast(self, freqs, pos):
        freqs = torch.einsum("nd,dhf->nhf", pos.to(torch.float32), freqs.to(torch.float32))
        return torch.cos(freqs), torch.sin(freqs)
    model._create_rope = types.MethodType(_create_rope_no_autocast, model)


def apply_tp_sharding(model, mesh):
    """Megatron-style TP sharding."""
    shard_specs = {}
    for block in model.transformer_blocks:
        shard_specs[block.attn1.to_q.weight] = ("model", None)
        shard_specs[block.attn1.to_k.weight] = ("model", None)
        shard_specs[block.attn1.to_v.weight] = ("model", None)
        if hasattr(block.attn1, 'add_q_proj') and block.attn1.add_q_proj is not None:
            shard_specs[block.attn1.add_q_proj.weight] = ("model", None)
        if hasattr(block.attn1, 'add_k_proj') and block.attn1.add_k_proj is not None:
            shard_specs[block.attn1.add_k_proj.weight] = ("model", None)
        if hasattr(block.attn1, 'add_v_proj') and block.attn1.add_v_proj is not None:
            shard_specs[block.attn1.add_v_proj.weight] = ("model", None)
        shard_specs[block.attn1.to_out[0].weight] = (None, "model")
        if block.attn1.to_out[0].bias is not None:
            shard_specs[block.attn1.to_out[0].bias] = (None,)
        if hasattr(block.attn1, 'to_add_out') and block.attn1.to_add_out is not None:
            shard_specs[block.attn1.to_add_out.weight] = (None, "model")
            if block.attn1.to_add_out.bias is not None:
                shard_specs[block.attn1.to_add_out.bias] = (None,)
        shard_specs[block.ff.net[0].proj.weight] = ("model", None)
        if hasattr(block.ff.net[0].proj, 'bias') and block.ff.net[0].proj.bias is not None:
            shard_specs[block.ff.net[0].proj.bias] = ("model",)
        shard_specs[block.ff.net[2].weight] = (None, "model")
        if hasattr(block.ff.net[2], 'bias') and block.ff.net[2].bias is not None:
            shard_specs[block.ff.net[2].bias] = (None,)
        if block.ff_context is not None:
            shard_specs[block.ff_context.net[0].proj.weight] = ("model", None)
            if hasattr(block.ff_context.net[0].proj, 'bias') and block.ff_context.net[0].proj.bias is not None:
                shard_specs[block.ff_context.net[0].proj.bias] = ("model",)
            shard_specs[block.ff_context.net[2].weight] = (None, "model")
            if hasattr(block.ff_context.net[2], 'bias') and block.ff_context.net[2].bias is not None:
                shard_specs[block.ff_context.net[2].bias] = (None,)

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    return len(shard_specs)


def compare_tensors(name, cpu_out, tt_out):
    """Compare two tensors and print metrics."""
    cpu_f = cpu_out.float()
    tt_f = tt_out.float()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_f.flatten().unsqueeze(0),
        tt_f.flatten().unsqueeze(0),
    ).item()

    # PSNR
    mse = ((cpu_f - tt_f) ** 2).mean().item()
    if mse > 0:
        max_val = max(cpu_f.abs().max().item(), tt_f.abs().max().item())
        psnr = 10 * np.log10(max_val ** 2 / mse)
    else:
        psnr = float('inf')

    # Max absolute error
    max_err = (cpu_f - tt_f).abs().max().item()

    # Relative error
    rel_err = ((cpu_f - tt_f).abs() / (cpu_f.abs() + 1e-8)).mean().item()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape:          {list(cpu_out.shape)}")
    print(f"  CPU range:      [{cpu_f.min():.4f}, {cpu_f.max():.4f}]")
    print(f"  TT range:       [{tt_f.min():.4f}, {tt_f.max():.4f}]")
    print(f"  Cosine sim:     {cos_sim:.6f}")
    print(f"  PSNR:           {psnr:.2f} dB")
    print(f"  Max abs error:  {max_err:.6f}")
    print(f"  Mean rel error: {rel_err:.6f}")

    return cos_sim, psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    h, w, nf = args.height, args.width, args.num_frames
    assert (nf - 1) % 6 == 0
    lf = (nf - 1) // 6 + 1

    print(f"Config: {h}x{w}, {nf} frames ({lf} latent frames)")

    # --- Generate fixed inputs ---
    torch.manual_seed(args.seed)
    hidden_states = torch.randn(1, 12, lf, h // 8, w // 8, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 256, 4096, dtype=torch.bfloat16)
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_attention_mask = torch.ones(1, 256, dtype=torch.bfloat16)

    print(f"  Latents: {list(hidden_states.shape)}")
    print(f"  Text:    {list(encoder_hidden_states.shape)}")

    # --- CPU Reference ---
    print("\n--- CPU Reference ---")
    from diffusers.models import MochiTransformer3DModel

    model_cpu = MochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview", subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).eval()

    # Apply same patches to CPU model for fair comparison
    patch_model(model_cpu)

    print("Running CPU forward...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        cpu_output = model_cpu(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
        )
    cpu_result = cpu_output.sample
    print(f"CPU done in {time.time() - t0:.1f}s")
    print(f"CPU output: {list(cpu_result.shape)}, range: [{cpu_result.min():.4f}, {cpu_result.max():.4f}]")

    # Free CPU model memory
    del model_cpu
    import gc; gc.collect()

    # --- TT Compiled ---
    print("\n--- TT Compiled ---")
    device, mesh, num_devices = setup_tt()
    print(f"SPMD: {num_devices} devices")

    model_tt = MochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview", subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).eval()

    patch_model(model_tt)
    model_tt = model_tt.to(device)
    n_sharded = apply_tp_sharding(model_tt, mesh)
    print(f"TP sharding: {n_sharded} tensors")

    compiled = torch.compile(model_tt, backend="tt", fullgraph=True)

    print("Running TT forward...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        tt_output = compiled(
            hidden_states=hidden_states.to(device),
            encoder_hidden_states=encoder_hidden_states.to(device),
            timestep=timestep.to(device),
            encoder_attention_mask=encoder_attention_mask.to(device),
        )
    if hasattr(tt_output, "sample"):
        tt_result = tt_output.sample.to("cpu")
    else:
        tt_result = tt_output.to("cpu")
    print(f"TT done in {time.time() - t0:.1f}s")
    print(f"TT output: {list(tt_result.shape)}, range: [{tt_result.min():.4f}, {tt_result.max():.4f}]")

    # --- Compare ---
    cos, psnr = compare_tensors("DiT Output (CPU vs TT)", cpu_result, tt_result)

    print(f"\n{'='*60}")
    if cos > 0.99:
        print(f"  ✅ GOOD: cosine={cos:.6f}, PSNR={psnr:.1f} dB")
    elif cos > 0.95:
        print(f"  ⚠️  MARGINAL: cosine={cos:.6f}, PSNR={psnr:.1f} dB")
    else:
        print(f"  ❌ BAD: cosine={cos:.6f}, PSNR={psnr:.1f} dB")
    print(f"{'='*60}")

    # Save for offline analysis
    torch.save({
        "cpu": cpu_result,
        "tt": tt_result,
        "config": {"h": h, "w": w, "nf": nf, "seed": args.seed},
    }, "mochi_accuracy_comparison.pt")
    print("Saved: mochi_accuracy_comparison.pt")


if __name__ == "__main__":
    main()
