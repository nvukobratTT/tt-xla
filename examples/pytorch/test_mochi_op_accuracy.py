#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolate accuracy divergence to individual ops with SPMD TP.

Tests each building block: Linear (column/row parallel), matmul, 
elementwise, etc. with CPU reference comparison.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


def setup():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
    return device, mesh, num_devices


def compare(name, cpu_out, tt_out):
    cpu_f = cpu_out.float().flatten()
    tt_f = tt_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(cpu_f.unsqueeze(0), tt_f.unsqueeze(0)).item()
    mse = ((cpu_f - tt_f) ** 2).mean().item()
    max_err = (cpu_f - tt_f).abs().max().item()
    status = "✅" if cos > 0.99 else "⚠️" if cos > 0.95 else "❌"
    print(f"  {status} {name}: cosine={cos:.6f}, max_err={max_err:.4f}, "
          f"cpu=[{cpu_out.float().min():.3f},{cpu_out.float().max():.3f}], "
          f"tt=[{tt_out.float().min():.3f},{tt_out.float().max():.3f}]")
    return cos


def test_linear_column_parallel(device, mesh):
    """Column-parallel linear: weight sharded on dim 0 (output features)."""
    print("\n=== Linear Column-Parallel (like Q/K/V projection) ===")
    torch.manual_seed(42)
    
    linear = nn.Linear(3072, 3072, bias=False).to(torch.bfloat16).eval()
    x = torch.randn(1, 256, 3072, dtype=torch.bfloat16)
    
    # CPU reference
    with torch.no_grad():
        cpu_out = linear(x)
    
    # TT with TP
    linear_tt = nn.Linear(3072, 3072, bias=False).to(torch.bfloat16).eval()
    linear_tt.load_state_dict(linear.state_dict())
    linear_tt = linear_tt.to(device)
    xs.mark_sharding(linear_tt.weight, mesh, ("model", None))
    
    compiled = torch.compile(linear_tt, backend="tt")
    with torch.no_grad():
        tt_out = compiled(x.to(device)).to("cpu")
    
    return compare("Linear col-parallel [1,256,3072]→[1,256,3072]", cpu_out, tt_out)


def test_linear_row_parallel(device, mesh):
    """Row-parallel linear: weight sharded on dim 1 (input features)."""
    print("\n=== Linear Row-Parallel (like output projection) ===")
    torch.manual_seed(42)
    
    linear = nn.Linear(3072, 3072, bias=True).to(torch.bfloat16).eval()
    x = torch.randn(1, 256, 3072, dtype=torch.bfloat16)
    
    with torch.no_grad():
        cpu_out = linear(x)
    
    linear_tt = nn.Linear(3072, 3072, bias=True).to(torch.bfloat16).eval()
    linear_tt.load_state_dict(linear.state_dict())
    linear_tt = linear_tt.to(device)
    xs.mark_sharding(linear_tt.weight, mesh, (None, "model"))
    xs.mark_sharding(linear_tt.bias, mesh, (None,))
    
    compiled = torch.compile(linear_tt, backend="tt")
    with torch.no_grad():
        tt_out = compiled(x.to(device)).to("cpu")
    
    return compare("Linear row-parallel [1,256,3072]→[1,256,3072]", cpu_out, tt_out)


def test_linear_no_tp(device, mesh):
    """Linear without TP sharding — baseline accuracy."""
    print("\n=== Linear NO TP (baseline) ===")
    torch.manual_seed(42)
    
    linear = nn.Linear(3072, 3072, bias=False).to(torch.bfloat16).eval()
    x = torch.randn(1, 256, 3072, dtype=torch.bfloat16)
    
    with torch.no_grad():
        cpu_out = linear(x)
    
    linear_tt = nn.Linear(3072, 3072, bias=False).to(torch.bfloat16).eval()
    linear_tt.load_state_dict(linear.state_dict())
    linear_tt = linear_tt.to(device)
    # No sharding
    
    compiled = torch.compile(linear_tt, backend="tt")
    with torch.no_grad():
        tt_out = compiled(x.to(device)).to("cpu")
    
    return compare("Linear no-TP [1,256,3072]→[1,256,3072]", cpu_out, tt_out)


def test_matmul_tp(device, mesh):
    """Batched matmul with TP — like attention QK^T."""
    print("\n=== Batched Matmul (like attention QK^T) ===")
    torch.manual_seed(42)
    
    # [B, heads, seq, head_dim] @ [B, heads, head_dim, seq]
    q = torch.randn(1, 24, 128, 128, dtype=torch.bfloat16)
    k = torch.randn(1, 24, 128, 128, dtype=torch.bfloat16)
    
    cpu_out = torch.matmul(q, k.transpose(-2, -1))
    
    def matmul_fn(q, k):
        return torch.matmul(q, k.transpose(-2, -1))
    
    compiled = torch.compile(matmul_fn, backend="tt")
    with torch.no_grad():
        tt_out = compiled(q.to(device), k.to(device)).to("cpu")
    
    return compare("Matmul [1,24,128,128]@[1,24,128,128]^T", cpu_out, tt_out)


def test_sdpa(device, mesh):
    """Scaled dot-product attention — no TP."""
    print("\n=== SDPA (F.scaled_dot_product_attention) ===")
    torch.manual_seed(42)
    
    import torch.nn.functional as F
    q = torch.randn(1, 6, 128, 128, dtype=torch.bfloat16)
    k = torch.randn(1, 6, 128, 128, dtype=torch.bfloat16)
    v = torch.randn(1, 6, 128, 128, dtype=torch.bfloat16)
    
    with torch.no_grad():
        cpu_out = F.scaled_dot_product_attention(q, k, v)
    
    def sdpa_fn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)
    
    compiled = torch.compile(sdpa_fn, backend="tt")
    with torch.no_grad():
        tt_out = compiled(q.to(device), k.to(device), v.to(device)).to("cpu")
    
    return compare("SDPA [1,6,128,128]", cpu_out, tt_out)


def test_rmsnorm(device, mesh):
    """RMSNorm — common normalization in Mochi."""
    print("\n=== RMSNorm ===")
    torch.manual_seed(42)
    
    from diffusers.models.normalization import RMSNorm
    norm = RMSNorm(3072, eps=1e-5).to(torch.bfloat16).eval()
    x = torch.randn(1, 256, 3072, dtype=torch.bfloat16)
    
    with torch.no_grad():
        cpu_out = norm(x)
    
    norm_tt = RMSNorm(3072, eps=1e-5).to(torch.bfloat16).eval()
    norm_tt.load_state_dict(norm.state_dict())
    norm_tt = norm_tt.to(device)
    
    compiled = torch.compile(norm_tt, backend="tt")
    with torch.no_grad():
        tt_out = compiled(x.to(device)).to("cpu")
    
    return compare("RMSNorm [1,256,3072]", cpu_out, tt_out)


def test_silu_gate(device, mesh):
    """SiLU + gate (SwiGLU pattern from FFN)."""
    print("\n=== SiLU Gate (SwiGLU) ===")
    torch.manual_seed(42)
    
    x = torch.randn(1, 256, 8192, dtype=torch.bfloat16)
    
    def swiglu(x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    
    cpu_out = swiglu(x)
    
    compiled = torch.compile(swiglu, backend="tt")
    with torch.no_grad():
        tt_out = compiled(x.to(device)).to("cpu")
    
    return compare("SwiGLU [1,256,8192]→[1,256,4096]", cpu_out, tt_out)


def test_single_block(device, mesh):
    """Single Mochi transformer block with TP — the key test."""
    print("\n=== Single MochiTransformerBlock with TP ===")
    torch.manual_seed(42)
    
    import types
    from diffusers.models import MochiTransformer3DModel
    from mochi_tt_compat import MochiAttnProcessorTT
    
    model = MochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview", subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).eval()
    
    # Get first block
    block = model.transformer_blocks[0]
    block.attn1.processor = MochiAttnProcessorTT()
    
    # Create inputs that match what the block expects
    # After PatchEmbed: [B, seq, hidden] = [1, 128+256, 3072] 
    hidden = torch.randn(1, 384, 3072, dtype=torch.bfloat16)
    encoder_hidden = torch.randn(1, 256, 1536, dtype=torch.bfloat16)
    temb = torch.randn(1, 3072, dtype=torch.bfloat16)
    # RoPE: (cos, sin) each [1, seq, heads, head_dim/2]
    image_rotary_emb = (
        torch.randn(384, 24, 64, dtype=torch.float32),
        torch.randn(384, 24, 64, dtype=torch.float32),
    )
    
    encoder_attention_mask = torch.ones(1, 256, dtype=torch.bfloat16)
    
    # CPU reference
    with torch.no_grad():
        cpu_hidden, cpu_enc = block(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )
    
    # TT with TP
    block2 = model.transformer_blocks[0]  # same block (already patched)
    block2 = block2.to(device)
    
    # Shard
    shard_specs = {}
    shard_specs[block2.attn1.to_q.weight] = ("model", None)
    shard_specs[block2.attn1.to_k.weight] = ("model", None)
    shard_specs[block2.attn1.to_v.weight] = ("model", None)
    if block2.attn1.add_q_proj is not None:
        shard_specs[block2.attn1.add_q_proj.weight] = ("model", None)
    if block2.attn1.add_k_proj is not None:
        shard_specs[block2.attn1.add_k_proj.weight] = ("model", None)
    if block2.attn1.add_v_proj is not None:
        shard_specs[block2.attn1.add_v_proj.weight] = ("model", None)
    shard_specs[block2.attn1.to_out[0].weight] = (None, "model")
    if block2.attn1.to_out[0].bias is not None:
        shard_specs[block2.attn1.to_out[0].bias] = (None,)
    if block2.attn1.to_add_out is not None:
        shard_specs[block2.attn1.to_add_out.weight] = (None, "model")
        if block2.attn1.to_add_out.bias is not None:
            shard_specs[block2.attn1.to_add_out.bias] = (None,)
    shard_specs[block2.ff.net[0].proj.weight] = ("model", None)
    shard_specs[block2.ff.net[2].weight] = (None, "model")
    if block2.ff_context is not None:
        shard_specs[block2.ff_context.net[0].proj.weight] = ("model", None)
        shard_specs[block2.ff_context.net[2].weight] = (None, "model")
    
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    
    compiled = torch.compile(block2, backend="tt")
    with torch.no_grad():
        tt_hidden, tt_enc = compiled(
            hidden_states=hidden.to(device),
            encoder_hidden_states=encoder_hidden.to(device),
            temb=temb.to(device),
            image_rotary_emb=(image_rotary_emb[0].to(device), image_rotary_emb[1].to(device)),
            encoder_attention_mask=encoder_attention_mask.to(device),
        )
    tt_hidden = tt_hidden.to("cpu")
    tt_enc = tt_enc.to("cpu")
    
    cos_h = compare("Block hidden_states [1,384,3072]", cpu_hidden, tt_hidden)
    cos_e = compare("Block encoder_hidden [1,256,1536]", cpu_enc, tt_enc)
    return min(cos_h, cos_e)


def main():
    device, mesh, num_devices = setup()
    print(f"SPMD: {num_devices} devices\n")
    
    results = {}
    
    # Basic ops
    results["linear_no_tp"] = test_linear_no_tp(device, mesh)
    results["linear_col"] = test_linear_column_parallel(device, mesh)
    results["linear_row"] = test_linear_row_parallel(device, mesh)
    results["matmul"] = test_matmul_tp(device, mesh)
    results["sdpa"] = test_sdpa(device, mesh)
    results["rmsnorm"] = test_rmsnorm(device, mesh)
    results["swiglu"] = test_silu_gate(device, mesh)
    
    # Single block (the key test)
    results["block"] = test_single_block(device, mesh)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, cos in results.items():
        status = "✅" if cos > 0.99 else "⚠️" if cos > 0.95 else "❌"
        print(f"  {status} {name}: {cos:.6f}")


if __name__ == "__main__":
    main()
