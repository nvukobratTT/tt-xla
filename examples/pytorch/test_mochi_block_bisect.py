#!/usr/bin/env python3
"""Bisect accuracy inside MochiTransformerBlock to find divergent subcomponent."""

import os
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
    return device, mesh


def compare(name, cpu_out, tt_out):
    cpu_f = cpu_out.float().flatten()
    tt_f = tt_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(cpu_f.unsqueeze(0), tt_f.unsqueeze(0)).item()
    max_err = (cpu_f - tt_f).abs().max().item()
    status = "✅" if cos > 0.99 else "⚠️" if cos > 0.95 else "❌"
    print(f"  {status} {name}: cosine={cos:.6f}, max_err={max_err:.4f}")
    return cos


def load_block():
    from diffusers.models import MochiTransformer3DModel
    from mochi_tt_compat import MochiAttnProcessorTT
    model = MochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch.bfloat16).eval()
    block = model.transformer_blocks[0]
    block.attn1.processor = MochiAttnProcessorTT()
    return block, model


def make_inputs(seq_v=128, seq_t=256):
    torch.manual_seed(42)
    hidden = torch.randn(1, seq_v, 3072, dtype=torch.bfloat16)
    encoder_hidden = torch.randn(1, seq_t, 1536, dtype=torch.bfloat16)
    temb = torch.randn(1, 3072, dtype=torch.bfloat16)
    rope = (
        torch.randn(seq_v, 24, 64, dtype=torch.float32),
        torch.randn(seq_v, 24, 64, dtype=torch.float32),
    )
    mask = torch.ones(1, seq_t, dtype=torch.bfloat16)
    return hidden, encoder_hidden, temb, rope, mask


def shard_attn(attn, mesh):
    specs = {}
    specs[attn.to_q.weight] = ("model", None)
    specs[attn.to_k.weight] = ("model", None)
    specs[attn.to_v.weight] = ("model", None)
    if attn.add_q_proj is not None:
        specs[attn.add_q_proj.weight] = ("model", None)
    if attn.add_k_proj is not None:
        specs[attn.add_k_proj.weight] = ("model", None)
    if attn.add_v_proj is not None:
        specs[attn.add_v_proj.weight] = ("model", None)
    specs[attn.to_out[0].weight] = (None, "model")
    if attn.to_out[0].bias is not None:
        specs[attn.to_out[0].bias] = (None,)
    if attn.to_add_out is not None:
        specs[attn.to_add_out.weight] = (None, "model")
        if attn.to_add_out.bias is not None:
            specs[attn.to_add_out.bias] = (None,)
    for t, s in specs.items():
        xs.mark_sharding(t, mesh, s)
    return len(specs)


def test_norm1(device, mesh):
    """MochiRMSNormZero — the initial normalization + scale/gate/shift."""
    print("\n=== MochiRMSNormZero (norm1) ===")
    block, _ = load_block()
    hidden, _, temb, _, _ = make_inputs()

    norm1 = block.norm1
    with torch.no_grad():
        cpu_out = norm1(hidden, temb)
    # Returns tuple: (normed_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    norm1_tt = block.norm1.to(device)
    compiled = torch.compile(norm1_tt, backend="tt")
    with torch.no_grad():
        tt_out = compiled(hidden.to(device), temb.to(device))
    
    results = []
    names = ["normed_hidden", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]
    for i, (c, t) in enumerate(zip(cpu_out, tt_out)):
        name = names[i] if i < len(names) else f"out_{i}"
        results.append(compare(f"norm1.{name}", c, t.to("cpu")))
    return min(results)


def test_attn_only(device, mesh):
    """MochiAttention with MochiAttnProcessorTT + TP — no norms."""
    print("\n=== MochiAttention + TP (attn1 only) ===")
    block, _ = load_block()
    hidden, encoder_hidden, _, rope, mask = make_inputs()

    attn = block.attn1
    # Attention needs pre-normed inputs. Just test with raw inputs.
    # The processor does: Q/K/V projection, RoPE, SDPA, output projection
    
    # We need to call the processor directly
    processor = attn.processor
    
    # Prepare Q/K/V on CPU
    with torch.no_grad():
        query = attn.to_q(hidden)
        key = attn.to_k(hidden)
        value = attn.to_v(hidden)
        
        # Context Q/K/V
        add_query = attn.add_q_proj(encoder_hidden)
        add_key = attn.add_k_proj(encoder_hidden)
        add_value = attn.add_v_proj(encoder_hidden)
        
        # Norm Q/K
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        add_query = attn.norm_added_q(add_query)
        add_key = attn.norm_added_k(add_key)
    
    print(f"  Q: {list(query.shape)}, K: {list(key.shape)}")
    print(f"  add_Q: {list(add_query.shape)}, add_K: {list(add_key.shape)}")
    
    # Now test just the SDPA + output projection part
    def attn_core(query, key, value, add_query, add_key, add_value, rope_cos, rope_sin):
        """Core attention: concat Q/K/V, apply RoPE, SDPA, split, output proj."""
        inner_dim = query.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape to heads
        query = query.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)

        add_query = add_query.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)
        add_key = add_key.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)
        add_value = add_value.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)

        # RoPE
        def apply_rope(x, cos, sin):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim/2]
            sin = sin.unsqueeze(0).unsqueeze(0)
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        query = apply_rope(query, rope_cos, rope_sin)
        key = apply_rope(key, rope_cos, rope_sin)

        # Concat video + text
        query = torch.cat([query, add_query], dim=2)
        key = torch.cat([key, add_key], dim=2)
        value = torch.cat([value, add_value], dim=2)

        # SDPA
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        # Split back
        seq_v = hidden.shape[1]
        out_v = out[:, :, :seq_v]
        out_t = out[:, :, seq_v:]

        # Reshape back
        out_v = out_v.transpose(1, 2).flatten(2)
        out_t = out_t.transpose(1, 2).flatten(2)

        # Output projections
        out_v = attn.to_out[0](out_v)
        out_t = attn.to_add_out(out_t)
        
        return out_v, out_t

    # Unflatten patch
    _orig = torch.Tensor.unflatten
    def _unflatten_via_view(self, dim, sizes):
        if dim < 0: dim = self.ndim + dim
        shape = list(self.shape)
        return self.view(shape[:dim] + list(sizes) + shape[dim+1:])
    torch.Tensor.unflatten = _unflatten_via_view

    with torch.no_grad():
        cpu_v, cpu_t = attn_core(query, key, value, add_query, add_key, add_value,
                                  rope[0], rope[1])

    # TT version — move everything to device, shard output projections
    attn_tt = block.attn1.to(device)
    shard_attn(attn_tt, mesh)

    def attn_core_tt(query, key, value, add_query, add_key, add_value, rope_cos, rope_sin):
        inner_dim = query.shape[-1]
        head_dim = inner_dim // attn_tt.heads

        query = query.unflatten(2, (attn_tt.heads, head_dim)).transpose(1, 2)
        key = key.unflatten(2, (attn_tt.heads, head_dim)).transpose(1, 2)
        value = value.unflatten(2, (attn_tt.heads, head_dim)).transpose(1, 2)
        add_query = add_query.unflatten(2, (attn_tt.heads, head_dim)).transpose(1, 2)
        add_key = add_key.unflatten(2, (attn_tt.heads, head_dim)).transpose(1, 2)
        add_value = add_value.unflatten(2, (attn_tt.heads, head_dim)).transpose(1, 2)

        def apply_rope(x, cos, sin):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        query = apply_rope(query, rope_cos, rope_sin)
        key = apply_rope(key, rope_cos, rope_sin)

        query = torch.cat([query, add_query], dim=2)
        key = torch.cat([key, add_key], dim=2)
        value = torch.cat([value, add_value], dim=2)

        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        seq_v = 128  # hardcoded for this test
        out_v = out[:, :, :seq_v]
        out_t = out[:, :, seq_v:]
        out_v = out_v.transpose(1, 2).flatten(2)
        out_t = out_t.transpose(1, 2).flatten(2)
        out_v = attn_tt.to_out[0](out_v)
        out_t = attn_tt.to_add_out(out_t)
        return out_v, out_t

    compiled = torch.compile(attn_core_tt, backend="tt")
    with torch.no_grad():
        tt_v, tt_t = compiled(
            query.to(device), key.to(device), value.to(device),
            add_query.to(device), add_key.to(device), add_value.to(device),
            rope[0].to(device), rope[1].to(device),
        )

    cos_v = compare("Attn video out [1,128,3072]", cpu_v, tt_v.to("cpu"))
    cos_t = compare("Attn text out [1,256,1536]", cpu_t, tt_t.to("cpu"))
    return min(cos_v, cos_t)


def test_modulated_rmsnorm(device, mesh):
    """MochiModulatedRMSNorm — RMSNorm with scale modulation."""
    print("\n=== MochiModulatedRMSNorm (norm2) ===")
    block, model = load_block()
    
    torch.manual_seed(42)
    hidden = torch.randn(1, 128, 3072, dtype=torch.bfloat16)
    # scale comes from pooling attention output, shape [1, 1, 3072]
    scale = torch.randn(1, 1, 3072, dtype=torch.bfloat16)

    norm2 = block.norm2
    with torch.no_grad():
        cpu_out = norm2(hidden, scale)

    norm2_tt = block.norm2.to(device)
    compiled = torch.compile(norm2_tt, backend="tt")
    with torch.no_grad():
        tt_out = compiled(hidden.to(device), scale.to(device))

    return compare("MochiModulatedRMSNorm [1,128,3072]", cpu_out, tt_out.to("cpu"))


def test_ff_with_tp(device, mesh):
    """FeedForward (SwiGLU) with TP sharding."""
    print("\n=== FeedForward with TP ===")
    block, _ = load_block()
    
    torch.manual_seed(42)
    hidden = torch.randn(1, 128, 3072, dtype=torch.bfloat16)

    ff = block.ff
    with torch.no_grad():
        cpu_out = ff(hidden)

    ff_tt = block.ff.to(device)
    xs.mark_sharding(ff_tt.net[0].proj.weight, mesh, ("model", None))
    xs.mark_sharding(ff_tt.net[2].weight, mesh, (None, "model"))
    
    compiled = torch.compile(ff_tt, backend="tt")
    with torch.no_grad():
        tt_out = compiled(hidden.to(device)).to("cpu")

    return compare("FF [1,128,3072]→[1,128,3072]", cpu_out, tt_out)


def main():
    device, mesh = setup()
    print(f"Bisecting MochiTransformerBlock accuracy...\n")

    results = {}
    results["norm1"] = test_norm1(device, mesh)
    results["modulated_rmsnorm"] = test_modulated_rmsnorm(device, mesh)
    results["ff_tp"] = test_ff_with_tp(device, mesh)
    results["attn_core_tp"] = test_attn_only(device, mesh)

    print(f"\n{'='*60}")
    print("BISECT SUMMARY")
    print(f"{'='*60}")
    for name, cos in results.items():
        status = "✅" if cos > 0.99 else "⚠️" if cos > 0.95 else "❌"
        print(f"  {status} {name}: {cos:.6f}")


if __name__ == "__main__":
    main()
