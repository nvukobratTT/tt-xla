# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Debug script: run Mochi components one at a time on TT to find compilation issues.
"""

import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


def setup_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, (1, num_devices), ("batch", "model"))
    print(f"SPMD: {num_devices} devices, mesh: {mesh}")
    return mesh, num_devices


def test_single_block():
    """Test a single MochiTransformerBlock on TT."""
    from diffusers.models.transformers.transformer_mochi import MochiTransformerBlock
    from mochi_tt_compat import MochiAttnProcessorTT

    mesh, num_devices = setup_spmd()
    device = torch_xla.device()

    print("\n=== Test: Single MochiTransformerBlock ===")

    # Create a block matching Mochi config
    block = MochiTransformerBlock(
        dim=3072,
        num_attention_heads=24,
        attention_head_dim=128,
        pooled_projection_dim=1536,
        qk_norm="rms_norm",
        activation_fn="swiglu",
        context_pre_only=False,
    )
    block.eval()

    # Patch attention
    block.attn1.processor = MochiAttnProcessorTT()

    block = block.to(device=device, dtype=torch.bfloat16)

    # Create test inputs
    batch, seq_len, text_seq = 1, 64, 32  # small for testing
    hidden = torch.randn(batch, seq_len, 3072, dtype=torch.bfloat16, device=device)
    encoder_hidden = torch.randn(batch, text_seq, 1536, dtype=torch.bfloat16, device=device)
    temb = torch.randn(batch, 3072, dtype=torch.bfloat16, device=device)
    attn_mask = torch.ones(batch, text_seq, dtype=torch.bfloat16, device=device)

    # RoPE: (seq_len, heads, dim_head//2)
    rope_cos = torch.randn(seq_len, 24, 64, dtype=torch.bfloat16, device=device)
    rope_sin = torch.randn(seq_len, 24, 64, dtype=torch.bfloat16, device=device)

    print(f"  hidden: {hidden.shape}, encoder: {encoder_hidden.shape}")
    print(f"  temb: {temb.shape}, mask: {attn_mask.shape}")

    start = time.time()
    with torch.no_grad():
        out_hidden, out_encoder = block(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            temb=temb,
            encoder_attention_mask=attn_mask,
            image_rotary_emb=(rope_cos, rope_sin),
        )
    xm.mark_step()
    elapsed = time.time() - start
    print(f"  Output: hidden={out_hidden.shape}, encoder={out_encoder.shape}")
    print(f"  Time: {elapsed:.1f}s")
    print("  PASSED ✓")


def test_transformer_forward():
    """Test full MochiTransformer3DModel forward on TT."""
    from diffusers.models import MochiTransformer3DModel
    from mochi_tt_compat import patch_mochi_for_tt

    mesh, num_devices = setup_spmd()
    device = torch_xla.device()

    print("\n=== Test: Full MochiTransformer3DModel ===")

    model_id = "genmo/mochi-1-preview"
    print(f"  Loading from {model_id}...")
    transformer = MochiTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()

    # Patch attention processors
    patch_mochi_for_tt(transformer)

    # Move to device
    transformer = transformer.to(device)

    # Apply TP sharding
    from mochi_t2v_tp import apply_tp_sharding_mochi_transformer
    apply_tp_sharding_mochi_transformer(transformer, mesh)

    # Create small test inputs
    # Mochi expects: (batch, channels=12, frames, height, width)
    batch = 1
    channels = 12
    frames = 7
    height = 64  # must be divisible by patch_size=2
    width = 64
    seq_len = 32

    hidden = torch.randn(batch, channels, frames, height, width, dtype=torch.bfloat16, device=device)
    encoder_hidden = torch.randn(batch, seq_len, 4096, dtype=torch.bfloat16, device=device)
    timestep = torch.tensor([500], dtype=torch.long, device=device)
    attn_mask = torch.ones(batch, seq_len, dtype=torch.bfloat16, device=device)

    print(f"  Input: {hidden.shape}")
    print(f"  Encoder: {encoder_hidden.shape}")
    print(f"  Compiling + running...")

    start = time.time()
    with torch.no_grad():
        output = transformer(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            timestep=timestep,
            encoder_attention_mask=attn_mask,
        )
    xm.mark_step()
    elapsed = time.time() - start

    if hasattr(output, "sample"):
        out_tensor = output.sample
    else:
        out_tensor = output[0]

    print(f"  Output: {out_tensor.shape}")
    print(f"  Time: {elapsed:.1f}s")
    print("  PASSED ✓")


if __name__ == "__main__":
    import sys

    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    test = sys.argv[1] if len(sys.argv) > 1 else "block"

    if test == "block":
        test_single_block()
    elif test == "transformer":
        test_transformer_forward()
    else:
        print(f"Unknown test: {test}. Use 'block' or 'transformer'")
