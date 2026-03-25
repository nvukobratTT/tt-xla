#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated test: Mochi SDPA on TT device.

Reproduces the SDPA validation error:
  "k_chunk_size % 32 == 0"

Mochi uses scaled_dot_product_attention with:
  Q/K/V: [1, 24, seq_len, 128] bf16
  seq_len = num_video_tokens + num_text_tokens

At 480p/5s: seq_len = 133560 + 256 = 133816
"""

import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr


def test_sdpa_mochi_full():
    """SDPA at full 480p/5s resolution (133816 tokens)."""
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()

    batch = 1
    heads = 24
    seq_len = 133816  # 133560 video + 256 text
    head_dim = 128

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    print(f"SDPA: Q/K/V shape = [{batch}, {heads}, {seq_len}, {head_dim}]")
    print(f"  seq_len = {seq_len} (133560 video + 256 text)")

    def sdpa_fn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    compiled = torch.compile(sdpa_fn, backend="tt")

    print("\nRunning compiled SDPA...")
    with torch.no_grad():
        output = compiled(q, k, v)
        result = output.to("cpu")

    print(f"Output: {list(result.shape)}")
    print("PASS")


def test_sdpa_small():
    """Smaller SDPA to check basic functionality."""
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()

    batch, heads, seq_len, head_dim = 1, 24, 256, 128

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    print(f"Small SDPA: [{batch}, {heads}, {seq_len}, {head_dim}]")

    def sdpa_fn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    compiled = torch.compile(sdpa_fn, backend="tt")

    with torch.no_grad():
        output = compiled(q, k, v)
        result = output.to("cpu")

    print(f"Output: {list(result.shape)}")
    print("PASS")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "small":
        test_sdpa_small()
    else:
        test_sdpa_mochi_full()
