#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated test: Mochi PatchEmbed Conv2D on TT device.

Reproduces the L1 buffer clash error from mochi_compile_dit.py.
The PatchEmbed uses a 2D Conv with kernel_size=2, stride=2 to convert
video latents [B, 12, T, H, W] -> [B, T*H/2*W/2, 3072].

The Conv2D is applied per-frame: input is reshaped to [B*T, 12, H, W].

Error: "Statically allocated circular buffers in program clash with L1 buffers"
"""

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr


def test_conv2d_mochi_patchembed():
    """Exact Conv2D from Mochi PatchEmbed at 480p/5s resolution."""
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()

    # Mochi PatchEmbed Conv2D params
    conv = nn.Conv2d(
        in_channels=12,
        out_channels=3072,
        kernel_size=2,
        stride=2,
        padding=0,
        bias=True,
    ).to(torch.bfloat16).eval().to(device)

    # Input: [B*T, C, H, W] = [1*21, 12, 60, 106]
    # (T=21 latent frames at 480p/5s, H=60=480/8, W=106=848/8)
    x = torch.randn(21, 12, 60, 106, dtype=torch.bfloat16, device=device)

    print(f"Conv2d: in={conv.in_channels}, out={conv.out_channels}, "
          f"kernel={conv.kernel_size}, stride={conv.stride}")
    print(f"Input: {list(x.shape)}")
    print(f"Expected output: [21, 3072, 30, 53]")

    compiled = torch.compile(conv, backend="tt")

    print("\nRunning compiled Conv2D...")
    with torch.no_grad():
        output = compiled(x)
        result = output.to("cpu")

    print(f"Output: {list(result.shape)}")
    print(f"Range: [{result.min():.4f}, {result.max():.4f}]")
    print("PASS")


def test_conv2d_small():
    """Smaller Conv2D to check if the issue is size-dependent."""
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()

    conv = nn.Conv2d(12, 3072, kernel_size=2, stride=2, bias=True
    ).to(torch.bfloat16).eval().to(device)

    # Small input: [1, 12, 8, 8]
    x = torch.randn(1, 12, 8, 8, dtype=torch.bfloat16, device=device)

    print(f"Small Conv2d: input {list(x.shape)}")
    compiled = torch.compile(conv, backend="tt")

    with torch.no_grad():
        output = compiled(x)
        result = output.to("cpu")

    print(f"Output: {list(result.shape)}")
    print("PASS")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "small":
        test_conv2d_small()
    else:
        test_conv2d_mochi_patchembed()
