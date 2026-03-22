#!/usr/bin/env python3
"""Test: transformer forward with CPU patch_embed workaround + CPU transfer."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_xla
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

xr.set_device_type("TT")
torch_xla.set_custom_compile_options({"optimization_level": 1})
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
num_devices = xr.global_runtime_device_count()
mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
device = torch_xla.device()

from diffusers.models import MochiTransformer3DModel
from mochi_tt_compat import patch_mochi_for_tt
from mochi_t2v_tp import apply_tp_sharding_mochi_transformer

print("Loading transformer...", flush=True)
transformer = MochiTransformer3DModel.from_pretrained(
    "genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch.bfloat16).eval()

# Move to device FIRST, then patch (so patch_embed starts on device)
transformer = transformer.to(device)
patch_mochi_for_tt(transformer)
apply_tp_sharding_mochi_transformer(transformer, mesh)

hidden = torch.randn(1, 12, 7, 64, 64, dtype=torch.bfloat16, device=device)
encoder = torch.randn(1, 32, 4096, dtype=torch.bfloat16, device=device)
timestep = torch.tensor([500], dtype=torch.long, device=device)
mask = torch.ones(1, 32, dtype=torch.bfloat16, device=device)

print("Forward pass (patch_embed on CPU, blocks on TT)...", flush=True)
start = time.time()
with torch.no_grad():
    output = transformer(
        hidden_states=hidden, encoder_hidden_states=encoder,
        timestep=timestep, encoder_attention_mask=mask,
    )
    result = output.sample
    torch_xla.sync()
    fwd_time = time.time() - start
    print(f"Forward + sync: {fwd_time:.1f}s", flush=True)

    result_cpu = result.cpu()
    torch_xla.sync()
    print(f"CPU transfer: {result_cpu.shape}", flush=True)
    print(f"min={result_cpu.min():.4f}, max={result_cpu.max():.4f}", flush=True)
    print("SUCCESS", flush=True)
