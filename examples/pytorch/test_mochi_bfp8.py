#!/usr/bin/env python3
"""Test: Mochi transformer with BFP8 weights to fit larger dims in DRAM."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

xr.set_device_type("TT")

# BFP8 weights + opt_level 0
torch_xla.set_custom_compile_options({
    "optimization_level": 0,
    "experimental_weight_dtype": "bfp8",
})
print("Config: optimization_level=0, experimental_weight_dtype=bfp8", flush=True)

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
num_devices = xr.global_runtime_device_count()
mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
device = torch_xla.device()

from diffusers.models import MochiTransformer3DModel
from mochi_tt_compat import MochiAttnProcessorTT
from mochi_t2v_tp import apply_tp_sharding_mochi_transformer

print("Loading transformer...", flush=True)
transformer = MochiTransformer3DModel.from_pretrained(
    "genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch.bfloat16).eval()
for block in transformer.transformer_blocks:
    block.attn1.processor = MochiAttnProcessorTT()
transformer = transformer.to(device)
apply_tp_sharding_mochi_transformer(transformer, mesh)

compiled = torch.compile(transformer, backend="tt")

# Try 64x64 — previously OOMed at bf16
h, w, f, t = int(os.environ.get("H", "64")), int(os.environ.get("W", "64")), int(os.environ.get("F", "7")), int(os.environ.get("T", "32"))
hidden = torch.randn(1, 12, f, h, w, dtype=torch.bfloat16, device=device)
encoder = torch.randn(1, t, 4096, dtype=torch.bfloat16, device=device)
timestep = torch.tensor([500], dtype=torch.long, device=device)
mask = torch.ones(1, t, dtype=torch.bfloat16, device=device)

print(f"Input: hidden=[1,12,{f},{h},{w}], encoder=[1,{t},4096]", flush=True)
print("Forward pass (BFP8 weights)...", flush=True)

start = time.time()
with torch.no_grad():
    output = compiled(
        hidden_states=hidden, encoder_hidden_states=encoder,
        timestep=timestep, encoder_attention_mask=mask,
    )
    result = output.sample
    result_cpu = result.cpu()
    elapsed = time.time() - start

print(f"Output: {result_cpu.shape}", flush=True)
print(f"Time: {elapsed:.1f}s", flush=True)
print(f"min={result_cpu.min():.4f}, max={result_cpu.max():.4f}", flush=True)
print("SUCCESS", flush=True)
