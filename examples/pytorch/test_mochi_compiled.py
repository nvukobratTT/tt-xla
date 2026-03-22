#!/usr/bin/env python3
"""Test: Mochi transformer with torch.compile(backend='tt') for proper tilization."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

xr.set_device_type("TT")
torch_xla.set_custom_compile_options({"optimization_level": 1})
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
num_devices = xr.global_runtime_device_count()
mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
device = torch_xla.device()

from diffusers.models import MochiTransformer3DModel
from mochi_tt_compat import MochiAttnProcessorTT, apply_rotary_emb_tt
from mochi_t2v_tp import apply_tp_sharding_mochi_transformer

print("Loading transformer...", flush=True)
transformer = MochiTransformer3DModel.from_pretrained(
    "genmo/mochi-1-preview", subfolder="transformer", torch_dtype=torch.bfloat16).eval()

# Patch attention processors only (no forward wrapping)
count = 0
for block in transformer.transformer_blocks:
    if hasattr(block, 'attn1') and hasattr(block.attn1, 'processor'):
        block.attn1.processor = MochiAttnProcessorTT()
        count += 1
print(f"  Patched {count} attention processors", flush=True)

# Move to device
transformer = transformer.to(device)

# Apply TP sharding
apply_tp_sharding_mochi_transformer(transformer, mesh)

# Compile with TT backend
print("Compiling with torch.compile(backend='tt')...", flush=True)
compiled_transformer = torch.compile(transformer, backend="tt")

# Test inputs
hidden = torch.randn(1, 12, 7, 64, 64, dtype=torch.bfloat16, device=device)
encoder = torch.randn(1, 32, 4096, dtype=torch.bfloat16, device=device)
timestep = torch.tensor([500], dtype=torch.long, device=device)
mask = torch.ones(1, 32, dtype=torch.bfloat16, device=device)

print("Forward pass...", flush=True)
start = time.time()
with torch.no_grad():
    output = compiled_transformer(
        hidden_states=hidden, encoder_hidden_states=encoder,
        timestep=timestep, encoder_attention_mask=mask,
    )
    result = output.sample
    result_cpu = result.cpu()
    print(f"Total time: {time.time()-start:.1f}s", flush=True)
    print(f"Output: {result_cpu.shape}", flush=True)
    print(f"min={result_cpu.min():.4f}, max={result_cpu.max():.4f}", flush=True)
    print("SUCCESS", flush=True)
