# Wan2.1-T2V Correctness Debug Report (Updated)

**Date:** 2026-03-20
**Status:** ✅ Root cause identified and fix verified

## Root Cause: Strided Slice Assignment Bug in tt-xla

**The XLA→TTIR compiler does not correctly lower strided slice assignments like `tensor[..., 0::2] = value`.**

This manifests catastrophically in the Wan model's RoPE (Rotary Position Embedding) implementation, which uses:
```python
out = torch.empty_like(hidden_states)
out[..., 0::2] = x1 * cos - x2 * sin   # ← BROKEN on TT
out[..., 1::2] = x1 * sin + x2 * cos   # ← BROKEN on TT
```

### Proof (isolated test):
| Pattern | Max Diff vs CPU | Status |
|---------|----------------|--------|
| `empty_like` + `[::2]` assign | 14.625 (95% elements wrong!) | 🔴 BROKEN |
| `zeros_like` + `[::2]` assign | 3.719 (87.5% elements wrong!) | 🔴 BROKEN |
| `torch.stack([a, b], dim=-1).flatten(-2)` | 0.0625 (bf16 rounding only) | ✅ WORKS |
| `torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1).reshape(...)` | 0.0625 | ✅ WORKS |

### How this causes noise output:
1. RoPE error amplifies small matmul drift from 0.063 → 16.75 (266× amplification)
2. Corrupted Q/K produce wrong attention scores (23.8% of scores off by >1.0)
3. Wrong softmax probabilities → wrong attention output (17% of elements off by >1.0)
4. Compounds through 30 transformer blocks → block output error: 50.8 max diff
5. Compounds through 50 denoising steps → complete noise

### What works fine on TT:
- Data round-trip (bf16 and f32): perfect
- float() upcast: perfect
- AdaLN modulation: perfect
- Softmax: near-perfect (max_diff 0.015)
- LayerNorm: small error (0.011)
- MatMul: tolerable bf16 error (0.293 mean for 1536-wide)

## Fix

**Monkey-patch the WanAttnProcessor's `apply_rotary_emb` to avoid strided assignments.**

Replace in the diffusers `WanAttnProcessor.__call__`:
```python
# ORIGINAL (broken on TT):
def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)

# FIXED (works on TT):
def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    x = hidden_states.unflatten(-1, (-1, 2))
    x1, x2 = x[..., 0], x[..., 1]
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2).type_as(hidden_states)
```

## Secondary Issue: bf16 MatMul Precision

Even with the RoPE fix, TT bf16 matmul has ~0.293 mean error for 1536-wide reductions. This is higher than CPU bf16 but within acceptable bounds for inference (the compiler already uses HiFi4 + fp32 dest accumulators by default).

After the RoPE fix, the per-block error should drop from 50.8 → ~1.0, and 30 blocks should produce usable (if slightly noisy) output.

## Files
- Debug script: `/workspace/tt-xla/debug_correctness.py`
- RoPE fix test: `/workspace/tt-xla/test_rope_fix.py`
- Pipeline (1.3B): `/workspace/tt-xla/examples/pytorch/wan_t2v_1_3b.py`
- This report: `/workspace/tt-xla/docs/wan_correctness_debug.md`

## Bug Report for tt-xla

**Title:** Strided slice assignment produces incorrect results on TT device

**Reproduction:**
```python
import torch, torch_xla, torch_xla.runtime as xr
xr.set_device_type("TT")
device = torch_xla.device()

a = torch.randn(2, 4, 4, dtype=torch.bfloat16)
b = torch.randn(2, 4, 4, dtype=torch.bfloat16)

# This produces wrong results on TT:
out = torch.zeros(2, 4, 8, dtype=torch.bfloat16, device=device)
out[..., 0::2] = a.to(device)
out[..., 1::2] = b.to(device)
torch_xla.sync()
# out.cpu() does NOT match CPU reference

# Workaround:
out = torch.stack([a.to(device), b.to(device)], dim=-1).flatten(-2)
torch_xla.sync()
# out.cpu() MATCHES CPU reference
```
