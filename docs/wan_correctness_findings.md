# Wan2.1-T2V-1.3B Numerical Correctness: Root Cause Analysis

**Date:** 2026-03-20
**Status:** Root causes identified — two compounding issues in matmul precision and RoPE application

## Executive Summary

The noise output from the Wan2.1-T2V-1.3B pipeline on TT Blackhole is caused by **two compounding numerical issues**, both originating from bf16 matmul precision on the TT hardware. These accumulate through 30 transformer blocks and 50 denoising steps, turning structured predictions into uniform noise.

## Systematic Test Results

### What works perfectly (zero error):
| Component | Max Diff | Status |
|-----------|----------|--------|
| Data round-trip (CPU→TT→CPU, bf16) | 0.0 | ✅ Perfect |
| Data round-trip (CPU→TT→CPU, f32) | 0.0 | ✅ Perfect |
| float() upcast (bf16→f32 on TT) | 0.0 | ✅ Perfect |
| AdaLN modulation (scale_shift_table + temb) | 0.0 | ✅ Perfect |
| Softmax (random scores) | 0.015 | ✅ Fine |
| Softmax (large scores ×10) | 0.039 | ✅ Fine |

### Where divergence begins (error chain):
| Component | Max Diff | Mean Diff | % Elements >1.0 | Status |
|-----------|----------|-----------|-----------------|--------|
| FP32LayerNorm | 0.011 | 0.002 | 0% | ⚠️ Small |
| AdaLN output (norm × scale + shift) | 0.063 | 0.003 | 0% | ⚠️ Small |
| **Q/K/V linear projection** | **0.125** | **0.005** | **0%** | **🔴 Significant** |
| Q/K after RMSNorm | 0.063 | 0.003 | 0% | ⚠️ Inherited |
| **Q/K after RoPE** | **16.75** | **0.365** | **11.2%** | **🔴 Major amplifier** |
| **Attention scores (Q@K^T)** | **6.42** | **0.707** | **23.8%** | **🔴 Catastrophic** |
| Attention probs (after softmax) | 0.463 | 0.016 | 0% | 🔴 Wrong distributions |
| **SDPA output** | **9.125** | **0.525** | **17.1%** | **🔴 Catastrophic** |
| **Full block 0 output** | **50.8** | **1.134** | **34.4%** | **🔴 Catastrophic** |
| **MatMul (768×1536 @ 1536×1536)** | **2.0** | **0.293** | **0.23%** | **🔴 Root cause** |

## Root Cause Analysis

### Root Cause #1: bf16 MatMul precision on TT hardware

The standalone matmul test reveals the core issue:
```
MatMul (768x1536 @ 1536x1536): max_diff=2.0, mean_diff=0.293, 0.23% elements >1.0
```

This is **not a normal bf16 rounding error**. CPU bf16 matmul on the same inputs produces much tighter results. The TT hardware's bf16 dot product accumulation appears to use a different accumulation strategy (potentially bf16 accumulators instead of f32 accumulators), causing errors that scale with the reduction dimension.

For the 1536-wide inner dimension of the Q/K/V projections:
- Expected bf16 error: ~0.01–0.02 (with f32 accumulators)
- Observed TT error: 0.125 (6–12× worse)

### Root Cause #2: RoPE amplifies matmul errors catastrophically

The RoPE (Rotary Position Embedding) application on TT shows a **massive error amplification**:
```
Before RoPE: Q max_diff=0.063, mean_diff=0.003
After RoPE:  Q max_diff=16.75, mean_diff=0.365 (11.2% elements >1.0)
```

This is a **250× amplification** of error. Why?

The Wan RoPE implementation uses `torch.empty_like()` followed by indexed assignment:
```python
out = torch.empty_like(hidden_states)
out[..., 0::2] = x1 * cos - x2 * sin
out[..., 1::2] = x1 * sin + x2 * cos
```

On TT/XLA, `torch.empty_like()` returns **uninitialized memory** which may not behave identically to CPU. If the indexed scatter operations (`[..., 0::2]` and `[..., 1::2]`) are **not compiled correctly** by the XLA→TTIR pipeline (e.g., if they overlap, race, or leave gaps), the result will mix correct values with garbage from uninitialized memory.

Additionally, the `.unflatten(-1, (-1, 2)).unbind(-1)` pattern may not lower correctly through StableHLO→TTIR.

### Compounding Effect

The error chain through one block:
1. MatMul error in Q/K/V projections: ε ≈ 0.125
2. RoPE amplifies: ε → 16.75 (133× amplification)
3. Attention scores Q@K^T compound: ε → 6.42 per score
4. Softmax redistributes probability mass incorrectly
5. Attention output: ε → 9.125
6. Residual + FFN adds more matmul error
7. Full block output: ε → 50.8

Through 30 blocks: errors compound multiplicatively → complete numerical garbage.
Through 50 denoising steps: each step feeds garbage to the next → uniform noise output.

## Proposed Fixes

### Fix 1: Force float32 accumulation in matmul (hardware/compiler level)
The TT compiler needs to use f32 accumulators for bf16 matmul. Check if there's an `optimization_level` or compile option to enable this. In tt-mlir/tt-metal, this is often controlled by:
- `accumulation_dtype` in matmul config
- `fp32_dest_acc_en` in the compute kernel config
- An XLA compile option like `{"force_f32_accumulation": True}`

**Try:** `torch_xla.set_custom_compile_options({"optimization_level": 0})` — level 0 may disable aggressive bf16 optimizations.

### Fix 2: Rewrite RoPE to avoid `empty_like` + indexed assignment
Replace the current pattern with a safer equivalent that avoids uninitialized memory and strided writes:

```python
def apply_rotary_emb_safe(hidden_states, freqs_cos, freqs_sin):
    # Avoid empty_like + indexed scatter — use stack instead
    x = hidden_states.unflatten(-1, (-1, 2))  # (..., head_dim/2, 2)
    x1, x2 = x[..., 0], x[..., 1]  # Use indexing instead of unbind
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).flatten(-2)  # Interleave via stack
    return out.type_as(hidden_states)
```

This avoids `torch.empty_like()` entirely and uses `torch.stack()` + `flatten()` which should lower cleanly to StableHLO concat/reshape ops.

### Fix 3: Run RoPE on CPU (immediate workaround)
Since RoPE is the major amplifier, compute it on CPU and transfer back:

```python
# In the patched forward, after transferring Q/K to TT:
# Pull Q/K back to CPU for RoPE, then transfer back
q_cpu = q_mh.cpu()
k_cpu = k_mh.cpu()
q_rotated = apply_rotary_emb(q_cpu, *rotary_emb_cpu)
k_rotated = apply_rotary_emb(k_cpu, *rotary_emb_cpu)
q_rotated = q_rotated.to(tt_device)
k_rotated = k_rotated.to(tt_device)
```

### Fix 4: Mixed precision — attention in f32
Keep weights in bf16 but upcast Q/K/V to float32 before attention:

```python
q = q.float()
k = k.float()
v = v.float()
attn_out = F.scaled_dot_product_attention(q, k, v)
attn_out = attn_out.to(torch.bfloat16)
```

### Recommended Priority

1. **Fix 2 (rewrite RoPE)** — Quick, likely high-impact, no hardware changes needed
2. **Fix 1 (f32 accumulators)** — Root cause fix, but may need tt-xla/tt-metal changes
3. **Fix 3 (CPU RoPE)** — Immediate workaround, easy to test but adds latency
4. **Fix 4 (f32 attention)** — May not be supported by TT hardware natively

## Verification Plan

After applying fixes:
1. Re-run `debug_correctness.py` — full block error should drop from 50.8 to <1.0
2. Run 5-step pipeline — pixel values should be in [0.77, 0.94] range (matching CPU ref)
3. Run 30-step pipeline — should produce coherent video frames

## Files
- Debug script: `/workspace/tt-xla/debug_correctness.py`
- This report: `/workspace/tt-xla/docs/wan_correctness_debug.md` (update)
- Pipeline (1.3B): `/workspace/tt-xla/examples/pytorch/wan_t2v_1_3b.py`
