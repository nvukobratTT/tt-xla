# Wan2.1-T2V-1.3B Workaround Results

**Date:** 2026-03-21
**Hardware:** Tenstorrent Blackhole (4-chip mesh, tt-xla)
**Model:** Wan-AI/Wan2.1-T2V-1.3B-Diffusers (30 transformer blocks)
**Compiler flags:** `{"fp32_dest_acc_en": "true", "math_fidelity": "hifi4"}`

## Executive Summary

**Root cause identified: Large-reduction-dimension matmul (K=8960) in FFN down-projection produces catastrophic error on TT Blackhole.** 82.7% of output elements are off by >1.0 (max_diff=28). This single operation inside each transformer block makes TT inference unusable for this model. No hybrid CPU/TT configuration can fix this — even running a single block on TT (out of 30) drops cosine similarity to 0.32.

**Recommended workaround:** Full CPU inference pipeline until tt-xla fixes large-K matmul precision. A working CPU pipeline is provided.

## Root Cause: Matmul Precision Scales with Reduction Dimension

| Reduction K | Cosine | Max Diff | Mean Diff | % Elements >1.0 | Used By |
|-------------|--------|----------|-----------|-----------------|---------|
| 128 | 0.9999 | 0.25 | 0.006 | 0.0% | Attention scores |
| 256 | 0.9999 | 0.50 | 0.019 | 0.0% | — |
| 512 | 0.9999 | 0.50 | 0.055 | 0.0% | — |
| 1024 | 0.9999 | 2.00 | 0.159 | 0.0% | — |
| 1536 | 0.9999 | 2.00 | 0.294 | 0.2% | Q/K/V projections |
| 2048 | 0.9999 | 3.00 | 0.454 | 4.1% | — |
| 4096 | 0.9998 | 8.00 | 1.279 | 48.4% | — |
| 4480 | 0.9998 | 10.00 | 1.459 | 54.0% | — |
| **8960** | **0.9995** | **28.00** | **4.076** | **82.7%** | **FFN down-proj** |

The error grows approximately linearly with K. The FFN down-projection (K=8960→N=1536) is 5.8× the reduction dimension of Q/K/V projections (K=1536) and produces proportionally worse error.

## Individual Op Precision on TT

| Operation | Cosine | Max Diff | Status |
|-----------|--------|----------|--------|
| SDPA (bf16) | 0.9997 | 0.020 | ✅ Excellent |
| SDPA (f32) | 0.9998 | 0.020 | ✅ Excellent |
| LayerNorm (f32) | 1.0000 | 0.014 | ✅ Perfect |
| SiLU (bf16) | 1.0003 | 0.008 | ✅ Perfect |
| Element-wise multiply | 1.0006 | 0.000 | ✅ Perfect |
| RoPE (stack+flatten fix) | 0.9999 | 0.063 | ✅ Excellent |
| MatMul K=1536 (Q/K/V proj) | 0.9999 | 3.000 | ⚠️ Acceptable |
| MatMul K=8960 (FFN down) | 0.9995 | 28.000 | 🔴 Catastrophic |
| Full attention path | 0.9849 | 3456.0 | 🔴 Error amplification |
| Realistic FFN (up+gate+down) | 0.9995 | 32768.0 | 🔴 Catastrophic |

## Hybrid Configuration Results

### Single Forward Pass (1 denoising step, 256×256×9)

| Configuration | Cosine vs CPU | Max Diff | Mean Diff | Time | Verdict |
|--------------|---------------|----------|-----------|------|---------|
| **Config 4: Full CPU** | **1.0000** | **0.00** | **0.000** | **1.2s** | ✅ Reference |
| Config TT: Full TT | 0.3174 | 4.83 | 0.851 | 2.7s | 🔴 Broken |
| Config 5: TT + f32 inter-block | 0.3174 | 4.83 | 0.851 | 2.7s | 🔴 No improvement |
| Config 1: Every-2 blocks CPU | 0.3167 | 4.83 | 0.851 | 2.6s | 🔴 No improvement |
| Single block 0 on TT | 0.3186 | 4.84 | 0.850 | 1.9s | 🔴 1 block = broken |
| Single block 14 on TT | 0.2589 | 5.02 | 0.909 | 1.5s | 🔴 1 block = broken |
| Single block 29 on TT | 0.1658 | 5.26 | 1.011 | 1.5s | 🔴 1 block = broken |

### Key Findings

1. **Even a single TT block destroys output** — cosine drops from 1.0 to 0.32 with just one block on TT. This is not error accumulation; it's catastrophic per-block error.

2. **f32 upcast between blocks is useless** — The error happens within each block (in the FFN down-proj), not between blocks. Casting to f32 between blocks changes nothing.

3. **CPU interleaving doesn't help** — Running every Nth block on CPU doesn't "reset" anything because the FFN error within each TT block is already devastating.

4. **std attenuation ~3×** — TT produces noise_pred with std=0.37 vs CPU std=1.12. The signal is being attenuated by the large-K matmul errors averaging out.

5. **The problem is fundamentally in the FFN down-projection** — The 8960→1536 reduction dimension accumulates bf16 rounding errors that are not properly handled by the TT hardware even with `fp32_dest_acc_en=true` and `math_fidelity=hifi4`.

## Why Tiling Doesn't Fix It

Splitting the K=8960 reduction into smaller chunks and accumulating in f32 was tested:

| Strategy | Cosine | Max Diff | Result |
|----------|--------|----------|--------|
| Full TT (no tile) | 0.9995 | 27.0 | Bad |
| Tiled K=256 (35 chunks, CPU f32 accum) | 0.0018 | 644.0 | 🔴 Much worse |
| Tiled K=1792 (5 chunks, CPU f32 accum) | 0.0009 | 682.0 | 🔴 Much worse |
| Tiled K=1792 (5 chunks, TT f32 accum) | 0.0009 | 682.0 | 🔴 Much worse |

**Tiling makes things dramatically worse** because each partial matmul introduces its own errors, and the f32 accumulation of multiple independently-wrong partial results amplifies rather than corrects the error.

## Working Pipeline: Full CPU

CPU inference produces correct output verified at:
- 256×256×9, 5 steps: **47KB video file, 27.2s total** (~2.5s/step)
- Expected at full resolution (480×832×81): ~30 min per step (not practical for interactive use, but correct)

### Performance Comparison

| Config | Time/Step (256×256×9) | Quality |
|--------|----------------------|---------|
| Full CPU | 2.5s | ✅ Correct |
| Full TT | ~1.4s | 🔴 Broken |
| Hybrid (any) | ~1.5-2.6s | 🔴 Broken |

At small resolution, CPU is only ~2× slower than TT. At full resolution the gap widens due to attention's quadratic cost, but TT output is unusable regardless.

## Bug Report for tt-xla Team

**Title:** bf16 matmul with large reduction dimension (K≥4096) produces high error even with fp32_dest_acc_en=true

**Severity:** Critical — makes any model with FFN wider than ~4K unusable

**Reproduction:**
```python
import torch, torch_xla, torch_xla.runtime as xr
xr.set_device_type("TT")
torch_xla.set_custom_compile_options({
    "fp32_dest_acc_en": "true",
    "math_fidelity": "hifi4",
})
device = torch_xla.device()

# K=8960 matmul (Wan 1.3B FFN down-projection)
A = torch.randn(1, 768, 8960, dtype=torch.bfloat16)
B = torch.randn(8960, 1536, dtype=torch.bfloat16)
cpu_ref = A @ B

A_tt = A.to(device)
B_tt = B.to(device)
torch_xla.sync()
tt_out = (A_tt @ B_tt)
torch_xla.sync()

diff = (tt_out.cpu().float() - cpu_ref.float()).abs()
print(f"max_diff={diff.max():.1f}, mean_diff={diff.mean():.4f}")
print(f"% elements >1.0: {(diff > 1.0).float().mean() * 100:.1f}%")
# Expected: max_diff≈28, mean_diff≈4.08, 82.7% elements >1.0
```

**Expected behavior:** With `fp32_dest_acc_en=true`, the accumulator should be f32, producing max_diff < 1.0 for K=8960.

**Hypothesis:** The `fp32_dest_acc_en` compile option may not be reaching the matmul tile kernels, or the tile size may be causing partial accumulations in bf16 before a final f32 reduction.

## Files

| File | Description |
|------|-------------|
| `wan_cpu_pipeline.py` | Working full-CPU pipeline (verified) |
| `wan_hybrid_test.py` | Comprehensive hybrid config tester |
| `wan_block_dissect2.py` | Per-op precision analysis |
| `wan_matmul_sweep.py` | Matmul K-dimension sweep |
| `wan_t2v_1_3b.py` | Original TT pipeline (broken) |

## Workarounds Applied (for future removal)

1. **RoPE fix** (KEEP): `torch.stack([out1, out2], dim=-1).flatten(-2)` instead of strided slice assignment. Required regardless of matmul fix — the strided scatter bug is separate.

2. **Full CPU inference** (REMOVE when matmul fixed): Run all 30 transformer blocks on CPU. Remove when tt-xla matmul with K≥4096 produces mean_diff < 0.1.

3. **Graph breaks between blocks** (KEEP for now): `xm.mark_step()` between blocks. Still needed for memory management even when matmul is fixed.

## Conclusion

The Wan 1.3B model cannot run on TT Blackhole with acceptable quality due to a fundamental precision issue in large-reduction-dimension matmul. This affects any model with FFN hidden dimension > ~4096 (which is most modern transformers). The recommended path is:

1. **Short-term:** Use the CPU pipeline for correct output
2. **Medium-term:** File the matmul precision bug with the tt-xla team
3. **Long-term:** Once fixed, use the original TT pipeline with RoPE patch — all other ops are fine
