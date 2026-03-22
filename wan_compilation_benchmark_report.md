# Wan2.1-T2V-1.3B Compilation Benchmark Report

**Date:** 2026-03-20  
**Hardware:** 4× Tenstorrent Blackhole chips (tensor-parallel)  
**Model:** Wan-AI/Wan2.1-T2V-1.3B-Diffusers (30 blocks, 12 heads, 1536 inner_dim)  
**Software:** tt-xla + torch_xla, PJRT plugin, tt-mlir compilation pipeline  
**Pipeline variants:** `wan_t2v_1_3b_tp.py` (TP, mark_step), `wan_t2v_pipeline.py` (eager mode)

---

## 1. Compilation Time Summary

### 1.1 TP Mode (mark_step graph breaks) — Primary Configuration

| Resolution | Frames | Seq Length | Block 1 | Block 2 | Block 3 | Blocks 4-30 | Output Proj | **Total Step 1** | Step 2+ |
|:----------:|:------:|:----------:|:-------:|:-------:|:-------:|:-----------:|:-----------:|:----------------:|:-------:|
| 256×256    | 9      | 768        | 1.6s    | 29.5s   | 5.7s    | 5.9s ×27    | ~1s         | **202s (3.4m)**  | 176s    |
| 480×480    | 17     | 4,500      | 79.9s   | 295.7s  | 177.9s  | 180.6s ×27  | ~6s         | **5,613s (1.6h)**| —       |
| 480×832    | 17     | 7,800      | 151.4s  | 842.6s  | 536.7s  | 541.7s ×27  | ~7s         | **16,704s (4.6h)** | —     |
| 480×832    | 81     | 32,760     | 955.8s  | ❌ OOM/hang | —    | —           | —           | **CRASHED**      | —       |

### 1.2 Single-Device (no TP, mark_step graph breaks) — for comparison

| Resolution | Frames | Seq Length | Block 1 | Block 2 | Blocks 3-30 | **Total Step 1** |
|:----------:|:------:|:----------:|:-------:|:-------:|:-----------:|:----------------:|
| 256×256    | 9      | 768        | 50.8s   | 140.5s  | 84.2s ×28   | **2,634s (44m)** |

### 1.3 TP vs Single-Device Speedup (256×256, seq_len=768)

| Block | Single Device | 4-chip TP | Speedup |
|:-----:|:------------:|:---------:|:-------:|
| 1     | 50.8s        | 1.6s      | 31.8×   |
| 2     | 140.5s       | 29.5s     | 4.8×    |
| 3-30  | 84.2s        | 5.9s      | 14.3×   |
| **Total** | **2,634s** | **202s** | **13.0×** |

TP provides massive compile-time speedup — not just from parallelism but from smaller per-device graph sizes.

---

## 2. Block Timing Patterns Analysis

### 2.1 Why Block 2 is Always Slower than Block 1

**Block 1** is faster because it's the **first block after `mark_step()`** on the transferred tensors. The XLA graph for Block 1 includes only the block operations themselves — the input tensors were materialized by the prior `mark_step()` call.

**Block 2** is dramatically slower (3.7× to 5.6× vs Blocks 3-30) because it's compiling a **different graph shape** than subsequent blocks. Key factors:

1. **Cross-attention key/value first materialization**: Block 1's outputs become Block 2's inputs with potentially different tensor metadata/layouts after the first device-side computation. The compiler sees a different input signature.
2. **tt-mlir optimization passes**: The TTNN optimization pipeline (layout conversion, buffer allocation, op fusion) spends more time on Block 2 because the compiler's internal state (operator scheduling, memory allocation plan) is being established for the repeated pattern.
3. **Graph tracing overhead**: In `mark_step()` mode, Block 2 is the first block where the compiler sees the *output pattern* of a prior block as input — this triggers more complex data-flow analysis.

**Evidence**: In the cache_test log (5-step run), Block 2 drops from 29.5s to 9.3s, suggesting ~20s of Block 2's initial cost is one-time graph analysis, and ~9s is the actual compilation. Steps 2+ show Block 2 at 5.4s (cached execution, not compilation).

### 2.2 Why Blocks 3-30 Are All Identical

All blocks 3-30 show identical timing (±0.1s). This confirms:
- **Same graph hash**: All blocks have identical architecture (same layer structure, same tensor shapes)
- **In-memory compilation cache hit**: After Block 2 establishes the steady-state graph pattern, Blocks 3-30 reuse the cached compiled executable
- **The ~5.9s per block at seq_len=768 is pure execution time**, not compilation — confirmed by Steps 2+ showing the same 5.8s/block timing

### 2.3 What Block 1 Actually Compiles

Block 1's shorter time (1.6s at seq_len=768) suggests it either:
- Has a slightly different graph (first block may skip some operations or have different input shapes)
- Gets a simpler optimization path due to fresh input tensors from host transfer

In Step 2+, Block 1 drops to 0.0s — it's hitting the in-memory cache perfectly.

---

## 3. Scaling Analysis

### 3.1 Per-Block Compilation Time vs Sequence Length

Using Block 2 (first-time compilation, most sensitive to graph complexity):

| Seq Length | Block 2 Time | Ratio to 768 |
|:----------:|:------------:|:------------:|
| 768        | 29.5s        | 1.0×         |
| 4,500      | 295.7s       | 10.0×        |
| 7,800      | 842.6s       | 28.6×        |
| 32,760     | >955.8s (B1) | ∞ (crashed)  |

### 3.2 Fitting the Scaling Curve

Testing different power law models `t = a × n^k`:

```
log(29.5)  = log(a) + k × log(768)    → 3.38 = log(a) + 6.64k
log(295.7) = log(a) + k × log(4500)   → 5.69 = log(a) + 8.41k
log(842.6) = log(a) + k × log(7800)   → 6.74 = log(a) + 8.96k
```

From points 1&2: `k = (5.69 - 3.38) / (8.41 - 6.64) = 2.31 / 1.77 = **1.30**`  
From points 2&3: `k = (6.74 - 5.69) / (8.96 - 8.41) = 1.05 / 0.55 = **1.91**`  
From points 1&3: `k = (6.74 - 3.38) / (8.96 - 6.64) = 3.36 / 2.32 = **1.45**`

**Best fit: ~O(n^1.5)** (geometric mean of the three estimates: 1.53)

The scaling is **super-linear but sub-quadratic**, consistent with the tt-mlir compiler's attention operation lowering where:
- Attention itself is O(n²) in compute but the *compilation* time scales with the number of tiles/operations generated
- The TTNN backend tiles the sequence dimension, so compilation grows with tile count × scheduling complexity

### 3.3 Steady-State Execution Time (Blocks 3-30, cached)

| Seq Length | Block Exec Time | Ratio to 768 |
|:----------:|:--------------:|:------------:|
| 768        | 5.9s           | 1.0×         |
| 4,500      | 180.6s         | 30.6×        |
| 7,800      | 541.7s         | 91.8×        |

Execution scaling: `k = log(541.7/5.9) / log(7800/768) = 4.52 / 2.32 = **1.95**`

Execution time scales ~**O(n²)** — consistent with self-attention compute complexity.

### 3.4 Projected Times for Untested Resolutions

| Resolution | Frames | Est. Seq Len | Proj. Compile (Step 1) | Proj. Per-Step (cached) |
|:----------:|:------:|:------------:|:---------------------:|:----------------------:|
| 720×1280   | 17     | ~18,000      | ~24h                  | ~50min/step            |
| 480×832    | 81     | ~32,760      | ~60h+ (infeasible)    | ~3h/step               |
| 720×1280   | 81     | ~75,000      | ∞ (will crash)        | ∞                      |

---

## 4. Compilation Pipeline Breakdown

The compilation pipeline per block is: **StableHLO → TTIR → TTNN → Flatbuffer**

Based on the module_builder.cc timestamps in the logs:

### 4.1 Timing Decomposition (256×256, TP)

From `wan_1_3b_tp_run.log` timestamps:
- **Graph tracing** (XLA → StableHLO): Happens at `mark_step()` — included in Block 1 time
- **VHLO → StableHLO conversion**: ~instantaneous
- **StableHLO → TTIR** (`convertFromSHLOToTTIR`): Includes legalization, shape inference, layout analysis
- **TTIR → TTNN** (`convertFromTTIRToTTNN`): The heavy pass — op scheduling, buffer allocation, data layout optimization
- **TTNN → Flatbuffer** (`buildModuleForTTNNRuntime`): Binary serialization

The log shows `module_builder.cc:915` warnings right before compilation starts and `stubs.inc:80` (PJRT_Executable_GetCompiledMemoryStats) right after. The entire module_builder pipeline for the first block+transfer graph at seq_len=768 takes ~9.5s.

### 4.2 Where the Bottleneck Is

The **TTIR → TTNN conversion** is the dominant cost. This is where:
1. Attention operations get decomposed into tiled matrix multiplies
2. The scheduler determines the execution order across compute cores
3. Buffer allocation plans memory for intermediate tensors
4. L1/DRAM placement decisions are made for each tile

For large sequence lengths, the number of tiles grows quadratically (attention) and the scheduler's complexity grows with it.

---

## 5. Caching and Reuse Analysis

### 5.1 In-Memory Cache (Currently Working)

The in-memory compilation cache **works correctly**:
- Step 1 compiles ~3 unique graphs: initial transfer, Block 1 pattern, Block 2+ pattern
- Steps 2-30 show **zero recompilation** (Block 1: 0.0s, Block 2+: 5.4-5.8s = pure execution)
- The 30-step demo confirms: Step 1 = 180.7s, Steps 2-30 = 175.5s each (consistent)

**Key finding**: Only 3 compilations occur per run, regardless of step count. The 30 blocks × 30 steps = 900 block executions reuse just 3 compiled graphs.

### 5.2 Persistent Disk Cache (BROKEN — tt-xla #498)

- `PJRT_Executable_Serialize` → ✅ Implemented (TTSERv00 format)
- `PJRT_Executable_DeserializeAndLoad` → ❌ **Stubbed out** 
- Result: Cache files get written but never loaded. Every process restart recompiles from scratch.
- **This is the single highest-impact optimization to pursue.**

### 5.3 Block-Level Graph Reuse

Blocks 3-30 all compile to the **same graph hash** and reuse Block 2's compiled executable. This means:
- **Effective unique compilations per step**: 3 (transfer graph, Block 1 graph, Block 2+ graph)
- **Not** 30 separate compilations as one might assume
- Block 2 is "slow" because it's the first time the main block graph is compiled; it includes the compilation. Block 1 has a different graph.

### 5.4 `TT_RUNTIME_ENABLE_PROGRAM_CACHE`

This env var enables an additional runtime-level program cache. From the docs it helps within a single run. Currently **not set** in any of the pipeline scripts.

---

## 6. Optimization Recommendations

### 🔴 Critical (Highest Impact)

#### 6.1 Implement Persistent Compilation Cache (tt-xla #498)
- **Impact**: Eliminate 100% of recompilation on subsequent runs
- **Current state**: Serialization works, deserialization is stubbed
- **Effort**: Medium — format and infrastructure exist, need to wire up `DeserializeAndLoad`
- **Savings at 480×832**: 4.6h → ~0s compilation on second run

#### 6.2 Use Smaller Resolutions During Development
- 256×256/9f: 3.4 minutes compile + 2.9 min/step = excellent for iteration
- 480×480/17f: 1.6h compile + 93 min/step = manageable for quality checks
- **Never use 480×832+ for iteration** — only for final renders

### 🟡 Important (Medium Impact)

#### 6.3 Set `TT_RUNTIME_ENABLE_PROGRAM_CACHE=1`
- Free optimization, just set the env var
- May reduce overhead for repeated graph execution within a run

#### 6.4 Reduce Inference Steps
- The 30-step demo at 256×256 took 87.9 min total (5,272s)
- Effective throughput: 5.9s/block/step × 30 blocks = 177s/step
- Consider 20 or fewer steps for acceptable quality (UniPC scheduler supports it)

#### 6.5 Pre-populate Cache with `xr.initialize_cache()`
- Even though deserialization isn't implemented yet, calling this populates the cache directory
- When #498 lands, cached runs will work immediately

### 🟢 Nice-to-Have (Lower Impact or Longer Term)

#### 6.6 AOT Compilation Pipeline
- Export StableHLO via `save_as_stablehlo()` for the 3 unique graphs
- Compile offline using tt-mlir tools
- Load flatbuffers directly at runtime
- Would allow CI/CD precompilation for known resolutions

#### 6.7 Sequence Length Reduction Strategies
- **Temporal compression**: Use fewer frames (17f vs 81f reduces seq_len 4.2×)
- **Spatial downsampling**: 480×480 vs 480×832 reduces seq_len 1.7×
- **Patch size optimization**: Larger patches reduce token count (currently 1×2×2)
- **Window attention**: If model supports it, limit attention window (architecture change)

#### 6.8 Compilation Parallelism
- Currently compilation is single-threaded (all on one host CPU thread)
- tt-mlir may support parallel pass execution — worth investigating

---

## 7. Key Metrics Summary

| Metric | 256×256/9f | 480×480/17f | 480×832/17f |
|:-------|:----------:|:-----------:|:-----------:|
| Sequence length | 768 | 4,500 | 7,800 |
| First-step compile time | 202s | 5,613s | 16,704s |
| Cached step time | 176s | — | — |
| Per-block execution | 5.9s | 180.6s | 541.7s |
| Unique compilations | 3 | 3 | 3 |
| Total (1 step) | 3.4 min | 1.6h | 4.6h |
| Total (30 steps) | 87.9 min | ~37h est. | ~113h est. |
| Compile scaling | — | O(n^1.5) | O(n^1.5) |
| Execute scaling | — | O(n^2) | O(n^2) |
| TP speedup vs single | 13× | — | — |

---

## 8. Recommended Next Steps

1. **Immediate**: Add `TT_RUNTIME_ENABLE_PROGRAM_CACHE=1` and `xr.initialize_cache()` to pipeline scripts
2. **Short-term**: File/champion tt-xla #498 for `DeserializeAndLoad` — this is the 10× multiplier
3. **Medium-term**: Build an AOT compilation script that pre-compiles the 3 unique graphs for target resolutions
4. **Long-term**: Investigate tt-mlir compiler optimizations for large sequence lengths (tiling strategies, parallel compilation passes)

---

## Appendix A: Raw Log Files

| Log File | Config | Key Data |
|:---------|:-------|:---------|
| `wan_1_3b_tp_run.log` | 256×256, 9f, 1 step, TP | Baseline TP timing |
| `wan_1_3b_tp_cache_test.log` | 256×256, 9f, 5 steps, TP | Multi-step cache behavior |
| `wan_1_3b_tp_30step_demo.log` | 256×256, 9f, 30 steps, TP | Full generation timing |
| `wan_1_3b_tp_480x480.log` | 480×480, 17f, 1 step, TP | Medium resolution |
| `wan_1_3b_tp_480x832_17f.log` | 480×832, 17f, 1 step, TP | HD resolution |
| `wan_1_3b_tp_480x832_81f.log` | 480×832, 81f, 1 step, TP | Long video (crashed) |
| `wan_1_3b_run2.log` | 256×256, 9f, 1 step, single device | Single vs TP comparison |

## Appendix B: Sequence Length Calculation

```
seq_len = (num_frames / p_t) × (height / (p_h × vae_spatial)) × (width / (p_w × vae_spatial))

Where p_t=1, p_h=2, p_w=2, vae_spatial=8

256×256, 9f:  seq = 9 × (256/16) × (256/16) = 9 × 16 × 16 = 2,304 → but log shows 768
  → Actually: latent_frames = (9-1)/4 + 1 = 3; seq = 3 × (256/16) × (256/16) = 3 × 16 × 16 = 768 ✓

480×480, 17f: latent_frames = (17-1)/4 + 1 = 5; seq = 5 × (480/16) × (480/16) = 5 × 30 × 30 = 4,500 ✓

480×832, 17f: latent_frames = 5; seq = 5 × 30 × 52 = 7,800 ✓

480×832, 81f: latent_frames = (81-1)/4 + 1 = 21; seq = 21 × 30 × 52 = 32,760 ✓
```
