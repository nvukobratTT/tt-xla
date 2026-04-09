# openclaw-patches — Blackhole / Wan2.1 Fix Collection

Patches and fixes developed during Mochi/Wan2.1-T2V bringup on 4× Blackhole chips.
Covers correctness bugs, compilation speedups, and runtime OOM improvements.

## Current State (as of 2026-04-09)

| Layer | Branch | Status |
|-------|--------|--------|
| **tt-xla** (PJRT layer) | `fix/oom-silent-hang` | **PR #12 open** — not yet merged to main |
| **tt-mlir** (compiler) | `fix/bf16-matmul-padop-precision` | **Local only** — 7 commits on top of submodule HEAD |

The `mochi-bringup` branch has these submodule pointer + patch files committed.
To run the full fixed stack you need to rebuild after ensuring both layers are on the right branches.

---

## tt-xla Patches (PJRT Layer) — [PR #12](https://github.com/nvukobratTT/tt-xla/pull/12)

These modify C++ files in the `pjrt_implementation/` directory of tt-xla.

### Fix 1 — Remove silent hang on device OOM
**File:** `fix-oom-silent-hang-on-submit-failure.patch`
**Issue:** [GH #11](https://github.com/nvukobratTT/tt-xla/issues/11)

When DRAM OOM occurs at >1184 tokens, `tt::runtime::submit` throws `TT_FATAL`. The exception
was caught by `invoke_noexcept`, which then called `closeMeshDevice()`. For SPMD multi-chip
models, ethernet cores are stuck waiting for all-reduce data that never arrives, so
`closeMeshDevice()` blocks indefinitely — appearing as a silent hang.

**Fix:** Adds `m_has_fatal_device_error` flag to `ClientInstance`. On submit failure, sets
the flag and returns `kInternal` immediately. Subsequent `execute()` calls fail fast.

**Recovery after OOM:** `tt-smi -r 0,1,2,3` from host, then restart the container.

### Fix 2 — Implement `PJRT_Executable_DeserializeAndLoad` (persistent compilation cache)
**File:** `implement-deserialize-and-load.patch`
**Issue:** [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2)

`PJRT_Executable_DeserializeAndLoad` was a `_STUB` returning `kUnimplemented`. This means
`xr.initialize_cache()` never reused compiled binaries across runs — every run recompiled
from scratch (93 minutes for Wan2.1 on 4 chips).

**Fix:** Reads the serialized TTSERv00 binary, extracts the flatbuffer payload, reconstructs
`FlatbufferLoadedExecutableImage` metadata from the flatbuffer binary API, wraps it in
`FlatbufferLoadedExecutableInstance`.

**Usage:**
```python
import torch_xla as xr
xr.initialize_cache("/tmp/tt_compile_cache", readonly=False)
# First run: compiles (~93 min). Subsequent runs: loads from cache (~0 s).
```

**Limitation:** Non-Identity SPMD I/O shardings are not reconstructed. For torch_xla TP
models (Mochi, Wan2.1), XLA-visible I/O tensors are replicated, so this is fine in practice.

---

## tt-mlir Patches (Compiler Layer) — Local Branch `fix/bf16-matmul-padop-precision`

Seven commits on the tt-mlir submodule. Apply with `git am` from inside the submodule.
The format-patch files are in `openclaw-patches/tt-mlir/`.

```bash
cd /workspace/tt-xla/third_party/tt-mlir/src/tt-mlir
git checkout fix/bf16-matmul-padop-precision   # already exists if submodule is correct
# OR apply from scratch:
git am /workspace/tt-xla/openclaw-patches/tt-mlir/0002-*.patch
git am /workspace/tt-xla/openclaw-patches/tt-mlir/0003-*.patch
git am /workspace/tt-xla/openclaw-patches/tt-mlir/0004-*.patch
git am /workspace/tt-xla/openclaw-patches/tt-mlir/0005-*.patch
git am /workspace/tt-xla/openclaw-patches/tt-mlir/0006-*.patch
git am /workspace/tt-xla/openclaw-patches/tt-mlir/0007-*.patch
git am /workspace/tt-xla/openclaw-patches/tt-mlir/0008-*.patch
```

### Patch 1 — bf16 matmul precision (packer_l1_acc)
**File:** `tt-mlir/0002-fix-TTNN-propagate-packer_l1_acc...`
**Issue:** [GH #1](https://github.com/nvukobratTT/tt-xla/issues/1)

`fp32_dest_acc_en=true` with bf16 ops should also enable `packer_l1_acc` (TTNN does this
automatically in `init_device_compute_kernel_config`). tt-xla's explicit
`DeviceComputeKernelConfigAttr` skipped this, causing catastrophic error for raw `torch.matmul`
with large K (>4096). `nn.Linear` was unaffected because its compute config path differed.

**Fix:** In `TTNNPipelines.cpp`, when `fp32DestAccEn` is true, also set `packerL1Acc = true`.

### Patch 2 — PadOp interior-padding index order (row-major)
**File:** `tt-mlir/0003-fix-StableHLO-TTIR-fix-PadOp...`
**Issue:** [GH #5](https://github.com/nvukobratTT/tt-xla/issues/5)

`out[..., 0::2] = val` (strided slice assignment → RoPE) lowers to `stablehlo.pad` with
interior padding, then to TTIR's `PadOp`. The index computation iterated outermost-dim-first
(column-major) while source elements were flattened row-major, placing elements at wrong
positions.

**Fix:** Change iteration in `StableHLOToTTIROpPadOpConversionPattern` to innermost-first.

### Patch 3 — `maxLegalLayoutsElementwise` compile time option
**File:** `tt-mlir/0004-feat-TTNN-optimizer-add-maxLegalLayoutsElementwise...`
**Issue:** [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2)

`LegalOpLayoutAnalysis.run()` explored too many layout combinations for elementwise ops
(which have low actual layout sensitivity). Capping at 2 cuts layout search time significantly.

**Fix:** New `maxLegalLayoutsElementwise` pipeline option. Set to 2 for a ~20% compile speedup.

### Patch 4 — `TTNNCheckDRAMBudget` compile-time DRAM OOM detection
**File:** `tt-mlir/0005-feat-TTNN-add-TTNNCheckDRAMBudget...`
**Issue:** [GH #11](https://github.com/nvukobratTT/tt-xla/issues/11)

Mochi at 2080 tokens silently hangs because DRAM OOM is caught by `invoke_noexcept` at
runtime. This pass simulates tensor lifetimes after `ttnn-deallocate` using a first-fit
freelist allocator (fragmentation-aware), and emits a compile-time diagnostic if estimated
peak DRAM exceeds per-chip budget (32 GB on Blackhole).

**Fix:** New MLIR pass `TTNNCheckDRAMBudget` added to the TTNN pipeline.

### Patch 5 — Exclude SDPA ops from `LegalOpLayoutAnalysis`
**File:** `tt-mlir/0006-fix-TTNN-exclude-SDPA-ops...`
**Issue:** [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2), upstream GH #5283

The optimizer ran after SDPA fusion and re-optimized the SDPA output layout. The SDPA kernel
wrote in its native format but the downstream op saw a different layout tag → golden mismatch.

**Fix:** `isValidAnalysisTarget()` excludes SDPA ops. This is a prerequisite for reliably
enabling SDPA fusion (which eliminates materialized QK^T tensors in DRAM, addressing GH #11).

### Patch 6 — Block-reuse cache for `LegalOpLayoutAnalysis`
**File:** `tt-mlir/0007-perf-TTNN-optimizer-block-reuse-cache...`
**Issue:** [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2)

Transformer models have 48 identical blocks. `LegalOpLayoutAnalysis.run()` ran independently
for each block. A structural fingerprint cache (op type + shapes + input layout attrs) lets
blocks 2–48 get cache hits without re-running the analysis.

**Expected speedup:** ~15–30% additional compile time reduction.

### Patch 7 — Async OpModel cache prefetch
**File:** `tt-mlir/0008-perf-TTNN-optimizer-async-OpModel-prefetch...`
**Issue:** [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2)

`DFShardingPolicy` for block N is CPU-bound (no device queries). `TTNNOpsModelCache` queries
for block N+1 can be prefetched in a background thread while block N's DFSharding runs.

**Changes:** `TTNNOpsModelCache.h` (thread-safe API), `TTNNOpModel.cpp` (prefetch worker),
`Optimizer.cpp` (async launch + await before block N+1 starts).

**Expected speedup:** 1.3×–1.8× additional reduction in compile time.

---

## Root-Level Patch Files (Legacy / Reference)

The `*.patch` files at the root of `openclaw-patches/` are earlier individual patch snapshots.
They overlap with the `tt-mlir/` format-patches above. **Prefer the `tt-mlir/` format-patches**
for applying (they are organized, numbered, and include commit messages).

The two tt-xla patches (`fix-oom-silent-hang-on-submit-failure.patch` and
`implement-deserialize-and-load.patch`) are unique to the root level and correspond to PR #12.

---

## Applying Everything from Scratch (Clean Container)

```bash
# 1. Set up tt-xla on the right branch (PR #12 changes)
cd /workspace/tt-xla
git checkout fix/oom-silent-hang   # or merge PR #12 when it lands

# 2. Set up tt-mlir submodule on the patch branch
cd third_party/tt-mlir/src/tt-mlir
git checkout fix/bf16-matmul-padop-precision
# (branch already exists if you cloned from mochi-bringup with correct submodule)

# 3. Rebuild tt-mlir
cd /workspace/tt-xla/third_party/tt-mlir/src/tt-mlir
cmake --build build --target all -j$(nproc)

# 4. Rebuild tt-xla
cd /workspace/tt-xla
cmake --build build --target all -j$(nproc)
# OR use the existing build scripts

# 5. Verify patches
cd /workspace/tt-xla
python patches/verify_patches.py
```

---

## Verify Patches

```bash
python /workspace/tt-xla/patches/verify_patches.py
```

Tests:
1. **bf16 matmul precision** — cosine similarity >0.999 for K=8192
2. **RoPE strided scatter** — `out[..., 0::2] = val` matches CPU reference
3. **Reshape segfault guard** — `[1, 30720]→[1, 6, 5120]` in TILE layout doesn't crash

---

## Performance Expectations (Wan2.1 10B, 4× Blackhole, SPMD TP)

| Phase | Before patches | After patches |
|-------|---------------|---------------|
| First compile | ~93 min | ~93 min (unavoidable — cache miss) |
| Subsequent compiles | ~93 min | **~0 s** (DeserializeAndLoad cache hit) |
| Per-block compile time | baseline | ~40–60% faster (patches 3, 6, 7) |
| Max sequence length | 1184 tokens | Limited by DRAM budget check (compile-time OOM error instead of silent hang) |
| bf16 raw matmul | catastrophic error at K>4096 | ✅ fixed |
| RoPE (strided slice) | wrong results | ✅ fixed |

---

## Links

| Issue | Topic |
|-------|-------|
| [GH #1](https://github.com/nvukobratTT/tt-xla/issues/1) | bf16 matmul precision (packer_l1_acc) |
| [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2) | Reduce tt-mlir compile time |
| [GH #3](https://github.com/nvukobratTT/tt-xla/issues/3) | Wan2.1-T2V integration tracking |
| [GH #4](https://github.com/nvukobratTT/tt-xla/issues/4) | ReshapeViewOperation segfault |
| [GH #5](https://github.com/nvukobratTT/tt-xla/issues/5) | Strided slice assignment (RoPE) |
| [GH #11](https://github.com/nvukobratTT/tt-xla/issues/11) | Device DRAM OOM >1184 tokens |
| [PR #12](https://github.com/nvukobratTT/tt-xla/pull/12) | tt-xla fix: OOM hang + DeserializeAndLoad |
