# Compilation Cache Research: Avoiding 40-min Recompilation for Wan2.1-T2V

**Date:** 2026-03-19
**Context:** Wan2.1-T2V-14B on Tenstorrent Blackhole via torch_xla + TT PJRT plugin.
Each of the 40 transformer blocks takes ~60s to compile (StableHLO -> TTIR -> TTNN -> flatbuffer),
totaling ~40 minutes per run. This document explores how to cache or skip that compilation.

---

## TL;DR

**The persistent compilation cache (`xr.initialize_cache()`) is currently write-only for the TT backend.**
Serialization works (PJRT `Serialize` is implemented), but deserialization is **stubbed out**
([tt-xla#498](https://github.com/tenstorrent/tt-xla/issues/498)). This means the cache directory
gets populated on first run, but subsequent runs cannot load from it — they recompile from scratch.

**Recommended path forward:** Implement `PJRT_Executable_DeserializeAndLoad` in the TT PJRT plugin.
The serialization format and infrastructure already exist; only the load-side is missing.

---

## 1. torch_xla Persistent Compilation Cache

### How it works

torch_xla provides a disk-based compilation cache via `xr.initialize_cache()`:

```python
import torch_xla.runtime as xr

# Must be called BEFORE any computation
xr.initialize_cache("/path/to/cache_dir")
```

**Source:** `venv/lib/python3.12/site-packages/torch_xla/runtime.py:262-276`

Under the hood, this sets two environment variables:
- `XLA_PERSISTENT_CACHE_PATH` — directory for cached executables
- `XLA_PERSISTENT_CACHE_READ_ONLY` — `'1'` for read-only, `'0'` for read-write

The cache flow:
1. When XLA compiles a graph, it computes a hash (`_get_graph_hash()`)
2. Checks in-memory cache first, then on-disk cache
3. On cache miss: compiles, then calls `PJRT_Executable_Serialize` to write to disk
4. On cache hit: calls `PJRT_Executable_DeserializeAndLoad` to reload from disk

### Additional APIs

```python
# Check how many graphs are cached in memory
xr.get_num_cached_compilation_graph()

# Clear the in-memory computation cache
xr.clear_computation_cache()
```

### Environment variable alternative

You can also set these before importing torch_xla:
```bash
export XLA_PERSISTENT_CACHE_PATH=/path/to/cache_dir
export XLA_PERSISTENT_CACHE_READ_ONLY=0
```

---

## 2. TT PJRT Plugin: Current Cache Support

### Serialization: IMPLEMENTED

The TT PJRT plugin implements `PJRT_Executable_Serialize`:

**Source:** `pjrt_implementation/src/api/executable_instance.cc:252-275`

It creates a `SerializedExecutableInstance` with the custom **TTSERv00** binary format:

```
Header (56 bytes):
  Magic:  "TTSERv00"           (8 bytes)
  TTIR:   offset + size        (16 bytes)
  TTNN:   offset + size        (16 bytes)
  Flatbuffer: offset + size    (16 bytes)

Body (variable):
  TTIR MLIR text
  TTNN MLIR text
  Flatbuffer binary (.ttnn compiled ops)
```

**Source:** `pjrt_implementation/inc/api/serialized_executable_instance.h:30-72`

### Deserialization: NOT IMPLEMENTED (STUBBED)

```c
// pjrt_implementation/src/stubs.inc:82-88
// Deserialization of our executable is not yet supported.
// https://github.com/tenstorrent/tt-xla/issues/498
_STUB(PJRT_Executable_DeserializeAndLoad);
```

**This is the blocker.** The cache files get written to disk, but torch_xla cannot load them back.
On the next run, it sees a cache miss (because deserialization returns "unimplemented") and recompiles.

### Program cache (runtime-level, in-memory only)

The TT runtime has a program cache controlled by `TT_RUNTIME_ENABLE_PROGRAM_CACHE`:

```cpp
// pjrt_implementation/src/api/client_instance.cc:555-562
// Enabled via TT_RUNTIME_ENABLE_PROGRAM_CACHE env var
```

This is an **in-memory** cache for the current process — it avoids recompilation of the same graph
within a single run (e.g., across denoising steps) but does NOT persist across runs.

---

## 3. Existing Serialization Tools

The codebase already has Python tools to extract compiled artifacts from the cache:

### PyTorch serialization tools

**Source:** `python_package/tt_torch/serialization.py`

```python
from tt_torch import (
    parse_compiled_artifacts_from_cache,
    parse_compiled_artifacts_from_cache_to_disk,
    save_system_descriptor_to_disk,
)

# Extract (ttir_mlir, ttnn_mlir, flatbuffer_binary) from cache
ttir, ttnn, fb = parse_compiled_artifacts_from_cache("/path/to/cachedir")

# Or save directly to disk files
parse_compiled_artifacts_from_cache_to_disk("/path/to/cachedir", "output/model")
# Creates: output/model_ttir.mlir, output/model_ttnn.mlir, output/model.ttnn

save_system_descriptor_to_disk("output/model")
# Creates: output/model_system_desc.ttsys
```

### Working example

**Source:** `examples/pytorch/serialization_example.py`

```python
cache_dir = f"{os.getcwd()}/cachedir"
xr.initialize_cache(cache_dir)

device = xm.xla_device()
model = SimpleModel().to(device)
x = torch.randn(3, 4).to(device)
y = torch.randn(3, 4).to(device)
output = model(x, y)
output.to("cpu")

parse_compiled_artifacts_from_cache_to_disk(cache_dir, "output/model")
save_system_descriptor_to_disk("output/model")
```

### Low-level parsing

**Source:** `python_package/ttxla_tools/serialization.py:11-93`

The `parse_executable()` function reads the TTSERv00 format and extracts the three components.

---

## 4. StableHLO Export Path

torch_xla can export StableHLO graphs (the input to the TT compiler):

```python
from torch_xla.stablehlo import save_as_stablehlo, save_torch_model_as_stablehlo

# From a torch.export.ExportedProgram
exported = torch.export.export(model, example_inputs)
save_as_stablehlo(exported, "stablehlo_dir/")

# Or directly from nn.Module
save_torch_model_as_stablehlo(model, (x, y), "stablehlo_dir/")
```

**Source:** `venv/lib/python3.12/site-packages/torch_xla/stablehlo.py`

**Limitation:** This saves the *input* to the compiler (StableHLO), not the *output*
(flatbuffer binary). You'd still need to run the full TTIR->TTNN->flatbuffer compilation.
However, this could be useful for an offline AOT compilation workflow.

---

## 5. Compiler Options: Export IR Artifacts

The TT compiler can dump intermediate representations via custom compile options:

```python
import torch_xla

torch_xla.set_custom_compile_options({
    "export_path": "ir_export/",           # Directory for IR dumps
    "export_model_name": "wan_block",      # Base name for files
    "optimization_level": 1,               # 0-3
})
```

**Source:** `pjrt_implementation/inc/api/compile_options.h:124-128`

This writes `ir_export/fb_*.ttnn` (flatbuffer), plus TTIR/TTNN MLIR dumps during compilation.
Useful for debugging but not for reloading compiled artifacts.

---

## 6. What Would It Take to Enable Persistent Caching

### Option A: Implement `PJRT_Executable_DeserializeAndLoad` (Recommended)

This is the right fix. The infrastructure is almost complete:

1. **Serialization format exists** — TTSERv00 with TTIR + TTNN + flatbuffer
2. **Serialization works** — `PJRT_Executable_Serialize` is implemented
3. **Python tools exist** — `parse_executable()` can read the format
4. **Missing piece:** `PJRT_Executable_DeserializeAndLoad` in `pjrt_implementation/src/stubs.inc`

**What the implementation needs to do:**
1. Read the TTSERv00 payload from the provided bytes
2. Extract the flatbuffer binary
3. Create a `FlatbufferExecutableImage` from the binary (skip TTIR/TTNN since compilation is done)
4. Create a `LoadedExecutableInstance` from the executable image
5. Return it via the PJRT API

**Tracked at:** https://github.com/tenstorrent/tt-xla/issues/498

**Effort estimate:** The serialization format and executable image classes already exist.
The main work is wiring up the deserialization path in `client_instance.cc`.

### Option B: Manual flatbuffer reload (workaround)

If deserialization is implemented for the runtime but not PJRT API, you could potentially:
1. Run once with `xr.initialize_cache()` + `parse_compiled_artifacts_from_cache_to_disk()`
2. Save the `.ttnn` flatbuffer files
3. On subsequent runs, load flatbuffers directly via `tt::runtime` APIs

**Status:** Not currently exposed as a Python API. Would require custom C++ or Python bindings.

### Option C: StableHLO AOT compilation (partial solution)

1. Export StableHLO graphs via `save_as_stablehlo()` or `get_stablehlo_bytecode()`
2. Compile offline using tt-mlir tools (e.g., `ttmlir-opt` + `ttnn-to-flatbuffer`)
3. Load flatbuffers at runtime

**Limitation:** Requires building a custom offline compilation pipeline.
The tt-mlir toolchain (`$TTMLIR_TOOLCHAIN_DIR`) has the tools, but this workflow
isn't documented or exposed as a user-facing path.

---

## 7. Applying to Wan2.1-T2V Pipeline

### Current state of wan_t2v_tp.py

The pipeline in `examples/pytorch/wan_t2v_tp.py`:
- Uses `torch_xla.set_custom_compile_options({"optimization_level": optimization_level})`
- Does NOT call `xr.initialize_cache()` — no cache is used today
- Uses `xm.mark_step()` between blocks for incremental compilation
- Each block compiles independently (~60s each)

### What you can do today

**1. Enable the persistent cache (write-only for now):**
```python
import torch_xla.runtime as xr

# Add this BEFORE xr.set_device_type("TT")
cache_dir = "/path/to/wan_cache"
xr.initialize_cache(cache_dir)
```

This will populate the cache directory with serialized executables. Even though deserialization
isn't implemented yet, having the cache ready means you'll benefit immediately once
`DeserializeAndLoad` is implemented.

**2. Enable the runtime program cache:**
```bash
export TT_RUNTIME_ENABLE_PROGRAM_CACHE=1
```

This helps within a single run — if the same graph shape appears in multiple denoising steps,
it won't recompile. For Wan2.1, all 50 denoising steps use the same transformer graph shapes,
so after the first step compiles all 40 blocks, subsequent steps should hit the in-memory cache.

**3. Export compiled artifacts for inspection:**
```python
torch_xla.set_custom_compile_options({
    "export_path": "wan_compiled/",
    "export_model_name": "wan_block",
    "optimization_level": 1,
})
```

### Proposed code change for wan_t2v_tp.py

```python
def run_wan_tp_pipeline(
    prompt, ..., cache_dir=None, optimization_level=1,
):
    # Enable persistent cache if path provided
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        xr.initialize_cache(cache_dir)

    # Enable runtime program cache for intra-run reuse
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

    torch_xla.set_custom_compile_options({"optimization_level": optimization_level})
    # ... rest of pipeline
```

---

## 8. Summary and Recommendations

| Approach | Status | Benefit | Effort |
|----------|--------|---------|--------|
| `xr.initialize_cache()` | **Partially works** (serialize yes, deserialize no) | Full reuse across runs once deserialize is implemented | Already exists, need deserialize |
| `TT_RUNTIME_ENABLE_PROGRAM_CACHE` | **Works** | In-memory cache within a single run | Zero effort — set env var |
| Implement `DeserializeAndLoad` | **Not implemented** ([#498](https://github.com/tenstorrent/tt-xla/issues/498)) | Complete persistent caching | Medium — format/infra exists |
| StableHLO export + offline compile | **Possible but manual** | Full AOT workflow | High — no integrated path |
| Manual flatbuffer reload | **Not exposed** | Direct binary reuse | High — needs new Python bindings |

### Priority actions

1. **Immediate:** Set `TT_RUNTIME_ENABLE_PROGRAM_CACHE=1` to avoid recompilation within a run
2. **Immediate:** Add `xr.initialize_cache()` to populate cache for future use
3. **Short-term:** Implement `PJRT_Executable_DeserializeAndLoad` (issue #498) — this is the
   highest-impact change and the infrastructure is 80% there
4. **Medium-term:** Consider an AOT compilation CLI tool that takes StableHLO and produces
   `.ttnn` flatbuffers, for headless CI/CD compilation

---

## 9. Key Source Files Reference

| File | What it does |
|------|-------------|
| `venv/.../torch_xla/runtime.py:262-276` | `initialize_cache()` API |
| `pjrt_implementation/src/api/executable_instance.cc:252-275` | `PJRT_Executable_Serialize` implementation |
| `pjrt_implementation/inc/api/serialized_executable_instance.h` | TTSERv00 format definition |
| `pjrt_implementation/src/api/serialized_executable_instance.cc` | Serialization logic |
| `pjrt_implementation/src/stubs.inc:82-84` | `DeserializeAndLoad` stub (THE BLOCKER) |
| `python_package/tt_torch/serialization.py` | Python cache extraction tools |
| `python_package/ttxla_tools/serialization.py` | Low-level TTSERv00 parser |
| `examples/pytorch/serialization_example.py` | Working serialization example |
| `examples/pytorch/wan_t2v_tp.py` | Wan2.1 TP pipeline (no cache today) |
| `pjrt_implementation/src/api/client_instance.cc:555-562` | `TT_RUNTIME_ENABLE_PROGRAM_CACHE` env var |
| `pjrt_implementation/src/api/module_builder/module_builder.cc` | Full compilation pipeline (VHLO->SHLO->TTIR->TTNN->FB) |
