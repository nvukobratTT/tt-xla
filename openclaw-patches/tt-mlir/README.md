# tt-mlir Patches

These patches apply to the  repository. Apply on top of the
commit referenced by  in this repo's  branch.

## Apply all

```bash
cd third_party/tt-mlir/src/tt-mlir
git am openclaw-patches/tt-mlir/0002-*.patch  # etc.
```

Or apply individually:

## Patches

| # | File | Issue | Description |
|---|------|-------|-------------|
| 1 | 0002-fix-TTNN-propagate... | [GH #1](https://github.com/nvukobratTT/tt-xla/issues/1) | bf16 matmul: propagate packer_l1_acc when fp32_dest_acc_en is set |
| 2 | 0003-fix-StableHLO... | [GH #5](https://github.com/nvukobratTT/tt-xla/issues/5) | PadOp interior-padding index order fix (row-major) |
| 3 | 0004-feat-TTNN-optimizer... | [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2) | maxLegalLayoutsElementwise option to reduce compile time |
| 4 | 0005-feat-TTNN-add-TTNNCheckDRAMBudget... | [GH #11](https://github.com/nvukobratTT/tt-xla/issues/11) | TTNNCheckDRAMBudget: compile-time DRAM OOM detection pass |
| 5 | 0006-fix-TTNN-exclude-SDPA... | [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2) | Exclude SDPA ops from LegalOpLayoutAnalysis optimizer target |
| 6 | 0007-perf-TTNN-optimizer-block-reuse-cache... | [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2) | Block-reuse cache for LegalOpLayoutAnalysis (15–30% speedup) |
| 7 | 0008-perf-TTNN-optimizer-async-OpModel-prefetch... | [GH #2](https://github.com/nvukobratTT/tt-xla/issues/2) | Async OpModel cache prefetch (1.3×–1.8× additional speedup) |
