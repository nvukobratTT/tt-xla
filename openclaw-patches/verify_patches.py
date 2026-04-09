#!/usr/bin/env python3
"""
verify_patches.py — Patch Verification Test Suite for Wan2.1-T2V on Blackhole
==============================================================================
Run this BEFORE the full Wan2.1 pipeline to confirm that three patches applied
correctly. Each test is cheap (~seconds once compiled, no full model load).

Patches tested:
  Test 1: GH #1 — bf16 matmul precision (packer_l1_acc propagation)
  Test 2: GH #5 — strided slice assignment / RoPE pattern (PadOp row-major fix)
  Test 3: GH #4 — ReshapeViewOperation segfault guard (condition_embedder)

Usage:
    python3 verify_patches.py              # all 3 tests
    python3 verify_patches.py --test 1     # just matmul precision
    python3 verify_patches.py --test 2     # just strided scatter
    python3 verify_patches.py --test 3     # just reshape segfault guard
"""

import argparse
import sys
import traceback

COSINE_SIM_THRESHOLD = 0.999
ABS_TOL_SCATTER = 1e-3


def cosine_sim(a, b):
    import torch
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def result_line(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    pad = max(0, 60 - len(name))
    line = f"  [{name}] {'.' * pad} {status}"
    if detail:
        line += f"  [{detail}]"
    print(line)
    return passed


# ── TEST 1 — bf16 large-K matmul precision (GH #1) ───────────────────────────
def test_bf16_large_k_matmul():
    name = "TEST 1 — bf16 large-K matmul precision (GH #1)"
    try:
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        M, K, N = 32, 8192, 64
        torch.manual_seed(42)
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        ref = torch.matmul(A, B)  # fp32 reference on CPU

        out = torch.matmul(A.bfloat16().to(device), B.bfloat16().to(device))
        xm.mark_step()

        sim = cosine_sim(out.float().cpu(), ref)
        passed = sim >= COSINE_SIM_THRESHOLD
        return result_line(name, passed,
                           f"cosine_sim={sim:.6f}, threshold={COSINE_SIM_THRESHOLD}")
    except Exception:
        print(f"  [{name}] ... ERROR")
        traceback.print_exc()
        return False


# ── TEST 2 — strided slice assignment / RoPE pattern (GH #5) ─────────────────
def test_strided_slice_assignment():
    name = "TEST 2 — strided slice assignment / RoPE pattern (GH #5)"
    try:
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        B, H, S, D = 1, 24, 64, 128
        torch.manual_seed(7)
        x = torch.zeros(B, H, S, D)
        cos = torch.randn(B, H, S, D // 2)
        sin = torch.randn(B, H, S, D // 2)

        # CPU reference
        ref = x.clone()
        ref[..., 0::2] = cos
        ref[..., 1::2] = sin

        # On device
        x_tt = x.to(device)
        cos_tt = cos.to(device)
        sin_tt = sin.to(device)
        x_tt[..., 0::2] = cos_tt
        x_tt[..., 1::2] = sin_tt
        xm.mark_step()

        err = (x_tt.cpu() - ref).abs().max().item()
        passed = err <= ABS_TOL_SCATTER
        return result_line(name, passed,
                           f"max_abs_err={err:.2e}, tol={ABS_TOL_SCATTER:.1e}")
    except Exception:
        print(f"  [{name}] ... ERROR")
        traceback.print_exc()
        return False


# ── TEST 3 — ReshapeViewOperation segfault / condition_embedder (GH #4) ──────
def test_reshape_segfault():
    name = "TEST 3 — ReshapeViewOp segfault guard / condition_embedder (GH #4)"
    try:
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        # Exact shape that triggers the padded-volume mismatch bug:
        # [1, 30720] → [1, 6, 5120]: mid-dims 1 and 6 both < tile_height(32),
        # both get padded to 32 → padded volumes diverge 6:1 → segfault.
        x = torch.randn(1, 30720, dtype=torch.bfloat16).to(device)
        out = x.unflatten(1, (6, 5120))
        xm.mark_step()

        out_cpu = out.cpu()
        shape_ok = tuple(out_cpu.shape) == (1, 6, 5120)
        values_ok = torch.allclose(
            out_cpu.float(),
            x.cpu().float().view(1, 6, 5120),
            atol=1e-3
        )
        passed = shape_ok and values_ok
        return result_line(name, passed,
                           f"shape={'ok' if shape_ok else 'WRONG'}, "
                           f"values={'ok' if values_ok else 'WRONG'}")
    except Exception:
        print(f"  [{name}] ... ERROR")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify Wan2.1-T2V compiler patches on Blackhole hardware."
    )
    parser.add_argument(
        "--test",
        type=int,
        choices=[1, 2, 3],
        help="Run only the specified test (1=matmul, 2=strided-scatter, 3=reshape)",
    )
    args = parser.parse_args()

    tests = {
        1: test_bf16_large_k_matmul,
        2: test_strided_slice_assignment,
        3: test_reshape_segfault,
    }
    run = {args.test: tests[args.test]} if args.test else tests

    print("\nWan2.1-T2V patch verification suite")
    print("=" * 70)
    results = {k: fn() for k, fn in sorted(run.items())}
    print("=" * 70)

    n_pass = sum(results.values())
    n_total = len(results)
    if n_pass == n_total:
        print(f"All {n_total} test(s) passed. Safe to run full Wan2.1 pipeline.\n")
        sys.exit(0)
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"{n_pass}/{n_total} passed. FAILED tests: {failed}")
        print("Do NOT proceed to full pipeline until all tests pass.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
