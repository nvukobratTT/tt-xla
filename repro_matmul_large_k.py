"""
Minimal repro: bf16 matmul precision degrades with large reduction dimension K,
even with fp32_dest_acc_en=true and math_fidelity=hifi4.

Expected: With fp32 destination accumulation, error should stay constant regardless of K.
Actual: Error scales ~linearly with K. At K=8960, 82% of elements are off by >1.0.

This makes any model with FFN hidden_dim > ~4K produce incorrect output
(e.g., Wan2.1-T2V-1.3B has ffn_dim=8960, LLaMA has ffn_dim=11008).

Hardware: Tenstorrent Blackhole (4-chip mesh)
"""
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.set_device_type("TT")
torch_xla.set_custom_compile_options({
    "fp32_dest_acc_en": "true",
    "math_fidelity": "hifi4",
})

device = xm.xla_device()
torch.manual_seed(42)

M, N = 768, 1536  # Fixed output shape

print(f"{'K':>6} | {'max_diff':>10} | {'mean_diff':>10} | {'% >1.0':>8} | {'cosine':>8}")
print("-" * 60)

for K in [128, 256, 512, 1024, 1536, 2048, 4096, 8960]:
    A = torch.randn(1, M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)

    # CPU reference
    cpu_out = A @ B

    # TT
    tt_out = A.to(device) @ B.to(device)
    xm.mark_step()
    tt_cpu = tt_out.cpu()

    diff = (tt_cpu.float() - cpu_out.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        tt_cpu.float().flatten().unsqueeze(0),
        cpu_out.float().flatten().unsqueeze(0)
    ).item()

    print(f"{K:>6} | {diff.max().item():>10.3f} | {diff.mean().item():>10.4f} | "
          f"{(diff > 1.0).float().mean().item() * 100:>7.1f}% | {cos:>8.6f}")

# Also verify nn.Linear is fine (same weights, same shapes)
print("\n--- nn.Linear comparison (K=8960) ---")
K = 8960
linear = torch.nn.Linear(K, N, bias=False, dtype=torch.bfloat16)
x = torch.randn(1, M, K, dtype=torch.bfloat16)

with torch.no_grad():
    cpu_ref = linear(x)
    tt_ref = linear.to(device)(x.to(device))
    xm.mark_step()

diff = (tt_ref.cpu().float() - cpu_ref.float()).abs()
print(f"nn.Linear K={K}: max_diff={diff.max().item():.3f}, "
      f"mean_diff={diff.mean().item():.4f}, "
      f"% >1.0 = {(diff > 1.0).float().mean().item() * 100:.1f}%")
print("(nn.Linear may take a different code path than raw matmul)")
