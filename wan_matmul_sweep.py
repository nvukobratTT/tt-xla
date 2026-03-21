"""
Sweep matmul reduction dimension to find where TT precision breaks down.
Also test tiling strategies (split large K into chunks).
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

tt_device = xm.xla_device()

def compare(name, tt_out, cpu_out):
    a = tt_out.float().flatten()
    b = cpu_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    maxd = (a - b).abs().max().item()
    meand = (a - b).abs().mean().item()
    pct = ((a - b).abs() > 1.0).float().mean().item() * 100
    print(f"  {name}: cos={cos:.6f} max={maxd:.4f} mean={meand:.6f} >1.0={pct:.1f}%")
    return cos, maxd

# ========== Part 1: Sweep reduction dimension ==========
print("=" * 70)
print("Part 1: MatMul precision vs reduction dimension K")
print("  Fixed M=768, N=1536, varying K")
print("=" * 70)

torch.manual_seed(42)
M, N = 768, 1536

for K in [128, 256, 512, 1024, 1536, 2048, 4096, 4480, 8960]:
    A = torch.randn(1, M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    cpu_out = A @ B

    A_tt = A.to(tt_device)
    B_tt = B.to(tt_device)
    xm.mark_step()
    tt_out = (A_tt @ B_tt)
    xm.mark_step()
    tt_cpu = tt_out.cpu()
    compare(f"K={K:5d}", tt_cpu, cpu_out)

# ========== Part 2: Tiled matmul for FFN down ==========
print("\n" + "=" * 70)
print("Part 2: Tiled matmul for FFN down (M=768, K=8960, N=1536)")
print("  Split K dimension into chunks, accumulate on CPU in f32")
print("=" * 70)

torch.manual_seed(42)
M, K, N = 768, 8960, 1536
A = torch.randn(1, M, K, dtype=torch.bfloat16)
B = torch.randn(K, N, dtype=torch.bfloat16)
cpu_out = A @ B

# Full TT
A_tt = A.to(tt_device)
B_tt = B.to(tt_device)
xm.mark_step()
tt_full = (A_tt @ B_tt)
xm.mark_step()
compare("Full TT (no tile)", tt_full.cpu(), cpu_out)

# Tiled: split K into chunks
for chunk_K in [256, 512, 1024, 1792, 4480]:
    num_chunks = K // chunk_K
    if K % chunk_K != 0:
        continue

    accum = torch.zeros(1, M, N, dtype=torch.float32)
    for i in range(num_chunks):
        start = i * chunk_K
        end = start + chunk_K
        A_chunk = A[:, :, start:end].to(tt_device)
        B_chunk = B[start:end, :].to(tt_device)
        xm.mark_step()
        partial = (A_chunk @ B_chunk)
        xm.mark_step()
        accum += partial.cpu().float()

    tiled_out = accum.to(torch.bfloat16)
    compare(f"Tiled K={chunk_K:4d} ({num_chunks} chunks)", tiled_out, cpu_out)

# ========== Part 3: Tiled on TT entirely (accumulate on device) ==========
print("\n" + "=" * 70)
print("Part 3: Tiled matmul accumulating on TT device (f32 accumulation)")
print("=" * 70)

for chunk_K in [1024, 1792, 4480]:
    num_chunks = K // chunk_K
    if K % chunk_K != 0:
        continue

    accum_tt = torch.zeros(1, M, N, dtype=torch.float32, device=tt_device)
    xm.mark_step()
    for i in range(num_chunks):
        start = i * chunk_K
        end = start + chunk_K
        A_chunk = A[:, :, start:end].to(tt_device)
        B_chunk = B[start:end, :].to(tt_device)
        partial = (A_chunk @ B_chunk).float()
        accum_tt = accum_tt + partial
        xm.mark_step()

    tiled_tt = accum_tt.to(torch.bfloat16)
    xm.mark_step()
    compare(f"TT-tiled K={chunk_K:4d} ({num_chunks} chunks, f32 acc)", tiled_tt.cpu(), cpu_out)

# ========== Part 4: The real FFN pattern ==========
print("\n" + "=" * 70)
print("Part 4: Realistic FFN (up + SiLU + gate + down)")
print("=" * 70)

torch.manual_seed(42)
x = torch.randn(1, 768, 1536, dtype=torch.bfloat16)
W_up = torch.randn(1536, 8960, dtype=torch.bfloat16)
W_gate = torch.randn(1536, 8960, dtype=torch.bfloat16)
W_down = torch.randn(8960, 1536, dtype=torch.bfloat16)

# CPU reference
up = x @ W_up
gate = x @ W_gate
hidden = torch.nn.functional.silu(gate) * up
cpu_ffn = hidden @ W_down

# TT
x_tt = x.to(tt_device)
Wu_tt = W_up.to(tt_device)
Wg_tt = W_gate.to(tt_device)
Wd_tt = W_down.to(tt_device)
xm.mark_step()

up_tt = x_tt @ Wu_tt
gate_tt = x_tt @ Wg_tt
hidden_tt = torch.nn.functional.silu(gate_tt) * up_tt
tt_ffn = hidden_tt @ Wd_tt
xm.mark_step()
compare("FFN on TT (full)", tt_ffn.cpu(), cpu_ffn)

# TT with tiled down projection
xm.mark_step()
up_tt2 = x_tt @ Wu_tt
gate_tt2 = x_tt @ Wg_tt
hidden_tt2 = torch.nn.functional.silu(gate_tt2) * up_tt2
xm.mark_step()

# Tile the down proj: 8960 = 1792 * 5
hidden_cpu = hidden_tt2.cpu()
accum = torch.zeros(1, 768, 1536, dtype=torch.float32)
chunk_K = 1792
for i in range(5):
    s, e = i * chunk_K, (i+1) * chunk_K
    h_chunk = hidden_cpu[:, :, s:e].to(tt_device)
    w_chunk = W_down[s:e, :].to(tt_device)
    xm.mark_step()
    partial = (h_chunk @ w_chunk)
    xm.mark_step()
    accum += partial.cpu().float()

tiled_ffn = accum.to(torch.bfloat16)
compare("FFN on TT (tiled down proj, cpu accum)", tiled_ffn, cpu_ffn)

# Full CPU FFN down proj with TT up/gate
hidden_cpu_f = hidden_cpu.float()
W_down_f = W_down.float()
cpu_down = (hidden_cpu_f @ W_down_f).to(torch.bfloat16)
compare("FFN TT up/gate + CPU down proj", cpu_down, cpu_ffn)

print("\nDONE")
