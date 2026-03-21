"""
Dissect: test individual ops on TT vs CPU at the sizes used by Wan 1.3B.
Focus on matmul, FFN, LayerNorm, and a full block with real preprocessing.
"""

import time
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.set_device_type("TT")
torch_xla.set_custom_compile_options({
    "fp32_dest_acc_en": "true",
    "math_fidelity": "hifi4",
})


def compare(name, tt_out, cpu_out):
    a = tt_out.float().flatten()
    b = cpu_out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    maxd = (a - b).abs().max().item()
    meand = (a - b).abs().mean().item()
    pct_gt1 = ((a - b).abs() > 1.0).float().mean().item() * 100
    print(f"  {name}:")
    print(f"    cosine={cos:.6f}  max_diff={maxd:.4f}  mean_diff={meand:.6f}  >1.0: {pct_gt1:.1f}%")
    print(f"    std_tt={a.std().item():.4f}  std_cpu={b.std().item():.4f}")
    return cos, maxd


def main():
    tt_device = xm.xla_device()

    # Wan 1.3B dimensions
    seq_len = 768  # for 256x256x9
    hidden_dim = 1536
    ffn_dim = 8960
    num_heads = 12
    head_dim = 128

    torch.manual_seed(42)

    # ========== Test 1: MatMul at various sizes ==========
    print("=" * 60)
    print("Test 1: MatMul precision")
    print("=" * 60)

    for M, K, N, label in [
        (768, 1536, 1536, "Q/K/V proj"),
        (768, 1536, 4608, "QKV fused"),
        (768, 1536, 8960, "FFN up"),
        (768, 8960, 1536, "FFN down"),
        (768, 128, 768, "Attn scores (per-head equivalent)"),
    ]:
        A = torch.randn(1, M, K, dtype=torch.bfloat16)
        B = torch.randn(K, N, dtype=torch.bfloat16)
        cpu_out = A @ B

        A_tt = A.to(tt_device)
        B_tt = B.to(tt_device)
        xm.mark_step()
        tt_out = (A_tt @ B_tt)
        xm.mark_step()
        tt_cpu = tt_out.cpu()
        compare(f"MatMul {M}x{K} @ {K}x{N} ({label})", tt_cpu, cpu_out)

    # ========== Test 2: Matmul with f32 inputs ==========
    print("\n" + "=" * 60)
    print("Test 2: MatMul in f32")
    print("=" * 60)

    for M, K, N, label in [
        (768, 1536, 1536, "Q/K/V proj f32"),
        (768, 1536, 8960, "FFN up f32"),
    ]:
        A = torch.randn(1, M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        cpu_out = A @ B

        A_tt = A.to(tt_device)
        B_tt = B.to(tt_device)
        xm.mark_step()
        tt_out = (A_tt @ B_tt)
        xm.mark_step()
        tt_cpu = tt_out.cpu()
        compare(f"MatMul f32 {M}x{K} @ {K}x{N} ({label})", tt_cpu, cpu_out)

    # ========== Test 3: SDPA ==========
    print("\n" + "=" * 60)
    print("Test 3: Scaled Dot-Product Attention")
    print("=" * 60)

    Q = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    K = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    V = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    cpu_sdpa = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    Q_tt, K_tt, V_tt = Q.to(tt_device), K.to(tt_device), V.to(tt_device)
    xm.mark_step()
    tt_sdpa = torch.nn.functional.scaled_dot_product_attention(Q_tt, K_tt, V_tt)
    xm.mark_step()
    tt_sdpa_cpu = tt_sdpa.cpu()
    compare("SDPA bf16", tt_sdpa_cpu, cpu_sdpa)

    # f32 SDPA
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    cpu_sdpa_f32 = torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf)

    Qf_tt, Kf_tt, Vf_tt = Qf.to(tt_device), Kf.to(tt_device), Vf.to(tt_device)
    xm.mark_step()
    tt_sdpa_f32 = torch.nn.functional.scaled_dot_product_attention(Qf_tt, Kf_tt, Vf_tt)
    xm.mark_step()
    tt_sdpa_f32_cpu = tt_sdpa_f32.cpu()
    compare("SDPA f32", tt_sdpa_f32_cpu, cpu_sdpa_f32)

    # ========== Test 4: LayerNorm ==========
    print("\n" + "=" * 60)
    print("Test 4: LayerNorm")
    print("=" * 60)

    ln = torch.nn.LayerNorm(hidden_dim).eval()
    x = torch.randn(1, seq_len, hidden_dim, dtype=torch.float32)
    cpu_ln = ln(x)

    ln_tt = ln.to(tt_device)
    x_tt = x.to(tt_device)
    xm.mark_step()
    tt_ln = ln_tt(x_tt)
    xm.mark_step()
    tt_ln_cpu = tt_ln.cpu()
    compare("LayerNorm f32", tt_ln_cpu, cpu_ln)

    # ========== Test 5: SiLU activation ==========
    print("\n" + "=" * 60)
    print("Test 5: SiLU activation")
    print("=" * 60)

    x_silu = torch.randn(1, seq_len, ffn_dim, dtype=torch.bfloat16)
    cpu_silu = torch.nn.functional.silu(x_silu)

    x_silu_tt = x_silu.to(tt_device)
    xm.mark_step()
    tt_silu = torch.nn.functional.silu(x_silu_tt)
    xm.mark_step()
    tt_silu_cpu = tt_silu.cpu()
    compare("SiLU bf16", tt_silu_cpu, cpu_silu)

    # ========== Test 6: Element-wise multiply (gating) ==========
    print("\n" + "=" * 60)
    print("Test 6: Element-wise multiply (gating)")
    print("=" * 60)

    a = torch.randn(1, seq_len, ffn_dim, dtype=torch.bfloat16)
    b = torch.randn(1, seq_len, ffn_dim, dtype=torch.bfloat16)
    cpu_mul = a * b

    a_tt = a.to(tt_device)
    b_tt = b.to(tt_device)
    xm.mark_step()
    tt_mul = a_tt * b_tt
    xm.mark_step()
    tt_mul_cpu = tt_mul.cpu()
    compare("Element-wise multiply bf16", tt_mul_cpu, cpu_mul)

    # ========== Test 7: RoPE isolated ==========
    print("\n" + "=" * 60)
    print("Test 7: RoPE (stack+flatten version)")
    print("=" * 60)

    hs = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    freqs_cos = torch.randn(1, seq_len, 1, head_dim, dtype=torch.bfloat16)
    freqs_sin = torch.randn(1, seq_len, 1, head_dim, dtype=torch.bfloat16)

    def apply_rope(hidden_states, freqs_cos, freqs_sin):
        x = hidden_states.unflatten(-1, (-1, 2))
        x1, x2 = x[..., 0], x[..., 1]
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2).type_as(hidden_states)

    cpu_rope = apply_rope(hs, freqs_cos, freqs_sin)

    hs_tt = hs.to(tt_device)
    fc_tt = freqs_cos.to(tt_device)
    fs_tt = freqs_sin.to(tt_device)
    xm.mark_step()
    tt_rope = apply_rope(hs_tt, fc_tt, fs_tt)
    xm.mark_step()
    tt_rope_cpu = tt_rope.cpu()
    compare("RoPE (stack+flatten)", tt_rope_cpu, cpu_rope)

    # ========== Test 8: Complete attention path ==========
    print("\n" + "=" * 60)
    print("Test 8: Complete attention path (proj + RoPE + SDPA + out_proj)")
    print("=" * 60)

    # Simulate: linear proj -> head split -> RoPE -> SDPA -> concat -> out_proj
    W_q = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16)
    W_k = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16)
    W_v = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16)
    W_o = torch.randn(hidden_dim, hidden_dim, dtype=torch.bfloat16)

    x_attn = torch.randn(1, seq_len, hidden_dim, dtype=torch.bfloat16)

    def attn_path(x, W_q, W_k, W_v, W_o, freqs_cos, freqs_sin):
        q = (x @ W_q.T).unflatten(2, (num_heads, head_dim))
        k = (x @ W_k.T).unflatten(2, (num_heads, head_dim))
        v = (x @ W_v.T).unflatten(2, (num_heads, head_dim))

        # RoPE
        xq = q.unflatten(-1, (-1, 2))
        x1, x2 = xq[..., 0], xq[..., 1]
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        q_rot1 = x1 * cos - x2 * sin
        q_rot2 = x1 * sin + x2 * cos
        q = torch.stack([q_rot1, q_rot2], dim=-1).flatten(-2).type_as(x)

        xk = k.unflatten(-1, (-1, 2))
        x1, x2 = xk[..., 0], xk[..., 1]
        k_rot1 = x1 * cos - x2 * sin
        k_rot2 = x1 * sin + x2 * cos
        k = torch.stack([k_rot1, k_rot2], dim=-1).flatten(-2).type_as(x)

        # SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Out proj
        attn_out = attn_out.transpose(1, 2).flatten(2, 3)
        return attn_out @ W_o.T

    cpu_attn = attn_path(x_attn, W_q, W_k, W_v, W_o, freqs_cos, freqs_sin)

    x_tt = x_attn.to(tt_device)
    Wq_tt = W_q.to(tt_device)
    Wk_tt = W_k.to(tt_device)
    Wv_tt = W_v.to(tt_device)
    Wo_tt = W_o.to(tt_device)
    fc_tt2 = freqs_cos.to(tt_device)
    fs_tt2 = freqs_sin.to(tt_device)
    xm.mark_step()
    tt_attn = attn_path(x_tt, Wq_tt, Wk_tt, Wv_tt, Wo_tt, fc_tt2, fs_tt2)
    xm.mark_step()
    tt_attn_cpu = tt_attn.cpu()
    compare("Full attention path", tt_attn_cpu, cpu_attn)

    # ========== Test 9: Residual connection impact ==========
    print("\n" + "=" * 60)
    print("Test 9: Residual connection (x + attn_out)")
    print("=" * 60)
    # How does small error in attn_out affect residual?
    residual = torch.randn(1, seq_len, hidden_dim, dtype=torch.bfloat16)
    noise = torch.randn_like(residual) * 0.1  # small perturbation
    cpu_res = residual + noise
    cpu_res2 = residual + noise * 1.5  # TT-like error

    cos_clean = torch.nn.functional.cosine_similarity(
        (residual + noise).float().flatten().unsqueeze(0),
        residual.float().flatten().unsqueeze(0)
    ).item()
    cos_noisy = torch.nn.functional.cosine_similarity(
        (residual + noise * 1.5).float().flatten().unsqueeze(0),
        residual.float().flatten().unsqueeze(0)
    ).item()
    print(f"  Residual + 0.1*noise cosine to residual: {cos_clean:.6f}")
    print(f"  Residual + 0.15*noise cosine to residual: {cos_noisy:.6f}")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
