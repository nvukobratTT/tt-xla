"""
Minimal repro: torch.empty_like() + strided slice assignment produces incorrect
results on TT/XLA. The pattern out[..., 0::2] / out[..., 1::2] does not lower
correctly through StableHLO -> TTIR.

This pattern is used in RoPE (Rotary Position Embeddings) in diffusers and
many other models.

Expected: Identical output to CPU.
Actual: ~250x error amplification, 93% of elements off by >0.1.

Hardware: Tenstorrent Blackhole (4-chip mesh)
"""
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.set_device_type("TT")
device = xm.xla_device()
torch.manual_seed(42)

B, S, H, D = 1, 768, 12, 128  # batch, seq, heads, head_dim

hidden = torch.randn(B, S, H, D, dtype=torch.bfloat16)
cos = torch.randn(B, S, 1, D, dtype=torch.bfloat16)
sin = torch.randn(B, S, 1, D, dtype=torch.bfloat16)


def rope_broken(hidden_states, freqs_cos, freqs_sin):
    """Original pattern (broken on TT): empty_like + strided scatter."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def rope_fixed(hidden_states, freqs_cos, freqs_sin):
    """Fixed pattern: stack + flatten (correct on TT)."""
    x = hidden_states.unflatten(-1, (-1, 2))
    x1, x2 = x[..., 0], x[..., 1]
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2).type_as(hidden_states)


# CPU reference (both implementations are correct on CPU)
cpu_ref = rope_broken(hidden, cos, sin)
cpu_fixed = rope_fixed(hidden, cos, sin)
assert (cpu_ref - cpu_fixed).abs().max().item() == 0.0, "Both should match on CPU"

# TT: broken pattern
tt_broken = rope_broken(hidden.to(device), cos.to(device), sin.to(device))
xm.mark_step()

# TT: fixed pattern
tt_fixed = rope_fixed(hidden.to(device), cos.to(device), sin.to(device))
xm.mark_step()


def report(name, tt_out, cpu_ref):
    diff = (tt_out.cpu().float() - cpu_ref.float()).abs()
    pct_01 = (diff > 0.1).float().mean().item() * 100
    pct_10 = (diff > 1.0).float().mean().item() * 100
    print(f"{name}:")
    print(f"  max_diff={diff.max().item():.4f}, mean_diff={diff.mean().item():.6f}")
    print(f"  % elements > 0.1: {pct_01:.1f}%")
    print(f"  % elements > 1.0: {pct_10:.1f}%")


print("=== RoPE Strided Scatter Bug ===\n")
report("Broken (empty_like + out[..., 0::2])", tt_broken, cpu_ref)
print()
report("Fixed  (stack + flatten)", tt_fixed, cpu_ref)
print()
print("Workaround: Replace empty_like + strided scatter with torch.stack + flatten.")
