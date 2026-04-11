# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan2.1-T2V-14B Text-to-Video Pipeline with Tensor Parallelism across 4 Blackhole chips.

Strategy: Megatron-style TP for the 14B transformer.
- 40 attention heads / 4 devices = 10 heads per device
- QKV projections: column-parallel (shard output dim)
- Output projection: row-parallel (shard input dim) + all-reduce
- FFN up/gate: column-parallel
- FFN down: row-parallel + all-reduce
- Pre-processing (rope, patch_embedding, condition_embedder) on CPU
- VAE and text encoder on CPU

This reduces per-device attention from (1, 40, seq, 128) to (1, 10, seq, 128),
which should bring tt-mlir compilation time from hours to minutes.
"""

import argparse
import gc
import math
import os
import time
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


def export_to_video(frames: np.ndarray, output_path: str, fps: int = 16):
    """Export frames to video, trying available backends."""
    try:
        from diffusers.utils import export_to_video as diffusers_export
        diffusers_export(list(frames), output_path, fps=fps)
        return
    except Exception:
        pass

    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return
    except ImportError:
        pass

    raise RuntimeError(
        "No video export backend available. Install imageio-ffmpeg or opencv-python."
    )




# =============================================================================
# TT SDPA Attention Processor — enables SDPA fusion in tt-mlir
# =============================================================================

def _compute_seq_len(height: int, width: int, num_frames: int) -> int:
    """Compute transformer sequence length for given resolution + frame count."""
    frames = (num_frames - 1) // 4 + 1
    h_patches = height // 16
    w_patches = width // 16
    return frames * h_patches * w_patches


def is_sdpa_compatible(height: int, width: int, num_frames: int) -> bool:
    """Return True if seq_len is divisible by 32 (required by tt.scaled_dot_product_attention)."""
    return _compute_seq_len(height, width, num_frames) % 32 == 0


class WanTTSDPAAttnProcessor:
    """
    TT-native attention processor for WanTransformerBlock.

    Replaces diffusers WanAttnProcessor to emit a stablehlo.CustomCall
    "tt.scaled_dot_product_attention" instead of decomposed matmul+softmax+matmul.
    tt-mlir fuses this into a single SDPA kernel (O(seq) memory vs O(seq^2)),
    breaking the sequence-length scaling barrier for high-resolution video.

    Shape contract:
        diffusers convention:  [B, seq, heads, head_dim]  (after unflatten)
        TTNN SDPA convention:  [B, heads, seq, head_dim]  (after transpose)
        -> transpose before/after the TT SDPA call

    Constraint: query sequence length must be divisible by 32.
    Use is_sdpa_compatible() to check before applying this processor.

    Compatible resolutions (14B, 4-chip TP):
        256x256  9f  -> seq=768   (32x24)
        256x256 49f  -> seq=3328  (32x104)
        320x512 17f  -> seq=3200  (32x100)
        480x768 17f  -> seq=7200  (32x225)
        512x832 17f  -> seq=8320  (32x260)

    Handles both self-attention (rotary_emb) and cross-attention (encoder kv).
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        from tt_torch.custom_ops import scaled_dot_product_attention as tt_sdpa
        from diffusers.models.transformers.transformer_wan import _get_qkv_projections

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # [B, seq, inner_dim] -> [B, seq, heads, head_dim]
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply RoPE (self-attention only)
        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # Transpose to TTNN SDPA shape: [B, heads, seq, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        q_seq = query.shape[2]
        assert q_seq % 32 == 0, (
            f"WanTTSDPAAttnProcessor: query seq_len={q_seq} is not divisible by 32. "
            f"Use is_sdpa_compatible(h, w, f) to check before applying. "
            f"Try 256x256/9f, 320x512/17f, 480x768/17f, or 512x832/17f."
        )

        # Fused TT SDPA: emits stablehlo.CustomCall tt.scaled_dot_product_attention
        # tt-mlir converts this to a single TTNN flash-attention kernel
        hidden_states = tt_sdpa(query, key, value, is_causal=False)

        # [B, heads, seq, head_dim] -> [B, seq, heads*head_dim]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def apply_tt_sdpa_to_transformer(transformer, height: int, width: int, num_frames: int):
    """
    Replace WanAttnProcessor with WanTTSDPAAttnProcessor on all blocks.

    Call AFTER apply_tp_sharding_wan_transformer(): SDPA processor works on
    per-device tensors (10 heads/chip for 4-chip TP) and the attention shapes
    must reflect the post-sharding head count.

    Returns True if replacement succeeded, False if seq_len not 32-aligned.
    """
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor

    seq_len = _compute_seq_len(height, width, num_frames)
    if seq_len % 32 != 0:
        print(f"  WARNING: TT SDPA skipped: seq_len={seq_len} not divisible by 32.")
        print(f"    Compatible: 256x256/9f (768), 320x512/17f (3200), 480x768/17f (7200), 512x832/17f (8320)")
        return False

    tt_proc = WanTTSDPAAttnProcessor()
    n_replaced = 0
    for block in transformer.blocks:
        if isinstance(block.attn1.processor, WanAttnProcessor):
            block.attn1.set_processor(tt_proc)
            n_replaced += 1
        if isinstance(block.attn2.processor, WanAttnProcessor):
            block.attn2.set_processor(tt_proc)
            n_replaced += 1

    print(f"  TT SDPA: replaced {n_replaced} attention processors")
    print(f"    seq={seq_len} ({seq_len//32}x32), O(seq) attention vs O(seq^2) with standard SDPA")
    print(f"    -> emits tt.scaled_dot_product_attention CustomCall -> tt-mlir SDPA kernel")
    return True




# =============================================================================
# Chunked SDPA helper (reduces peak DRAM for large frame counts)
# =============================================================================

def chunked_tt_sdpa(query, key, value, chunk_size: int = 0, is_causal: bool = False):
    """
    Chunked SDPA: splits the query sequence into chunks to bound peak DRAM.

    Peak SDPA tensor becomes [B, heads, chunk_size, seq_k] instead of the full
    [B, heads, seq_q, seq_k], enabling larger frame counts at 480×832.

    Memory math (480×832 Ulysses SP):
      - 77f (seq=31200): full SDPA = 31200^2 * 10 * 2B = 19.47 GB -> OOM
      - chunk_size=8192:  peak  = 8192  * 31200 * 10 * 2B =  4.59 GB  OK

    chunk_size=0 (default): no chunking - full SDPA (original behaviour).

    IMPORTANT: xm.mark_step() is called between chunks to force sequential
    execution on device. Without this, XLA compiles all N chunks into a single
    graph and the TT runtime allocates all N x QK^T intermediates simultaneously,
    negating the memory benefit of chunking entirely.
    """
    from tt_torch.custom_ops import scaled_dot_product_attention as tt_sdpa

    seq_q = query.shape[2]
    seq_k = key.shape[2]
    # Skip chunking if: disabled, fits in one chunk, or K/V seq is tiny
    # (cross-attention has seq_k=226 - no benefit to chunking)
    # Asymmetric Q/K shapes crash in tt.scaled_dot_product_attention (Error code 13 at
    # runtime or 20+ min hang in tt-mlir compilation). Use decomposed fallback.
    # Also pad Q to 32-aligned since TTNN matmul requires tile-aligned dimensions.
    if seq_q != seq_k:
        import math as _math
        import torch.nn.functional as _F
        # Pad Q to 32-aligned (TTNN tile requirement for matmul kernels)
        pad_q = (32 - seq_q % 32) % 32
        q_padded = _F.pad(query, (0, 0, 0, pad_q)) if pad_q > 0 else query
        _scale = 1.0 / _math.sqrt(q_padded.shape[-1])
        scores = torch.matmul(q_padded, key.transpose(-2, -1)) * _scale
        weights = torch.nn.functional.softmax(scores.float(), dim=-1).to(query.dtype)
        out = torch.matmul(weights, value)
        if pad_q > 0:
            out = out[:, :, :seq_q, :]  # trim padded Q tokens
        print(f"  [chunked_tt_sdpa] asymmetric fallback: Q_seq={seq_q} (padded+{pad_q}), K_seq={seq_k}")
        return out
    # TTNN streaming compute (Flash Attention 2) requires q_chunk_size>=64 to activate.
    # This parameter is not configurable via the tt_sdpa Python API; without it,
    # TTNN allocates the full O(seq^2) QK^T buffer which is 21.47 GB for seq=32768
    # and causes OOM on Blackhole (only ~21 GB free after model weights).
    # Workaround: auto-enable Python-level chunking for large symmetric SDPA.
    # Each chunk: [1,H,chunk_size,Sk] bf16 = 335 MB/chip (4096 chunks of seq=32768).
    _AUTO_CHUNK_THRESHOLD = 4096
    _AUTO_CHUNK_SIZE = 4096  # tile-aligned (4096 % 32 == 0), 335 MB/chip per chunk
    if chunk_size <= 0 and seq_q > _AUTO_CHUNK_THRESHOLD:
        chunk_size = _AUTO_CHUNK_SIZE
        print(f"  [chunked_tt_sdpa] auto-chunking: seq={seq_q} > {_AUTO_CHUNK_THRESHOLD}, chunk_size={chunk_size}")
    if chunk_size <= 0 or seq_q <= chunk_size or seq_k <= chunk_size:
        return tt_sdpa(query, key, value, is_causal=is_causal)

    # For chunked SDPA, use decomposed matmul+softmax+matmul to avoid
    # the slow TTNN SDPA CustomCall compilation for asymmetric shapes.
    # tt.scaled_dot_product_attention with Q_len != K_len (e.g., [8192,32768])
    # takes 20+ min to compile in tt-mlir; plain XLA HLO matmuls compile in ~10s.
    # Memory: Q_chunk@K^T = [1,nh,chunk_size,seq_k]*bf16 = same budget as CustomCall.
    import math as _math
    _scale = 1.0 / _math.sqrt(query.shape[-1])
    chunks = []
    for i in range(0, seq_q, chunk_size):
        q_chunk = query[:, :, i:i + chunk_size, :]
        # [1,nh,chunk,dh] @ [1,nh,dh,seq_k] -> [1,nh,chunk,seq_k]
        scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * _scale
        weights = torch.nn.functional.softmax(scores, dim=-1)
        # [1,nh,chunk,seq_k] @ [1,nh,seq_k,dh] -> [1,nh,chunk,dh]
        out_chunk = torch.matmul(weights, value)
        # Force XLA to execute this chunk before building the next graph.
        # Without mark_step(), all N chunk SDPA ops are fused into one graph
        # and the runtime allocates N x QK^T intermediates simultaneously.
        xm.mark_step()
        chunks.append(out_chunk)
    return torch.cat(chunks, dim=2)


# =============================================================================
# Ulysses Sequence Parallel attention processor
# =============================================================================

class WanUlyssesAttnProcessor:
    """
    DeepSpeed-Ulysses-style Sequence Parallel attention for Wan2.1 transformer.

    Strategy vs head-TP:
      head-TP: FFN sees full seq=32,760 → SliceOp DRAM OOM at seq > 13,440
      Ulysses SP: FFN sees seq/N per chip → 4× DRAM reduction → fits within ceiling

    Each device holds [B, seq/N, dim] throughout FFN and non-attention layers.
    Two all-to-all ops per attention block (via SPMD mark_sharding reshuffle):
      1. Before SDPA: [B, seq/N, heads, d] → [B, heads/N, seq, d]  per device
      2. After SDPA:  [B, heads/N, seq, d] → [B, seq/N, heads, d]  per device

    SDPA padding: 480×832 81f has seq=32,760 (not 32-aligned per chip: 8,190).
      Pads per-chip seq 8,190 -> 8,192 BEFORE all-to-all (pre-alltoall padding).
      After all-to-all: global Sk = 4 x 8,192 = 32,768 (32-aligned).
      This enables TTNN streaming compute (Flash Attn 2), cutting SDPA memory
      from 21.47 GB (O(seq^2)) to ~1.7 GB (O(chunk*seq)), preventing OOM.
      Key: padding BEFORE mark_sharding avoids losing SPMD sharding annotations.
      Output is trimmed back to original seq after the output projection.

    Cross-attention: Q is seq-sharded; encoder K/V are replicated (T5 seq=512).
      Q: all-to-all → head-sharded [B, heads/N, seq, d]
      K/V: static head partition [B, heads/N, T5_seq, d]  (no communication)
    """

    def __init__(self, mesh, num_devices: int, full_seq_len: int, sdpa_chunk_size: int = 0):
        self.mesh = mesh
        self.num_devices = num_devices
        self.full_seq_len = full_seq_len
        self.sdpa_chunk_size = sdpa_chunk_size
        # Per-chip padding: pad per-chip seq to 32-aligned BEFORE all-to-all.
        # This ensures global Sk = num_devices * padded_per_chip is 32-aligned,
        # enabling TTNN streaming compute (Flash Attention 2) without losing
        # sharding annotations (F.pad AFTER mark_sharding drops annotations).
        seq_per_chip = full_seq_len // num_devices
        self.pad_len = (32 - seq_per_chip % 32) % 32  # per-chip padding
        self.padded_seq = full_seq_len + self.pad_len * num_devices
        if self.pad_len > 0:
            print(
                f"  [Ulysses] seq={full_seq_len} (per-chip={seq_per_chip}) not 32-aligned; "
                f"padding per-chip +{self.pad_len} -> global padded={self.padded_seq} for TT SDPA"
            )
        else:
            print(f"  [Ulysses] seq={full_seq_len} (per-chip={seq_per_chip}) is 32-aligned -- no padding")
        if sdpa_chunk_size > 0:
            n_chunks = (self.padded_seq + sdpa_chunk_size - 1) // sdpa_chunk_size
            peak_gb = sdpa_chunk_size * self.padded_seq * 10 * 2 / 1e9
            print(
                f"  [Ulysses] chunked SDPA enabled: chunk_size={sdpa_chunk_size}, "
                f"n_chunks={n_chunks}, peak_SDPA~{peak_gb:.2f} GB/chip"
            )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        import torch.nn.functional as F
        from tt_torch.custom_ops import scaled_dot_product_attention as tt_sdpa
        from diffusers.models.transformers.transformer_wan import _get_qkv_projections

        is_cross_attn = encoder_hidden_states is not None
        print(f"  [Ulysses DEBUG] __call__ entry: hidden_states.shape={hidden_states.shape} is_cross_attn={is_cross_attn}")

        # ── QKV projections ───────────────────────────────────────────────────
        # hidden_states: [B, seq/N, inner_dim]  (seq-sharded)
        # encoder_hidden_states: [B, T5_seq, inner_dim]  (replicated, cross-attn only)
        # Weights are replicated; SPMD propagates seq-sharding through matmul.
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Unflatten: [..., heads, head_dim]
        query = query.unflatten(2, (attn.heads, -1))   # [B, seq/N, heads, head_dim]
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply RoPE (self-attention only; seq/N slice of freqs matches seq/N of Q/K)
        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # Post-RoPE padding: align global seq to next multiple of (32 * num_devices)
        # BEFORE mark_sharding/all-to-all. This ensures per-chip K/Q seq is 32-aligned
        # after the all-to-all, enabling TTNN streaming compute (Flash Attn 2).
        # Padding AFTER RoPE avoids rotary_emb shape mismatch (RoPE uses original seq).
        # Padding BEFORE mark_sharding preserves SPMD sharding annotations.
        # CROSS-ATTN: skip padding. Q_len x K_len = 32760 x 512 = 333 MB (fits fine).
        # Asymmetric tt.SDPA with padded Q (32768) and K=512 hangs in tt-mlir (TOOLS.md).
        orig_S = query.shape[1]  # global seq after QKV/RoPE, before padding
        _alignment = 32 * self.num_devices  # 128 for 4-device mesh
        global_pad = (_alignment - orig_S % _alignment) % _alignment if not is_cross_attn else 0
        if global_pad > 0:
            # Self-attention only: pad Q/K/V for 32-aligned per-chip seq after all-to-all
            query = F.pad(query, (0, 0, 0, 0, 0, global_pad))  # [B, S+pad, H, D]
            key = F.pad(key, (0, 0, 0, 0, 0, global_pad))
            value = F.pad(value, (0, 0, 0, 0, 0, global_pad))
            print(
                f"  [Ulysses DEBUG] post-RoPE pad: seq {orig_S} -> {orig_S + global_pad}"
            )
            # TASK-074: Re-annotate seq-sharding after F.pad to ensure SPMD redistributes
            # evenly (8192/chip for seq=32768) rather than keeping uneven shards.
            # Without this, the all-to-all may see mismatched Q/K seq lengths.
            xs.mark_sharding(query, self.mesh, (None, "model", None, None))
            xs.mark_sharding(key, self.mesh, (None, "model", None, None))
            xs.mark_sharding(value, self.mesh, (None, "model", None, None))

        # ── ALL-TO-ALL 1: seq-sharding → head-sharding ────────────────────────
        # Q is [B, seq/N, heads, head_dim] sharded on dim-1.
        # Transpose → [B, heads, seq/N, head_dim] sharded on dim-2.
        # Re-annotating dim-1 as 'model' tells SPMD we want head-sharding:
        # XLA inserts all-to-all → each device gets [B, heads/N, seq, head_dim].

        xs.mark_sharding(query, self.mesh, (None, "model", None, None))
        query = query.transpose(1, 2)                    # [B, heads, seq/N, head_dim]
        xs.mark_sharding(query, self.mesh, (None, "model", None, None))
        # Now per device: [B, heads/N, seq, head_dim]

        if not is_cross_attn:
            xs.mark_sharding(key, self.mesh, (None, "model", None, None))
            key = key.transpose(1, 2)
            xs.mark_sharding(key, self.mesh, (None, "model", None, None))

            xs.mark_sharding(value, self.mesh, (None, "model", None, None))
            value = value.transpose(1, 2)
            xs.mark_sharding(value, self.mesh, (None, "model", None, None))
        else:
            # Cross-attention: K/V from replicated encoder — partition across heads
            # (static slice, no all-to-all needed)
            key = key.transpose(1, 2)                    # [B, heads, T5_seq, head_dim]
            xs.mark_sharding(key, self.mesh, (None, "model", None, None))
            # → [B, heads/N, T5_seq, head_dim] per device

            value = value.transpose(1, 2)
            xs.mark_sharding(value, self.mesh, (None, "model", None, None))

        # -- TT SDPA ------------------------------------------------------------------
        # Note: no post-alltoall padding needed here -- hidden_states was already
        # padded BEFORE all-to-all so Q/K/V arrive already 32-aligned.
        print(
            f"  [Ulysses DEBUG] SDPA shapes: "
            f"Q={tuple(query.shape)} K={tuple(key.shape)} V={tuple(value.shape)}"
        )

        # TASK-072: Use tt_sdpa directly for self-attention so the runtime's
        # q_chunk_size=128 setting (in runScaledDotProductAttentionOp) enables
        # Flash Attention 2 streaming compute (O(q_chunk*Sk) memory vs O(Sq*Sk)).
        # Python-level chunking via chunked_tt_sdpa auto-chunks to 4096 but those
        # chunks get FUSED into one SPMD graph, allocating all intermediates at once:
        # 8 chunks × 2.68 GB = 21.47 GB → OOM. Calling tt_sdpa directly lets the
        # TTNN kernel handle streaming internally with q_chunk_size=128.
        # Cross-attention: asymmetric Q/K shapes (Q_len=~33024 ≠ K_len=226/512).
        # tt.SDPA hangs for asymmetric shapes (tt-mlir TOOLS.md); use matmul fallback.
        if not is_cross_attn:
            # Self-attention: runtime streaming handles memory (q_chunk_size=128).
            hidden_states_out = tt_sdpa(query, key, value, is_causal=False)
        else:
            # Cross-attention: decomposed matmul (asymmetric seq lengths).
            hidden_states_out = chunked_tt_sdpa(
                query, key, value,
                chunk_size=0,
                is_causal=False,
            )

        # ── ALL-TO-ALL 2: head-sharding → seq-sharding ────────────────────────
        # [B, heads/N, seq, head_dim] sharded on dim-1.
        # Transpose → [B, seq, heads/N, head_dim] sharded on dim-2.
        # Re-annotating dim-1 → SPMD all-to-all back to seq-sharding.
        # Result per device: [B, seq/N, heads, head_dim]

        hidden_states_out = hidden_states_out.transpose(1, 2)  # [B, seq, heads/N, head_dim]
        xs.mark_sharding(hidden_states_out, self.mesh, (None, "model", None, None))
        # → [B, seq/N, heads, head_dim] per device

        # -- Flatten + output projection -----------------------------------------------
        hidden_states_out = hidden_states_out.flatten(2, 3)    # [B, seq/N+pad, inner_dim]
        hidden_states_out = hidden_states_out.type_as(query)
        hidden_states_out = attn.to_out[0](hidden_states_out)  # [B, seq/N+pad, inner_dim]
        hidden_states_out = attn.to_out[1](hidden_states_out)
        # Trim padding tokens back to original seq (restores global seq=orig_S)
        if global_pad > 0:
            hidden_states_out = hidden_states_out[:, :orig_S, :]   # [B, seq, inner_dim]
        return hidden_states_out


def apply_ulysses_to_transformer(
    transformer, mesh, num_devices,
    height: int, width: int, num_frames: int,
    sdpa_chunk_size: int = 0,
):
    """
    Replace WanAttnProcessor with WanUlyssesAttnProcessor on all blocks.

    Call AFTER model load (no weight sharding in SP mode — weights are replicated).
    seq_len is the full logical sequence. Per-chip seq for FFN = seq_len / num_devices.
    TT SDPA runs on full seq per chip after the all-to-all (heads/N per chip).

    sdpa_chunk_size: split query into chunks of this size to bound peak DRAM.
      0 = no chunking (full SDPA). 8192 reduces 77f peak from 19.47→~4.6 GB/chip.
    """
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor

    seq_len = _compute_seq_len(height, width, num_frames)
    seq_per_chip = seq_len // num_devices

    pad_len = (32 - seq_len % 32) % 32
    print(f"  [Ulysses SP] seq={seq_len} | seq/chip (FFN)={seq_per_chip} "
          f"| SDPA seq={seq_len}+{pad_len}={seq_len+pad_len} | heads/chip={40 // num_devices}")

    proc = WanUlyssesAttnProcessor(mesh, num_devices, seq_len, sdpa_chunk_size=sdpa_chunk_size)
    n_replaced = 0
    for block in transformer.blocks:
        if isinstance(block.attn1.processor, WanAttnProcessor):
            block.attn1.set_processor(proc)
            n_replaced += 1
        if isinstance(block.attn2.processor, WanAttnProcessor):
            block.attn2.set_processor(proc)
            n_replaced += 1

    print(f"  [Ulysses SP] replaced {n_replaced} attention processors")
    return proc


# =============================================================================
# Ring SDPA Attention Processor — distributed SDPA with online softmax
# =============================================================================

class WanRingSDPAAttnProcessor:
    """
    Ring-style Distributed SDPA for Wan2.1 transformer.

    Strategy vs Ulysses SP:
      Ulysses SP:  all-to-all (seq→head) before SDPA → each chip runs full-seq SDPA
                   with heads/N heads. Peak QK^T = seq² × heads/N per chip.
      Ring SDPA:   Q stays seq-sharded (seq/N per chip). K/V are all-gathered
                   then processed in N chunks with online softmax accumulation.
                   Peak QK^T per chunk = seq/N × seq/N × heads per chip.

    Memory comparison (480×832 81f, seq=32760, N=4, heads=40):
      Ulysses (unchunked):  32768² × 10 × 2B = 21.5 GB/chip  → OOM
      Ulysses (chunk=8192): 8192 × 32768 × 10 × 2B = 5.37 GB/chip → OK
      Ring SDPA (step/chip): 8192 × 8192 × 40 × 2B = 5.37 GB/chip → OK

    Note: TTNN ring_distributed_scaled_dot_product_attention exists in C++ but
    is CAUSAL-ONLY (for autoregressive decode). Wan2.1 uses bidirectional
    attention, so this Python-level ring with online softmax is required.

    Cross-attention: Q is seq-sharded; encoder K/V are replicated (T5 seq≤512).
      Uses simple SDPA — seq/N × T5_seq cost is trivial, no chunking needed.
    """

    def __init__(self, mesh, num_devices: int, full_seq_len: int, ring_steps: int = 0):
        self.mesh = mesh
        self.num_devices = num_devices
        self.full_seq_len = full_seq_len
        # Chunk K/V into ring_steps pieces (not necessarily == num_devices).
        # Auto-select ring_steps to keep chunk_size < 4096, avoiding slow tt-mlir
        # compiler paths for large-K raw matmuls (TOOLS.md: K>4K gotcha).
        if ring_steps <= 0:
            # Auto: smallest N such that ceil(seq/N) < 4096
            import math
            ring_steps = max(num_devices, math.ceil(full_seq_len / 4095))
            # Round up to next multiple of num_devices for even distribution
            if ring_steps % num_devices != 0:
                ring_steps = ((ring_steps // num_devices) + 1) * num_devices
        self.ring_steps = ring_steps
        remainder = full_seq_len % ring_steps
        self.pad_len = (ring_steps - remainder) % ring_steps
        self.padded_seq = full_seq_len + self.pad_len
        self.chunk_size = self.padded_seq // ring_steps
        peak_gb = 40 * self.chunk_size * self.chunk_size * 2 / 1e9
        print(
            f"  [RingSDPA] seq={full_seq_len}+{self.pad_len}={self.padded_seq} "
            f"| chunk/step={self.chunk_size} | ring_steps={ring_steps} (devices={num_devices}) "
            f"| peak_QKT≈{peak_gb:.2f} GB/chip "
            f"(heads=40 × {self.chunk_size}² × 2B)"
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        import torch.nn.functional as F
        from diffusers.models.transformers.transformer_wan import _get_qkv_projections

        is_cross_attn = encoder_hidden_states is not None

        # ── QKV projections ───────────────────────────────────────────────────
        # hidden_states: [B, seq/N, inner_dim]  (seq-sharded)
        # encoder_hidden_states: [B, T5_seq, inner_dim]  (replicated, cross-attn only)
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key   = attn.norm_k(key)

        # Unflatten: [..., heads, head_dim]
        query = query.unflatten(2, (attn.heads, -1))   # [B, seq/N, heads, head_dim]
        key   = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply RoPE (self-attention only; seq/N RoPE slice already sharded to match Q/K)
        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)
            query = apply_rotary_emb(query, *rotary_emb)
            key   = apply_rotary_emb(key, *rotary_emb)

        # Transpose: [B, seq/N, heads, head_dim] → [B, heads, seq/N, head_dim]
        # Q stays seq-sharded on dim-2 throughout — no all-to-all needed.
        query = query.transpose(1, 2)   # [B, heads, seq/N, head_dim], sharded on dim-2
        key   = key.transpose(1, 2)     # [B, heads, seq/N, head_dim], sharded on dim-2
        value = value.transpose(1, 2)   # same

        B, H, Sq, D = query.shape
        scale = D ** -0.5

        if is_cross_attn:
            # ── Cross-attention ────────────────────────────────────────────────
            # K/V from replicated encoder (T5_seq ≤ 512) — no all-gather needed.
            # Q is seq-sharded: [B, heads, seq/N, head_dim]
            # K/V replicated:   [B, heads, T5_seq, head_dim]
            # Simple SDPA (T5_seq is tiny — no memory concern).
            scores = (query.float() @ key.float().transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores, dim=-1).to(query.dtype)
            hidden_states_out = attn_weights @ value
            # hidden_states_out: [B, heads, seq/N, head_dim] — seq/N per chip ✓

        else:
            # ── Self-attention: ring distributed SDPA ─────────────────────────
            # Step 1: All-gather K and V to get full sequence on each chip.
            #
            # v4 approach: CPU-roundtrip all-gather (outside XLA compiled graph).
            # xs.clear_sharding() inserts an all-gather collective INTO the XLA
            # graph, producing a [1,40,seq_full,128] output tensor that tt-mlir
            # must plan memory for.  For long sequences (seq=26520 for 65f) this
            # graph takes >60 min to compile.
            #
            # Instead: flush the pending lazy graph, then do a blocking CPU
            # transfer (which triggers the real all-gather at the UMD/mesh level
            # outside XLA's compiler), and move the result back to device as a
            # fresh tensor with no collective op in its computation history.
            # Ring step graphs then contain only: slice + matmul + softmax merge,
            # which compile quickly.
            xm.mark_step()  # flush QKV/RoPE graph before host transfer
            key_dev   = key.device
            key_dtype = key.dtype
            # .to("cpu") triggers blocking all-gather: seq-sharded → full seq
            key_full_cpu   = key.to("cpu")
            value_full_cpu = value.to("cpu")
            # Move back to XLA device as a fresh, unsharded (replicated) tensor
            key   = key_full_cpu.to(key_dev).to(key_dtype)
            value = value_full_cpu.to(key_dev).to(key_dtype)

            # Pad K/V so full_seq divides cleanly into ring_steps chunks
            if self.pad_len > 0:
                key   = F.pad(key,   (0, 0, 0, self.pad_len))
                value = F.pad(value, (0, 0, 0, self.pad_len))

            # Flush the device-placement graph so ring steps each compile
            # as their own independent graph (just matmul+softmax merge).
            xm.mark_step()

            # Step 2: Online softmax ring accumulation (flash-attention style).
            #   For each ring step i, compute partial attention over K_i/V_i chunk.
            #   Accumulate using the flash-attention running max/sum update rule.
            #   Peak QK^T per step: B × H × Sq × chunk_size
            #   = 1 × 40 × 8190 × 8192 × 2B ≈ 5.37 GB for 81f, N=4  ✓
            running_out = torch.zeros(B, H, Sq, D,    dtype=torch.float32, device=query.device)
            running_m   = torch.full((B, H, Sq, 1), float("-inf"), dtype=torch.float32, device=query.device)
            running_s   = torch.zeros((B, H, Sq, 1), dtype=torch.float32, device=query.device)

            for i in range(self.ring_steps):
                k_chunk = key  [:, :, i * self.chunk_size:(i + 1) * self.chunk_size, :]
                v_chunk = value[:, :, i * self.chunk_size:(i + 1) * self.chunk_size, :]

                # Attention scores: [B, H, Sq, chunk_size]
                scores = (query.float() @ k_chunk.float().transpose(-2, -1)) * scale

                # Per-chunk stable softmax components
                m_i   = scores.amax(dim=-1, keepdim=True)          # [B, H, Sq, 1]
                exp_i = torch.exp(scores - m_i)                    # [B, H, Sq, chunk]
                s_i   = exp_i.sum(dim=-1, keepdim=True)            # [B, H, Sq, 1]
                o_i   = exp_i @ v_chunk.float()                    # [B, H, Sq, D]

                # Flash-attention merge rule
                m_new         = torch.maximum(running_m, m_i)
                rescale_old   = torch.exp(running_m - m_new)
                rescale_chunk = torch.exp(m_i       - m_new)

                running_s   = running_s   * rescale_old + s_i * rescale_chunk
                running_out = running_out * rescale_old + o_i * rescale_chunk
                running_m   = m_new

                # Force XLA to execute this ring step before building the next graph.
                # Without mark_step(), all N steps fuse into one graph and the runtime
                # allocates N × QK^T intermediates simultaneously — negating the savings.
                xm.mark_step()

            # Normalize: divide accumulated weighted sum by accumulated denominator
            hidden_states_out = (running_out / running_s).to(query.dtype)  # [B, H, Sq, D]

        # ── Output: already seq-sharded ───────────────────────────────────────
        # hidden_states_out: [B, heads, seq/N, head_dim] — seq/N is local
        # Transpose back and mark seq-sharding so SPMD propagates correctly.
        hidden_states_out = hidden_states_out.transpose(1, 2)  # [B, seq/N, heads, head_dim]
        xs.mark_sharding(hidden_states_out, self.mesh, (None, "model", None, None))

        # ── Flatten + output projection ───────────────────────────────────────
        hidden_states_out = hidden_states_out.flatten(2, 3)       # [B, seq/N, inner_dim]
        hidden_states_out = hidden_states_out.type_as(query)
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        return hidden_states_out


def apply_ring_sdpa_to_transformer(
    transformer, mesh, num_devices,
    height: int, width: int, num_frames: int,
    ring_steps: int = 0,
):
    """
    Replace WanAttnProcessor with WanRingSDPAAttnProcessor on all blocks.

    Ring SDPA: Q stays seq-sharded, K/V all-gathered then chunked for online
    softmax accumulation. Requires seq-parallel activation sharding (same as
    Ulysses SP — both activated via the sp_mode flag in patch_transformer_with_tp).
    """
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor

    seq_len = _compute_seq_len(height, width, num_frames)
    seq_per_chip = seq_len // num_devices
    chunk_size = (seq_len + num_devices - 1) // num_devices
    peak_gb = 40 * chunk_size * chunk_size * 2 / 1e9

    print(f"  [Ring SDPA] seq={seq_len} | seq/chip={seq_per_chip} "
          f"| chunk_size={chunk_size} | peak_QKT≈{peak_gb:.2f} GB/chip")

    proc = WanRingSDPAAttnProcessor(mesh, num_devices, seq_len, ring_steps=ring_steps)
    n_replaced = 0
    for block in transformer.blocks:
        if isinstance(block.attn1.processor, WanAttnProcessor):
            block.attn1.set_processor(proc)
            n_replaced += 1
        if isinstance(block.attn2.processor, WanAttnProcessor):
            block.attn2.set_processor(proc)
            n_replaced += 1

    print(f"  [Ring SDPA] replaced {n_replaced} attention processors")
    return proc



# =============================================================================
# CPU SDPA Attention Processor — full attention on host CPU (bypasses XLA SDPA)
# =============================================================================

class WanCPUSDPAAttnProcessor:
    """
    CPU-bridge attention processor for Wan2.1 transformer.

    Strategy: run the entire SDPA on host CPU, bypassing XLA compilation for
    attention entirely. After mark_step() flushes the QKV+RoPE graph to device,
    Q/K/V are materialized to CPU (triggering the real all-gather at UMD level),
    then torch.nn.functional.scaled_dot_product_attention runs on CPU (180 GB RAM),
    and the result is moved back to the XLA device for the output projection.

    Memory on CPU: [1, 40, 32760, 128] Q/K/V @ bfloat16 = 3x 0.53 GB = 1.6 GB total.
    CPU SDPA (PyTorch uses efficient attention): no seq^2 intermediate allocation.

    Cross-attention: K/V from replicated encoder (T5 seq<=512) -- trivially small.

    Trade-off vs ring-sdpa: slower step time (CPU SDPA ~2-5s vs device ~0.5s),
    but ZERO XLA compilation for attention -> no 3h compile wall for 81f.
    """

    def __init__(self, mesh, num_devices: int, full_seq_len: int):
        self.mesh = mesh
        self.num_devices = num_devices
        self.full_seq_len = full_seq_len
        q_gb = full_seq_len * 40 * 128 * 2 / 1e9
        print(
            f"  [CPU SDPA] seq={full_seq_len} | Q/K/V on CPU each ~{q_gb:.2f} GB | "
            f"SDPA runs on host (no XLA compilation for attention)"
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        import torch.nn.functional as F
        from diffusers.models.transformers.transformer_wan import _get_qkv_projections

        is_cross_attn = encoder_hidden_states is not None

        # -- QKV projections on XLA device ------------------------------------
        # hidden_states: [B, seq/N, inner_dim]  (seq-sharded by SP)
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key   = attn.norm_k(key)

        # Unflatten: [..., heads, head_dim]
        query = query.unflatten(2, (attn.heads, -1))   # [B, seq/N, heads, head_dim]
        key   = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply RoPE on device (seq/N slice for self-attention)
        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)
            query = apply_rotary_emb(query, *rotary_emb)
            key   = apply_rotary_emb(key, *rotary_emb)

        # -- Flush QKV+RoPE graph, then move to CPU ---------------------------
        # mark_step() compiles + executes the lazy XLA graph for projections+RoPE.
        # .to("cpu") triggers a blocking all-gather (seq-sharded -> full seq on CPU).
        xm.mark_step()

        xla_device = query.device
        xla_dtype  = query.dtype

        # Transpose for SDPA: [B, seq/N, heads, head_dim] -> [B, heads, seq, head_dim]
        # (after CPU transfer the all-gather gives full seq)
        query_cpu = query.to("cpu").transpose(1, 2).float()   # [B, heads, seq, head_dim]
        key_cpu   = key.to("cpu").transpose(1, 2).float()
        value_cpu = value.to("cpu").transpose(1, 2).float()

        # -- SDPA on CPU ------------------------------------------------------
        # PyTorch selects flash_attention / memory_efficient / math backend.
        # On CPU: memory_efficient (sdp_kernel) or math. No OOM for 32k seq.
        out_cpu = F.scaled_dot_product_attention(
            query_cpu, key_cpu, value_cpu, is_causal=False
        )  # [B, heads, seq, head_dim]

        # -- Move output back to XLA, transpose to seq-sharded format ---------
        # Transpose: [B, heads, seq, head_dim] -> [B, seq, heads, head_dim]
        out_xla = out_cpu.to(xla_dtype).to(xla_device)   # whole seq on each chip
        out_xla = out_xla.transpose(1, 2)                 # [B, seq, heads, head_dim]

        # Re-introduce seq-sharding: mark the seq dim as sharded so SPMD
        # distributes it across chips (matches the SP FFN expectation).
        xs.mark_sharding(out_xla, self.mesh, (None, "model", None, None))
        # After mark_sharding: each chip holds [B, seq/N, heads, head_dim]

        # -- Flatten + output projection --------------------------------------
        hidden_states_out = out_xla.flatten(2, 3)          # [B, seq/N, inner_dim]
        hidden_states_out = hidden_states_out.to(xla_dtype)
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        return hidden_states_out


def apply_cpu_sdpa_to_transformer(
    transformer, mesh, num_devices,
    height: int, width: int, num_frames: int,
):
    """
    Replace WanAttnProcessor with WanCPUSDPAAttnProcessor on all blocks.

    CPU SDPA: all attention runs on host CPU after flushing QKV+RoPE to device.
    Completely bypasses XLA SDPA compilation. Requires seq-parallel activation
    sharding (sp_mode=True) for FFN layers to stay within DRAM budget.
    """
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor

    seq_len = _compute_seq_len(height, width, num_frames)
    seq_per_chip = seq_len // num_devices

    print(f"  [CPU SDPA] seq={seq_len} | seq/chip (FFN)={seq_per_chip} "
          f"| SDPA on CPU host (PyTorch flash/efficient backend)")

    proc = WanCPUSDPAAttnProcessor(mesh, num_devices, seq_len)
    n_replaced = 0
    for block in transformer.blocks:
        if isinstance(block.attn1.processor, WanAttnProcessor):
            block.attn1.set_processor(proc)
            n_replaced += 1
        if isinstance(block.attn2.processor, WanAttnProcessor):
            block.attn2.set_processor(proc)
            n_replaced += 1

    print(f"  [CPU SDPA] replaced {n_replaced} attention processors")
    return proc


def apply_tp_sharding_wan_block(block, mesh, num_devices):
    """
    Apply Megatron-style tensor parallel sharding to a single WanTransformerBlock.
    
    Block structure:
        attn1 (self-attention): to_q, to_k, to_v [5120, 5120], to_out.0 [5120, 5120]
            norm_q, norm_k [5120] — per-head norms, must be sharded
        attn2 (cross-attention): to_q [5120, 5120], to_k [5120, 5120], to_v [5120, 5120], to_out.0 [5120, 5120]
            add_k_proj, add_v_proj [5120, 5120] (for added KV from encoder)
            norm_q, norm_k [5120]
        ffn.net.0.proj [13824, 5120] (GELU up-projection)
        ffn.net.2 [5120, 13824] (down-projection)
    
    Sharding strategy:
        Q/K/V projections: column-parallel — ("model", None) on weight
        Output projection: row-parallel — (None, "model") on weight
        FFN up: column-parallel — ("model", None) on weight
        FFN down: row-parallel — (None, "model") on weight
    """
    shard_specs = {}

    # Self-attention (attn1)
    shard_specs[block.attn1.to_q.weight] = ("model", None)     # column-parallel
    shard_specs[block.attn1.to_q.bias] = ("model",)
    shard_specs[block.attn1.to_k.weight] = ("model", None)     # column-parallel
    shard_specs[block.attn1.to_k.bias] = ("model",)
    shard_specs[block.attn1.to_v.weight] = ("model", None)     # column-parallel
    shard_specs[block.attn1.to_v.bias] = ("model",)
    shard_specs[block.attn1.to_out[0].weight] = (None, "model")  # row-parallel
    shard_specs[block.attn1.to_out[0].bias] = (None,)           # replicated
    # Per-head norms (5120 = 40 heads * 128 dim_head)
    shard_specs[block.attn1.norm_q.weight] = ("model",)
    shard_specs[block.attn1.norm_k.weight] = ("model",)

    # Cross-attention (attn2)
    shard_specs[block.attn2.to_q.weight] = ("model", None)
    shard_specs[block.attn2.to_q.bias] = ("model",)
    shard_specs[block.attn2.to_k.weight] = ("model", None)
    shard_specs[block.attn2.to_k.bias] = ("model",)
    shard_specs[block.attn2.to_v.weight] = ("model", None)
    shard_specs[block.attn2.to_v.bias] = ("model",)
    shard_specs[block.attn2.to_out[0].weight] = (None, "model")
    shard_specs[block.attn2.to_out[0].bias] = (None,)
    shard_specs[block.attn2.norm_q.weight] = ("model",)
    shard_specs[block.attn2.norm_k.weight] = ("model",)

    # Cross-attention added KV projections (from encoder hidden states)
    if hasattr(block.attn2, 'add_k_proj') and block.attn2.add_k_proj is not None:
        shard_specs[block.attn2.add_k_proj.weight] = ("model", None)
        shard_specs[block.attn2.add_k_proj.bias] = ("model",)
    if hasattr(block.attn2, 'add_v_proj') and block.attn2.add_v_proj is not None:
        shard_specs[block.attn2.add_v_proj.weight] = ("model", None)
        shard_specs[block.attn2.add_v_proj.bias] = ("model",)

    # FFN: ffn.net.0.proj (up, GELU) — column-parallel
    shard_specs[block.ffn.net[0].proj.weight] = ("model", None)
    shard_specs[block.ffn.net[0].proj.bias] = ("model",)
    # FFN: ffn.net.2 (down) — row-parallel
    shard_specs[block.ffn.net[2].weight] = (None, "model")
    shard_specs[block.ffn.net[2].bias] = (None,)               # replicated

    return shard_specs


def apply_tp_sharding_wan_transformer(transformer, mesh, num_devices, sp_mode=False):
    """
    Apply tensor-parallel sharding to all 40 blocks of the WanTransformer3DModel.
    
    The pre-processing modules (patch_embedding, condition_embedder, rope) stay on CPU
    and are not sharded. Only the transformer blocks + final norm/proj are sharded.
    """
    if sp_mode:
        # Ulysses SP: no weight sharding — weights are replicated, activation (seq) is sharded
        print("  SP mode: weight sharding skipped (seq-parallel activations instead)")
        return {}

    all_specs = {}

    for i, block in enumerate(transformer.blocks):
        block_specs = apply_tp_sharding_wan_block(block, mesh, num_devices)
        all_specs.update(block_specs)

    # Apply all sharding annotations
    for tensor, spec in all_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    print(f"  Applied TP sharding to {len(transformer.blocks)} blocks ({len(all_specs)} tensors)")
    return all_specs


def patch_transformer_with_tp(transformer, mesh, num_devices, sp_mode=False):
    """
    Monkey-patch the WanTransformer3DModel forward for TP execution:
    1. Pre-processing on CPU (rope, patch_embedding, condition_embedder)
    2. Transfer to TT device — inputs are replicated across all devices
    3. Run blocks with SPMD — XLA handles the sharded matmuls + all-reduces
    4. Insert xm.mark_step() between blocks for incremental compilation
    """
    from diffusers.models.transformers.transformer_wan import Transformer2DModelOutput

    def forward_with_tp(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        tt_device = xm.xla_device()

        # === Phase 1: Pre-processing on CPU ===
        hidden_cpu = hidden_states.to("cpu")
        timestep_cpu = timestep.to("cpu")
        encoder_cpu = encoder_hidden_states.to("cpu")

        # Move preprocessing modules to CPU
        self.rope = self.rope.to("cpu")
        self.patch_embedding = self.patch_embedding.to("cpu")
        self.condition_embedder = self.condition_embedder.to("cpu")

        with torch.no_grad():
            rotary_emb = self.rope(hidden_cpu)

            hidden_states = self.patch_embedding(hidden_cpu)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)

            if timestep_cpu.ndim == 2:
                ts_seq_len = timestep_cpu.shape[1]
                timestep_flat = timestep_cpu.flatten()
            else:
                ts_seq_len = None
                timestep_flat = timestep_cpu

            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                timestep_flat, encoder_cpu,
                encoder_hidden_states_image.to("cpu") if encoder_hidden_states_image is not None else None,
                timestep_seq_len=ts_seq_len,
            )
            if ts_seq_len is not None:
                timestep_proj = timestep_proj.unflatten(2, (6, -1))
            else:
                timestep_proj = timestep_proj.unflatten(1, (6, -1))

            if encoder_hidden_states_image is not None:
                encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        print(f"    Pre-processing done on CPU. hidden={hidden_states.shape}, temb={temb.shape}", flush=True)

        # Move preprocessing modules back to TT device
        self.rope = self.rope.to(tt_device)
        self.patch_embedding = self.patch_embedding.to(tt_device)
        self.condition_embedder = self.condition_embedder.to(tt_device)

        # === Phase 2: Transfer to TT device (replicated across all chips) ===
        hidden_states = hidden_states.to(dtype=torch.bfloat16, device=tt_device)
        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.bfloat16, device=tt_device)
        timestep_proj = timestep_proj.to(dtype=torch.bfloat16, device=tt_device)
        if isinstance(rotary_emb, (tuple, list)):
            rotary_emb = tuple(r.to(dtype=torch.bfloat16, device=tt_device) for r in rotary_emb)
        else:
            rotary_emb = rotary_emb.to(dtype=torch.bfloat16, device=tt_device)

        # Sharding strategy depends on mode:
        #   head-TP:  all inputs replicated  → SPMD handles sharded matmuls (head-parallel)
        #   Ulysses SP: hidden_states seq-sharded → seq/N per device for FFN
        _sp_mode = getattr(forward_with_tp, "__sp_mode__", False)
        if _sp_mode:
            # Seq-sharded: hidden_states [B, seq, dim] → [B, seq/N, dim] per device
            xs.mark_sharding(hidden_states, mesh, (None, "model", None))
            # Encoder (cross-attn KV) stays replicated (T5 seq small)
            xs.mark_sharding(encoder_hidden_states, mesh, (None, None, None))
            # Timestep replicated (broadcasts over seq/N in LayerNorm modulation)
            if timestep_proj.ndim == 3:
                xs.mark_sharding(timestep_proj, mesh, (None, None, None))
            elif timestep_proj.ndim == 4:
                xs.mark_sharding(timestep_proj, mesh, (None, None, None, None))
            # RoPE freqs: seq-sharded so each device has the matching seq/N slice
            if isinstance(rotary_emb, (tuple, list)):
                for r in rotary_emb:
                    # freqs_cos/sin: [1, seq, 1, head_dim_part] → shard dim-1 (seq)
                    if r.ndim >= 2:
                        spec = (None, "model") + tuple(None for _ in range(r.ndim - 2))
                        xs.mark_sharding(r, mesh, spec)
        else:
            # Head-TP: mark inputs as replicated — SPMD handles the sharded matmuls
            xs.mark_sharding(hidden_states, mesh, (None, None, None))
            xs.mark_sharding(encoder_hidden_states, mesh, (None, None, None))
            if timestep_proj.ndim == 3:
                xs.mark_sharding(timestep_proj, mesh, (None, None, None))
            elif timestep_proj.ndim == 4:
                xs.mark_sharding(timestep_proj, mesh, (None, None, None, None))
            if isinstance(rotary_emb, (tuple, list)):
                for r in rotary_emb:
                    spec = tuple(None for _ in range(r.ndim))
                    xs.mark_sharding(r, mesh, spec)

        xm.mark_step()

        # === Phase 3: Transformer blocks with graph breaks ===
        # With SPMD + TP, XLA will compile sharded graphs -- each device only compiles
        # for its shard of the attention heads (10 heads instead of 40)
        #
        # TASK-074 FIX: Pad hidden_states + timestep_proj ONCE before the block loop.
        # Per-block F.pad on a seq-sharded tensor causes asymmetric Q/K shapes
        # after Ulysses all-to-all -> TTNN decomposes SDPA to matmul+TypecastOp
        # with full O(seq^2) QK^T (21.47 GB) -> OOM on Block 2+.
        # Padding once keeps seq=32768 for all 40 blocks (global_pad==0 in attn).
        # ALSO pad timestep_proj: for WAN 14B SP mode, timestep_proj is 4D
        # [B, seq, 6, inner_dim] -- must match hidden_states seq for
        # broadcasting in WanTransformerBlock (scale_msa/shift_msa + hidden_states).
        _hidden_states_orig_seq = None
        if _sp_mode:
            _ph3_align = 32 * num_devices
            _hs_seq = hidden_states.shape[1]
            _ph3_pad = (_ph3_align - _hs_seq % _ph3_align) % _ph3_align
            if _ph3_pad > 0:
                print(f"    [TASK-074] Pre-loop pad: seq {_hs_seq}->{_hs_seq+_ph3_pad}", flush=True)
                _hidden_states_orig_seq = _hs_seq
                import torch.nn.functional as _F3
                hidden_states = _F3.pad(hidden_states, (0, 0, 0, _ph3_pad))
                xs.mark_sharding(hidden_states, mesh, (None, "model", None))
                # Also pad timestep_proj if it has a seq dim matching hidden_states.
                # SP mode: timestep_proj is [B, seq, 6, inner_dim] (4D) for WAN 14B;
                # gets broadcast with hidden_states in scale_msa/shift_msa.
                if timestep_proj.ndim == 4 and timestep_proj.shape[1] == _hs_seq:
                    # Pad seq dim (dim 1) of [B, seq, 6, inner_dim]
                    # F.pad spec from last dim: (d3_l, d3_r, d2_l, d2_r, d1_l, d1_r, ...)
                    timestep_proj = _F3.pad(timestep_proj, (0, 0, 0, 0, 0, _ph3_pad, 0, 0))
                    print(f"    [TASK-074] Also padded timestep_proj: {timestep_proj.shape}", flush=True)
                # Also pad rotary_emb (RoPE freqs) if it has a seq dim matching hidden_states.
                # rotary_emb is a tuple (freqs_cos, freqs_sin) with shape [B, seq, 1, head_dim/2].
                # Pad with zeros -- extra tokens are trimmed after the block loop anyway.
                if isinstance(rotary_emb, (tuple, list)) and len(rotary_emb) == 2:
                    fc, fs = rotary_emb
                    # Find which dim is seq (== _hs_seq) -- typically dim 1 for [B, seq, 1, D]
                    # or dim 0 for [seq, 1, D]
                    _seq_dim = None
                    for _di, _ds in enumerate(fc.shape):
                        if _ds == _hs_seq:
                            _seq_dim = _di
                            break
                    if _seq_dim is not None:
                        # Build pad spec: pad only the seq dim on the right
                        _pad_spec = [0] * (fc.ndim * 2)
                        # F.pad spec is reversed (last dim first)
                        _pad_spec[2 * (fc.ndim - 1 - _seq_dim) + 1] = _ph3_pad
                        fc = _F3.pad(fc, _pad_spec)
                        fs = _F3.pad(fs, _pad_spec)
                        rotary_emb = (fc, fs)
                        print(f"    [TASK-074] Also padded rotary_emb: {fc.shape}", flush=True)
        for i, block in enumerate(self.blocks):
            block_start = time.time()
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            xm.mark_step()
            block_time = time.time() - block_start
            print(f"      Block {i+1}/{len(self.blocks)} done ({block_time:.1f}s)", flush=True)

        # TASK-074: Trim the pre-loop padding back to original seq length
        if _sp_mode and _hidden_states_orig_seq is not None:
            print(f"    [TASK-074] Post-loop trim: seq {hidden_states.shape[1]}->{_hidden_states_orig_seq}", flush=True)
            hidden_states = hidden_states[:, :_hidden_states_orig_seq, :]

        # === Phase 4: Output projection ===
        # In SP mode, hidden_states is [B, seq/N, dim] per chip (seq-sharded).
        # Problem: norm_out uses temb [B, seq, dim] for shift/scale (full-seq),
        # and the reshape below expects full seq (not seq/N).
        # Solution: flush TT graph, move to CPU (which materializes the full
        # logical [B, seq, dim] by all-gathering all shards), run norm+proj+reshape
        # on CPU, then move the result back to TT device.
        # This SP-mode CPU detour runs once per block per denoising step (not hot path).
        if _sp_mode:
            xm.mark_step()  # flush blocks graph first
            # Moving seq-sharded TT tensor to CPU gives the full logical tensor
            # (XLA gathers all shards into the global shape on host transfer)
            hidden_states_cpu = hidden_states.to(device="cpu", dtype=torch.float32)
            temb_cpu = temb.to(device="cpu", dtype=torch.float32)
            # Move output modules to CPU temporarily
            self.norm_out.to("cpu")
            self.proj_out.to("cpu")
            if temb_cpu.ndim == 3:
                sst = self.scale_shift_table.unsqueeze(0).to("cpu").float()
                shift, scale = (sst + temb_cpu.unsqueeze(2)).chunk(2, dim=2)
                shift = shift.squeeze(2)
                scale = scale.squeeze(2)
            else:
                sst = self.scale_shift_table.to("cpu").float()
                shift, scale = (sst + temb_cpu.unsqueeze(1)).chunk(2, dim=1)
            hidden_states_cpu = (
                self.norm_out(hidden_states_cpu) * (1 + scale) + shift
            ).to(torch.bfloat16)
            hidden_states_cpu = self.proj_out(hidden_states_cpu)
            # Restore modules to TT device
            self.norm_out.to(tt_device)
            self.proj_out.to(tt_device)
            hidden_states = hidden_states_cpu.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
                p_t, p_h, p_w, -1
            )
            hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        temb = temb.to(dtype=torch.bfloat16, device=tt_device)
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        xm.mark_step()

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    # Store sp_mode for the forward closure
    forward_with_tp.__sp_mode__ = sp_mode
    transformer.forward = types.MethodType(forward_with_tp, transformer)
    print(f"  Patched transformer forward with TP + graph breaks ({len(transformer.blocks)} blocks)")


class WanT2VTPConfig:
    """Configuration for Wan2.1-T2V pipeline with tensor parallelism or sequence parallelism."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        seq_parallel: bool = False,
        sdpa_chunk_size: int = 0,
        ring_sdpa: bool = False,
        cpu_sdpa: bool = False,
    ):
        self.model_id = model_id
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.use_tt_sdpa = True   # use TT fused SDPA when seq_len % 32 == 0
        self.seq_parallel = seq_parallel  # Ulysses SP: seq-sharded activations
        self.sdpa_chunk_size = sdpa_chunk_size  # 0 = no chunking; >0 chunks query seq
        self.ring_sdpa = ring_sdpa  # Ring SDPA: online-softmax ring instead of Ulysses all-to-all
        self.cpu_sdpa = cpu_sdpa  # CPU SDPA: full attention on CPU host, bypasses XLA compilation
        self.vae_tiling = False


class WanT2VTPPipeline:
    """
    Wan2.1 T2V pipeline with tensor parallelism across multiple Blackhole chips.
    """

    def __init__(self, config: WanT2VTPConfig):
        self.config = config
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.mesh = None
        self.num_devices = None

    def load_models(self):
        """Load all model components and apply TP sharding to transformer."""
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.models import WanTransformer3DModel
        from transformers import T5TokenizerFast, UMT5EncoderModel

        # Setup SPMD mesh
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()

        self.num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(self.num_devices))
        self.mesh = Mesh(device_ids, (1, self.num_devices), ("batch", "model"))

        print(f"SPMD enabled with {self.num_devices} devices")
        print(f"Mesh: {self.mesh}")

        # Validate head count divisibility
        num_heads = 40  # Wan2.1-T2V-14B has 40 attention heads
        if num_heads % self.num_devices != 0:
            raise ValueError(
                f"Number of attention heads ({num_heads}) must be divisible by "
                f"number of devices ({self.num_devices}) for head-parallel TP."
            )
        print(f"  {num_heads} heads / {self.num_devices} devices = {num_heads // self.num_devices} heads per device")

        print(f"\nLoading models from {self.config.model_id}...")
        start = time.time()

        # 1. Text encoder — CPU
        print("  Loading text encoder (UMT5-XXL)...")
        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.config.model_id, subfolder="tokenizer"
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to("cpu")
        self.text_encoder.eval()

        # 2. VAE — CPU
        print("  Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to("cpu")
        self.vae.eval()
        if getattr(self.config, 'vae_tiling', False):
            self.vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256)
            print('  VAE tiling enabled (256x256 tiles)')

        # 3. Scheduler
        print("  Loading scheduler...")
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.config.model_id, subfolder="scheduler"
        )

        # 4. Transformer — load then move to XLA device with TP sharding
        print("  Loading transformer (14B)...")
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.config.model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.transformer.eval()

        # Move to XLA device first, then apply sharding
        print("  Moving transformer to XLA device...")
        device = torch_xla.device()
        self.transformer = self.transformer.to(device)

        # Apply TP sharding annotations
        print("  Applying tensor-parallel sharding...")
        apply_tp_sharding_wan_transformer(self.transformer, self.mesh, self.num_devices)

        # Apply attention processors and patch forward
        sp_mode = getattr(self.config, "seq_parallel", False)
        ring_sdpa = getattr(self.config, "ring_sdpa", False)
        cpu_sdpa = getattr(self.config, "cpu_sdpa", False)
        if cpu_sdpa:
            # CPU SDPA: full attention on host CPU, zero XLA compilation for attention
            # Also requires seq-parallel activation sharding for FFN
            print("  Using CPU SDPA mode (full attention on host CPU, no XLA SDPA compilation)")
            apply_cpu_sdpa_to_transformer(
                self.transformer, self.mesh, self.num_devices,
                self.config.height, self.config.width, self.config.num_frames,
            )
            sp_mode = True  # CPU SDPA needs seq-parallel activation sharding
        elif ring_sdpa:
            # Ring SDPA: Q stays seq-sharded, K/V all-gathered + chunked online softmax
            # Also requires seq-parallel activation sharding (sp_mode=True)
            print("  Using Ring SDPA mode (Q seq-sharded, K/V all-gather + online softmax)")
            apply_ring_sdpa_to_transformer(
                self.transformer, self.mesh, self.num_devices,
                self.config.height, self.config.width, self.config.num_frames,
            )
            sp_mode = True  # Ring SDPA needs seq-parallel activation sharding
        elif sp_mode:
            # Ulysses SP: install Ulysses attn processors (handles all-to-all + SDPA)
            print("  Using Ulysses Sequence Parallelism (SP) mode")
            apply_ulysses_to_transformer(
                self.transformer, self.mesh, self.num_devices,
                self.config.height, self.config.width, self.config.num_frames,
                sdpa_chunk_size=getattr(self.config, "sdpa_chunk_size", 0),
            )
        else:
            # Head-TP: apply TT SDPA processors (if seq_len is 32-aligned)
            if getattr(self.config, "use_tt_sdpa", True):
                apply_tt_sdpa_to_transformer(
                    self.transformer, self.config.height, self.config.width, self.config.num_frames
                )

        # Patch forward for CPU pre-processing + graph breaks
        patch_transformer_with_tp(self.transformer, self.mesh, self.num_devices, sp_mode=sp_mode)

        elapsed = time.time() - start
        print(f"Models loaded and sharded in {elapsed:.1f}s")

    def encode_prompt(self, prompt: str, negative_prompt: str = "", max_sequence_length: int = 512):
        """Encode text prompt using UMT5."""
        print("Encoding prompt...")

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_inputs.input_ids.to("cpu"),
                attention_mask=text_inputs.attention_mask.to("cpu"),
            )[0]
        prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)

        negative_prompt_embeds = None
        if negative_prompt:
            uncond_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                negative_prompt_embeds = self.text_encoder(
                    uncond_inputs.input_ids.to("cpu"),
                    attention_mask=uncond_inputs.attention_mask.to("cpu"),
                )[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=torch.bfloat16)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels, height, width, num_frames, generator=None):
        """Prepare random latent noise."""
        vae_temporal_compression = 4
        vae_spatial_compression = 8

        latent_num_frames = (num_frames - 1) // vae_temporal_compression + 1
        latent_height = height // vae_spatial_compression
        latent_width = width // vae_spatial_compression

        shape = (batch_size, num_channels, latent_num_frames, latent_height, latent_width)
        latents = torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")

        return latents

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a video from text prompt using TP."""
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt)

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        num_channels = 16
        latents = self.prepare_latents(
            batch_size=1, num_channels=num_channels,
            height=height, width=width, num_frames=num_frames,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        timesteps = self.scheduler.timesteps
        latents = latents * self.scheduler.init_noise_sigma
        mask = torch.ones_like(latents)

        tt_device = torch_xla.device()
        tt_cast = lambda x: x.to(dtype=torch.bfloat16, device=tt_device)
        cpu_cast = lambda x: x.to(device="cpu", dtype=torch.float32)

        print(f"Running denoising loop ({num_inference_steps} steps)...", flush=True)
        loop_start = time.time()

        for i, t in enumerate(timesteps):
            step_start = time.time()
            print(f"  Step {i+1}/{num_inference_steps} (t={t.item():.2f})", flush=True)

            latent_model_input = latents.to(dtype=torch.bfloat16)

            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

            latent_input_tt = tt_cast(latent_model_input)
            timestep_tt = tt_cast(timestep)
            prompt_embeds_tt = tt_cast(prompt_embeds)

            noise_pred = self.transformer(
                hidden_states=latent_input_tt,
                timestep=timestep_tt,
                encoder_hidden_states=prompt_embeds_tt,
            )
            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample
            noise_pred = cpu_cast(noise_pred)

            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                neg_embeds_tt = tt_cast(negative_prompt_embeds)
                noise_pred_uncond = self.transformer(
                    hidden_states=latent_input_tt,
                    timestep=timestep_tt,
                    encoder_hidden_states=neg_embeds_tt,
                )
                if hasattr(noise_pred_uncond, "sample"):
                    noise_pred_uncond = noise_pred_uncond.sample
                noise_pred_uncond = cpu_cast(noise_pred_uncond)

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            step_time = time.time() - step_start
            print(f"    Step took {step_time:.2f}s", flush=True)

        loop_time = time.time() - loop_start
        print(f"Denoising complete in {loop_time:.1f}s ({loop_time/num_inference_steps:.1f}s/step)")

        # VAE decode on CPU
        print("Decoding latents with VAE...", flush=True)
        decode_start = time.time()

        latents = latents.to(dtype=self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]

        decode_time = time.time() - decode_start
        print(f"VAE decode took {decode_time:.1f}s")

        video = video.float().cpu()
        video = (video.clamp(-1, 1) + 1) / 2 * 255
        video = video[0].permute(1, 2, 3, 0).to(torch.uint8).numpy()

        return video


def run_wan_tp_pipeline(
    prompt: str = "A serene lake surrounded by mountains at sunset",
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    seed: Optional[int] = 42,
    output_path: str = "output_video_tp.mp4",
    optimization_level: int = 1,
    use_tt_sdpa: bool = True,
    seq_parallel: bool = False,
    sdpa_chunk_size: int = 0,
    vae_tiling: bool = False,
    ring_sdpa: bool = False,
    cpu_sdpa: bool = False,
):
    """Run the Wan T2V pipeline with tensor parallelism."""
    torch_xla.set_custom_compile_options({"optimization_level": optimization_level, "fp32_dest_acc_en": False})

    config = WanT2VTPConfig(
        height=height,
        width=width,
        num_frames=num_frames,
        seq_parallel=seq_parallel,
        sdpa_chunk_size=sdpa_chunk_size,
        ring_sdpa=ring_sdpa,
        cpu_sdpa=cpu_sdpa,
    )
    config.use_tt_sdpa = use_tt_sdpa
    config.vae_tiling = vae_tiling
    pipeline = WanT2VTPPipeline(config)
    pipeline.load_models()

    video_frames = pipeline.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    print(f"Generated {video_frames.shape[0]} frames at {video_frames.shape[1]}x{video_frames.shape[2]}")
    export_to_video(video_frames, output_path, fps=16)
    print(f"Video saved to: {output_path}")

    return output_path


def test_wan_t2v_tp():
    """Quick test: 1 denoising step at minimum resolution to verify TP compilation works."""
    xr.set_device_type("TT")

    output_path = "test_wan_tp_output.mp4"
    if Path(output_path).exists():
        Path(output_path).unlink()

    try:
        run_wan_tp_pipeline(
            prompt="a cat",
            num_inference_steps=1,
            height=256,
            width=256,
            num_frames=9,
            output_path=output_path,
        )
        assert Path(output_path).exists(), f"Output video {output_path} was not created"
        print(f"Test passed: video created at {output_path}")
    finally:
        if Path(output_path).exists():
            Path(output_path).unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-14B with Tensor Parallelism on Blackhole")
    parser.add_argument("--prompt", type=str, default="A serene lake surrounded by mountains at sunset")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="generated_videos")
    parser.add_argument("--optimization_level", type=int, default=1)
    parser.add_argument("--no_tt_sdpa", action="store_true",
                        help="Disable TT fused SDPA (fall back to F.scaled_dot_product_attention)")
    parser.add_argument("--seq-parallel", "--seq_parallel", action="store_true",
                        help="Use Ulysses Sequence Parallelism instead of head-TP. "
                             "Reduces per-chip FFN memory from seq to seq/N, enabling larger resolutions. "
                             "Recommended for 480x832 81f (seq=32,760 -> 8,190/chip for FFN).")
    parser.add_argument("--sdpa_chunk_size", "--sdpa-chunk-size", type=int, default=0,
                        help="Split SDPA query into chunks of this size to reduce peak DRAM. "
                             "0 = no chunking (default). 8192 recommended for 77f/81f at 480x832 "
                             "with --seq-parallel: drops peak from ~19.5 GB to ~4.6 GB/chip.")
    parser.add_argument("--vae-tiling", action="store_true",
                        help="Enable spatial tiling in VAE decoder to reduce peak memory")
    parser.add_argument("--ring-sdpa", "--ring_sdpa", action="store_true",
                        help="Use Ring SDPA: Q stays seq-sharded, K/V all-gathered then "
                             "processed in N ring steps with online softmax accumulation. "
                             "Peak QK^T = seq/N * seq/N * heads per chip (5.37 GB for 81f). "
                             "Enables 81f @ 480x832 (vs 65f ceiling without ring). "
                             "Implicitly enables seq-parallel activation sharding.")
    parser.add_argument("--cpu-sdpa", "--cpu_sdpa", action="store_true",
                        help="Use CPU SDPA: full attention runs on host CPU (180 GB RAM), "
                             "bypassing XLA compilation for SDPA entirely. "
                             "Enables 81f @ 480x832 with no compile overhead. "
                             "Implicitly enables seq-parallel activation sharding.")
    parser.add_argument("--ring-joint-sdpa", "--ring_joint_sdpa", action="store_true",
                        help="Alias for --cpu-sdpa (TTNN ring_joint SDPA not available; "
                             "uses CPU SDPA bridge instead).")
    parser.add_argument("--output-video", "--output_video", type=str, default=None,
                        help="Direct output video path (overrides --output_dir based naming)")
    args = parser.parse_args()

    xr.set_device_type("TT")

    os.makedirs(args.output_dir, exist_ok=True)

    filename = args.prompt[:50].lower().replace(" ", "_")
    filename = "".join(c if c.isalnum() or c == "_" else "" for c in filename)
    default_output_path = os.path.join(args.output_dir, f"{filename}_tp.mp4")
    # --output-video takes precedence over the auto-generated path
    output_path = getattr(args, "output_video", None) or default_output_path

    run_wan_tp_pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        output_path=output_path,
        optimization_level=args.optimization_level,
        use_tt_sdpa=not args.no_tt_sdpa,
        seq_parallel=getattr(args, "seq_parallel", False),
        sdpa_chunk_size=getattr(args, "sdpa_chunk_size", 0),
        vae_tiling=getattr(args, "vae_tiling", False),
        ring_sdpa=getattr(args, "ring_sdpa", False),
        cpu_sdpa=getattr(args, "cpu_sdpa", False) or getattr(args, "ring_joint_sdpa", False),
    )
