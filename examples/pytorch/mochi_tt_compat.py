# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-compatible replacements for Mochi operations that don't compile on TT hardware.

Issues addressed:
1. MochiAttnProcessor2_0 uses torch.nonzero + dynamic indexing (not compilable)
2. RoPE uses strided slicing x[..., 0::2] (broken on TT, see tt-xla #5)
3. Per-batch for loop in attention (not XLA-friendly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MochiAttnProcessorTT:
    """
    TT-compatible attention processor for Mochi.

    Differences from MochiAttnProcessor2_0:
    - No dynamic masking (torch.nonzero) — assumes full attention mask
    - No per-batch for loop — batched SDPA
    - Concatenates video + text tokens for joint attention directly
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        # Video stream projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Context stream projections
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        # Apply RoPE (TT-compatible: no strided slicing)
        if image_rotary_emb is not None:
            query = apply_rotary_emb_tt(query, *image_rotary_emb)
            key = apply_rotary_emb_tt(key, *image_rotary_emb)

        # Transpose to (batch, heads, seq, dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        encoder_query = encoder_query.transpose(1, 2)
        encoder_key = encoder_key.transpose(1, 2)
        encoder_value = encoder_value.transpose(1, 2)

        sequence_length = query.size(2)
        encoder_sequence_length = encoder_query.size(2)

        # Joint attention: concatenate video + text tokens
        # No dynamic masking — use all tokens
        joint_query = torch.cat([query, encoder_query], dim=2)
        joint_key = torch.cat([key, encoder_key], dim=2)
        joint_value = torch.cat([value, encoder_value], dim=2)

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            joint_query, joint_key, joint_value, dropout_p=0.0, is_causal=False
        )

        # Split back into video and text streams
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )

        # Output projections
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


def apply_rotary_emb_tt(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """
    TT-compatible RoPE application.

    Avoids strided slicing (x[..., 0::2]) which is broken on TT hardware.
    Uses reshape + slice instead: reshape last dim to (half, 2), then index [:, :, :, :, 0] and [:, :, :, :, 1].
    """
    # x shape: (batch, seq, heads, dim_head) e.g. (1, N, 24, 128)
    *leading, d = x.shape
    half_d = d // 2

    # Reshape to (..., half_d, 2) to separate even/odd
    x_pairs = x.reshape(*leading, half_d, 2)
    x_even = x_pairs[..., 0].float()
    x_odd = x_pairs[..., 1].float()

    # Apply rotation
    cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
    sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

    # Interleave back: stack on last dim and flatten
    return torch.stack([cos, sin], dim=-1).reshape(*leading, d)


def patch_mochi_for_tt(transformer):
    """
    Replace all MochiAttnProcessor2_0 instances with TT-compatible version.
    """
    count = 0
    for block in transformer.transformer_blocks:
        if hasattr(block, 'attn1') and hasattr(block.attn1, 'processor'):
            block.attn1.processor = MochiAttnProcessorTT()
            count += 1
    print(f"  Patched {count} attention processors for TT compatibility")
    return count
