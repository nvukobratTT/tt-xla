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
    Replace all MochiAttnProcessor2_0 instances with TT-compatible version
    and wrap the forward to run PatchEmbed on CPU (workaround for Conv2d tilization bug).
    """
    import types

    count = 0
    for block in transformer.transformer_blocks:
        if hasattr(block, 'attn1') and hasattr(block.attn1, 'processor'):
            block.attn1.processor = MochiAttnProcessorTT()
            count += 1
    print(f"  Patched {count} attention processors for TT compatibility")

    # Wrap forward to run patch_embed (Conv2d) on CPU
    # This works around tt-xla #10: Conv2d output not auto-tilized
    _original_forward = transformer.forward.__func__ if hasattr(transformer.forward, '__func__') else transformer.forward

    def forward_with_cpu_patch_embed(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        attention_kwargs=None,
        return_dict=True,
    ):
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        device = hidden_states.device

        # Run patch_embed on CPU (Conv2d workaround)
        self.patch_embed = self.patch_embed.to("cpu")
        h_cpu = hidden_states.to("cpu")
        h_cpu = h_cpu.permute(0, 2, 1, 3, 4).flatten(0, 1)
        h_cpu = self.patch_embed(h_cpu)
        h_cpu = h_cpu.unflatten(0, (batch_size, -1)).flatten(1, 2)
        hidden_states = h_cpu.to(device=device, dtype=hidden_states.dtype)
        self.patch_embed = self.patch_embed.to(device)

        # time_embed on CPU (also has potential issues)
        self.time_embed = self.time_embed.to("cpu")
        temb, enc_out = self.time_embed(
            timestep.to("cpu"),
            encoder_hidden_states.to("cpu"),
            encoder_attention_mask.to("cpu"),
            hidden_dtype=torch.bfloat16,
        )
        temb = temb.to(device=device, dtype=torch.bfloat16)
        encoder_hidden_states = enc_out.to(device=device, dtype=torch.bfloat16)
        self.time_embed = self.time_embed.to(device)

        # RoPE on CPU
        self.rope = self.rope.to("cpu")
        pos_freq_cpu = self.pos_frequencies.to("cpu")
        image_rotary_emb = self.rope(
            pos_freq_cpu, num_frames, post_patch_height, post_patch_width,
            device="cpu", dtype=torch.float32,
        )
        image_rotary_emb = tuple(r.to(device=device, dtype=torch.bfloat16) for r in image_rotary_emb)
        self.rope = self.rope.to(device)

        # Transformer blocks — all on TT
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb,
                    encoder_attention_mask.to(device), image_rotary_emb,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    encoder_attention_mask=encoder_attention_mask.to(device),
                    image_rotary_emb=image_rotary_emb,
                )

        # Output norm + projection — on TT
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Reshape to video
        hidden_states = hidden_states.reshape(
            batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1
        )
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    transformer.forward = types.MethodType(forward_with_cpu_patch_embed, transformer)
    print(f"  Wrapped forward: PatchEmbed/TimeEmbed/RoPE on CPU, blocks on TT")
    return count
