"""
Wan2.1-T2V-1.3B Hybrid CPU/TT Workaround Testing

Tests multiple configurations to find the best quality/performance tradeoff
for dealing with per-block numerical error compounding on TT Blackhole.

Configs:
  4: Full CPU (reference baseline)
  5: TT with f32 upcast between blocks
  1: Every-N-blocks on CPU (N=2,5,10,15)
  2: First/last K blocks on CPU
  3: CPU blocks with TT attention only (future - complex)
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Must set before any XLA device usage
xr.set_device_type("TT")

# ============================================================
# Shared: model loading, prompt encoding, RoPE patch
# ============================================================

def patch_rope_for_tt():
    """Fix RoPE: stack+flatten instead of broken strided assignment."""
    import diffusers.models.transformers.transformer_wan as wan_module
    original_call = wan_module.WanAttnProcessor.__call__

    def patched_call(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, rotary_emb=None):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = wan_module._get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            def apply_rotary_emb_tt_safe(hidden_states, freqs_cos, freqs_sin):
                x = hidden_states.unflatten(-1, (-1, 2))
                x1, x2 = x[..., 0], x[..., 1]
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out1 = x1 * cos - x2 * sin
                out2 = x1 * sin + x2 * cos
                return torch.stack([out1, out2], dim=-1).flatten(-2).type_as(hidden_states)
            query = apply_rotary_emb_tt_safe(query, *rotary_emb)
            key = apply_rotary_emb_tt_safe(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = wan_module._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = wan_module.dispatch_attention_fn(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0,
                is_causal=False, backend=self._attention_backend, parallel_config=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states = wan_module.dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False, backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    wan_module.WanAttnProcessor.__call__ = patched_call
    print("  Patched WanAttnProcessor with TT-safe RoPE")


def load_models(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"):
    """Load all model components. Returns dict of components."""
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from diffusers.models import WanTransformer3DModel
    from transformers import T5TokenizerFast, UMT5EncoderModel

    print(f"Loading models from {model_id}...")
    start = time.time()

    tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).to("cpu").eval()

    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    ).to("cpu").eval()

    scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()

    patch_rope_for_tt()

    print(f"Models loaded in {time.time() - start:.1f}s")
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "scheduler": scheduler,
        "transformer": transformer,
    }


def encode_prompt(tokenizer, text_encoder, prompt, negative_prompt="", max_len=512):
    """Encode text prompt on CPU."""
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=max_len,
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_inputs.input_ids.to("cpu"),
            attention_mask=text_inputs.attention_mask.to("cpu"),
        )[0].to(dtype=torch.bfloat16)

    negative_prompt_embeds = None
    if negative_prompt:
        uncond_inputs = tokenizer(
            negative_prompt, padding="max_length", max_length=max_len,
            truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        with torch.no_grad():
            negative_prompt_embeds = text_encoder(
                uncond_inputs.input_ids.to("cpu"),
                attention_mask=uncond_inputs.attention_mask.to("cpu"),
            )[0].to(dtype=torch.bfloat16)

    return prompt_embeds, negative_prompt_embeds


def prepare_latents(height, width, num_frames, seed=42):
    """Create initial noise latents on CPU."""
    latent_num_frames = (num_frames - 1) // 4 + 1
    latent_height = height // 8
    latent_width = width // 8
    shape = (1, 16, latent_num_frames, latent_height, latent_width)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")


def do_preprocessing_on_cpu(transformer, hidden_states, timestep, encoder_hidden_states):
    """Run patch_embedding, rope, condition_embedder on CPU. Returns all intermediates."""
    hidden_cpu = hidden_states.to("cpu")
    timestep_cpu = timestep.to("cpu")
    encoder_cpu = encoder_hidden_states.to("cpu")

    # Ensure preprocessing modules are on CPU
    transformer.rope = transformer.rope.to("cpu")
    transformer.patch_embedding = transformer.patch_embedding.to("cpu")
    transformer.condition_embedder = transformer.condition_embedder.to("cpu")

    with torch.no_grad():
        rotary_emb = transformer.rope(hidden_cpu)
        hidden_out = transformer.patch_embedding(hidden_cpu)
        hidden_out = hidden_out.flatten(2).transpose(1, 2)

        if timestep_cpu.ndim == 2:
            ts_seq_len = timestep_cpu.shape[1]
            timestep_flat = timestep_cpu.flatten()
        else:
            ts_seq_len = None
            timestep_flat = timestep_cpu

        temb, timestep_proj, enc_out, enc_img = transformer.condition_embedder(
            timestep_flat, encoder_cpu, None, timestep_seq_len=ts_seq_len,
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

    return hidden_out, temb, timestep_proj, rotary_emb, enc_out


def do_output_projection(transformer, hidden_states, temb, device="cpu"):
    """Run the final norm + projection."""
    # Move modules to target device (use .to() on module, not direct assignment)
    transformer.norm_out.to(device)
    transformer.proj_out.to(device)
    # scale_shift_table is a Parameter — use .data to get the tensor
    sst = transformer.scale_shift_table.data.to(device)

    hidden_states = hidden_states.to(device)
    temb = temb.to(device)

    if temb.ndim == 3:
        shift, scale = (sst.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (sst + temb.unsqueeze(1)).chunk(2, dim=1)

    hidden_states = (transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).to(torch.bfloat16)
    hidden_states = transformer.proj_out(hidden_states)
    return hidden_states


def reshape_output(hidden_states, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w):
    """Reshape transformer output back to video shape."""
    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


# ============================================================
# Config 4: Full CPU forward pass
# ============================================================

def forward_cpu(transformer, hidden_states_input, timestep, encoder_hidden_states):
    """Full forward pass on CPU (reference baseline)."""
    batch_size, num_channels, num_frames, height, width = hidden_states_input.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # Everything on CPU in bf16
    hidden_out, temb, timestep_proj, rotary_emb, enc_out = do_preprocessing_on_cpu(
        transformer, hidden_states_input, timestep, encoder_hidden_states
    )

    hidden_states = hidden_out.to(dtype=torch.bfloat16, device="cpu")
    enc_hs = enc_out.to(dtype=torch.bfloat16, device="cpu")
    ts_proj = timestep_proj.to(dtype=torch.bfloat16, device="cpu")
    rope = tuple(r.to(dtype=torch.bfloat16, device="cpu") for r in rotary_emb)

    # Ensure all blocks on CPU
    for block in transformer.blocks:
        block.to("cpu")

    for i, block in enumerate(transformer.blocks):
        hidden_states = block(hidden_states, enc_hs, ts_proj, rope)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"      CPU Block {i+1}/{len(transformer.blocks)}: std={hidden_states.std().item():.4f}", flush=True)

    output = do_output_projection(transformer, hidden_states, temb, device="cpu")
    output = reshape_output(output, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)
    return output


# ============================================================
# Config TT: Full TT forward pass (existing approach)
# ============================================================

def forward_tt(transformer, hidden_states_input, timestep, encoder_hidden_states):
    """Full forward pass on TT (existing approach with graph breaks)."""
    batch_size, num_channels, num_frames, height, width = hidden_states_input.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    tt_device = xm.xla_device()

    hidden_out, temb, timestep_proj, rotary_emb, enc_out = do_preprocessing_on_cpu(
        transformer, hidden_states_input, timestep, encoder_hidden_states
    )

    # Move to TT
    hidden_states = hidden_out.to(dtype=torch.bfloat16, device=tt_device)
    enc_hs = enc_out.to(dtype=torch.bfloat16, device=tt_device)
    ts_proj = timestep_proj.to(dtype=torch.bfloat16, device=tt_device)
    rope = tuple(r.to(dtype=torch.bfloat16, device=tt_device) for r in rotary_emb)
    xm.mark_step()

    # Ensure all blocks on TT
    for block in transformer.blocks:
        block.to(tt_device)

    for i, block in enumerate(transformer.blocks):
        hidden_states = block(hidden_states, enc_hs, ts_proj, rope)
        xm.mark_step()
        if (i + 1) % 10 == 0 or i == 0:
            hs_cpu = hidden_states.cpu()
            print(f"      TT Block {i+1}/{len(transformer.blocks)}: std={hs_cpu.std().item():.4f}", flush=True)

    # Output projection — pull to CPU to avoid parameter assignment issues
    hidden_states_cpu = hidden_states.cpu()
    output = do_output_projection(transformer, hidden_states_cpu, temb, device="cpu")
    output = reshape_output(output, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)
    return output


# ============================================================
# Config 5: TT with f32 upcast between blocks
# ============================================================

def forward_tt_f32_interblock(transformer, hidden_states_input, timestep, encoder_hidden_states):
    """TT forward with f32 hidden_states between blocks to reduce error accumulation."""
    batch_size, num_channels, num_frames, height, width = hidden_states_input.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    tt_device = xm.xla_device()

    hidden_out, temb, timestep_proj, rotary_emb, enc_out = do_preprocessing_on_cpu(
        transformer, hidden_states_input, timestep, encoder_hidden_states
    )

    hidden_states = hidden_out.to(dtype=torch.bfloat16, device=tt_device)
    enc_hs = enc_out.to(dtype=torch.bfloat16, device=tt_device)
    ts_proj = timestep_proj.to(dtype=torch.bfloat16, device=tt_device)
    rope = tuple(r.to(dtype=torch.bfloat16, device=tt_device) for r in rotary_emb)
    xm.mark_step()

    for block in transformer.blocks:
        block.to(tt_device)

    for i, block in enumerate(transformer.blocks):
        hidden_states = block(hidden_states, enc_hs, ts_proj, rope)
        # Upcast to f32 then back to bf16 — forces rounding at each step
        # This prevents accumulated bf16 drift within the residual stream
        hidden_states = hidden_states.float().to(torch.bfloat16)
        xm.mark_step()
        if (i + 1) % 10 == 0 or i == 0:
            hs_cpu = hidden_states.cpu()
            print(f"      TT+f32 Block {i+1}/{len(transformer.blocks)}: std={hs_cpu.std().item():.4f}", flush=True)

    hidden_states_cpu = hidden_states.cpu()
    output = do_output_projection(transformer, hidden_states_cpu, temb, device="cpu")
    output = reshape_output(output, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)
    return output


# ============================================================
# Config 1: Every-N-blocks on CPU
# ============================================================

def forward_hybrid_every_n(transformer, hidden_states_input, timestep, encoder_hidden_states, n_tt=2):
    """
    Run every Nth block on CPU to 'reset' accumulated error.
    n_tt: run this many consecutive blocks on TT before one CPU block.
    E.g., n_tt=2 means: TT, TT, CPU, TT, TT, CPU, ...
    """
    batch_size, num_channels, num_frames, height, width = hidden_states_input.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    tt_device = xm.xla_device()
    num_blocks = len(transformer.blocks)

    hidden_out, temb, timestep_proj, rotary_emb, enc_out = do_preprocessing_on_cpu(
        transformer, hidden_states_input, timestep, encoder_hidden_states
    )

    # Prepare both CPU and TT versions of context
    enc_hs_cpu = enc_out.to(dtype=torch.bfloat16, device="cpu")
    ts_proj_cpu = timestep_proj.to(dtype=torch.bfloat16, device="cpu")
    rope_cpu = tuple(r.to(dtype=torch.bfloat16, device="cpu") for r in rotary_emb)

    enc_hs_tt = enc_hs_cpu.to(device=tt_device)
    ts_proj_tt = ts_proj_cpu.to(device=tt_device)
    rope_tt = tuple(r.to(device=tt_device) for r in rope_cpu)
    xm.mark_step()

    hidden_states = hidden_out.to(dtype=torch.bfloat16, device="cpu")

    for i, block in enumerate(transformer.blocks):
        # Every (n_tt+1)th block runs on CPU
        use_cpu = ((i + 1) % (n_tt + 1) == 0) if n_tt < num_blocks else False

        if use_cpu:
            block.to("cpu")
            hidden_states = hidden_states.to("cpu")
            hidden_states = block(hidden_states, enc_hs_cpu, ts_proj_cpu, rope_cpu)
        else:
            block.to(tt_device)
            hidden_states = hidden_states.to(tt_device)
            hidden_states = block(hidden_states, enc_hs_tt, ts_proj_tt, rope_tt)
            xm.mark_step()
            hidden_states = hidden_states.cpu()  # Always pull back to CPU for transfer

        if (i + 1) % 10 == 0 or i == 0:
            hs = hidden_states.cpu() if hidden_states.device.type != "cpu" else hidden_states
            dev_str = "CPU" if use_cpu else "TT"
            print(f"      [{dev_str}] Block {i+1}/{num_blocks}: std={hs.std().item():.4f}", flush=True)

    output = do_output_projection(transformer, hidden_states, temb, device="cpu")
    output = reshape_output(output, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)
    return output


# ============================================================
# Config 2: First/last K blocks on CPU
# ============================================================

def forward_hybrid_first_last(transformer, hidden_states_input, timestep, encoder_hidden_states, cpu_first=3, cpu_last=3):
    """Run first K and last K blocks on CPU, middle on TT."""
    batch_size, num_channels, num_frames, height, width = hidden_states_input.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    tt_device = xm.xla_device()
    num_blocks = len(transformer.blocks)

    hidden_out, temb, timestep_proj, rotary_emb, enc_out = do_preprocessing_on_cpu(
        transformer, hidden_states_input, timestep, encoder_hidden_states
    )

    enc_hs_cpu = enc_out.to(dtype=torch.bfloat16, device="cpu")
    ts_proj_cpu = timestep_proj.to(dtype=torch.bfloat16, device="cpu")
    rope_cpu = tuple(r.to(dtype=torch.bfloat16, device="cpu") for r in rotary_emb)

    enc_hs_tt = enc_hs_cpu.to(device=tt_device)
    ts_proj_tt = ts_proj_cpu.to(device=tt_device)
    rope_tt = tuple(r.to(device=tt_device) for r in rope_cpu)
    xm.mark_step()

    hidden_states = hidden_out.to(dtype=torch.bfloat16, device="cpu")

    for i, block in enumerate(transformer.blocks):
        use_cpu = (i < cpu_first) or (i >= num_blocks - cpu_last)

        if use_cpu:
            block.to("cpu")
            hidden_states = hidden_states.to("cpu")
            hidden_states = block(hidden_states, enc_hs_cpu, ts_proj_cpu, rope_cpu)
        else:
            block.to(tt_device)
            hidden_states = hidden_states.to(tt_device)
            hidden_states = block(hidden_states, enc_hs_tt, ts_proj_tt, rope_tt)
            xm.mark_step()
            hidden_states = hidden_states.cpu()

        if (i + 1) % 10 == 0 or i == 0:
            hs = hidden_states if hidden_states.device.type == "cpu" else hidden_states.cpu()
            dev_str = "CPU" if use_cpu else "TT"
            print(f"      [{dev_str}] Block {i+1}/{num_blocks}: std={hs.std().item():.4f}", flush=True)

    output = do_output_projection(transformer, hidden_states, temb, device="cpu")
    output = reshape_output(output, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)
    return output


# ============================================================
# Diagnostic: Single TT block
# ============================================================

def forward_hybrid_single_tt(transformer, hidden_states_input, timestep, encoder_hidden_states, tt_block_idx=0):
    """Run only ONE block on TT, all others on CPU. Measures per-block TT impact."""
    batch_size, num_channels, num_frames, height, width = hidden_states_input.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    tt_device = xm.xla_device()
    num_blocks = len(transformer.blocks)

    hidden_out, temb, timestep_proj, rotary_emb, enc_out = do_preprocessing_on_cpu(
        transformer, hidden_states_input, timestep, encoder_hidden_states
    )

    enc_hs_cpu = enc_out.to(dtype=torch.bfloat16, device="cpu")
    ts_proj_cpu = timestep_proj.to(dtype=torch.bfloat16, device="cpu")
    rope_cpu = tuple(r.to(dtype=torch.bfloat16, device="cpu") for r in rotary_emb)

    enc_hs_tt = enc_hs_cpu.to(device=tt_device)
    ts_proj_tt = ts_proj_cpu.to(device=tt_device)
    rope_tt = tuple(r.to(device=tt_device) for r in rope_cpu)
    xm.mark_step()

    hidden_states = hidden_out.to(dtype=torch.bfloat16, device="cpu")

    for i, block in enumerate(transformer.blocks):
        if i == tt_block_idx:
            block.to(tt_device)
            hidden_states = hidden_states.to(tt_device)
            hidden_states = block(hidden_states, enc_hs_tt, ts_proj_tt, rope_tt)
            xm.mark_step()
            hidden_states = hidden_states.cpu()
        else:
            block.to("cpu")
            hidden_states = block(hidden_states, enc_hs_cpu, ts_proj_cpu, rope_cpu)

    output = do_output_projection(transformer, hidden_states, temb, device="cpu")
    output = reshape_output(output, batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)
    return output


# ============================================================
# Comparison utilities
# ============================================================

def compare_tensors(name, a, b):
    """Compare two tensors and print metrics."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()

    cos_sim = torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
    max_diff = (a_f - b_f).abs().max().item()
    mean_diff = (a_f - b_f).abs().mean().item()
    a_std = a_f.std().item()
    b_std = b_f.std().item()

    print(f"  {name}:")
    print(f"    cosine_sim={cos_sim:.6f}  max_diff={max_diff:.4f}  mean_diff={mean_diff:.6f}")
    print(f"    std_a={a_std:.4f}  std_b={b_std:.4f}  ratio={a_std/max(b_std, 1e-8):.4f}")
    return {"cosine_sim": cos_sim, "max_diff": max_diff, "mean_diff": mean_diff,
            "std_a": a_std, "std_b": b_std}


# ============================================================
# Main test harness
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="all",
                        help="Which config to test: 4,tt,5,1,2,all")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="A cat walking in the garden")
    parser.add_argument("--steps", type=int, default=1, help="Denoising steps for comparison")
    args = parser.parse_args()

    # Set compiler options for best TT precision
    torch_xla.set_custom_compile_options({
        "fp32_dest_acc_en": "true",
        "math_fidelity": "hifi4",
    })

    models = load_models()
    transformer = models["transformer"]
    scheduler = models["scheduler"]

    # Encode prompt
    prompt_embeds, neg_embeds = encode_prompt(
        models["tokenizer"], models["text_encoder"], args.prompt
    )

    # Prepare latents
    latents = prepare_latents(args.height, args.width, args.num_frames, args.seed)

    # Set up scheduler
    scheduler.set_timesteps(args.steps, device="cpu")
    timesteps = scheduler.timesteps
    latents = latents * scheduler.init_noise_sigma

    # For single-step comparison, we just run one forward pass
    t = timesteps[0]
    latent_input = latents.to(dtype=torch.bfloat16)
    timestep = t.expand(latents.shape[0])

    results = {}
    configs_to_test = []

    if args.config == "all":
        configs_to_test = ["4", "tt", "5", "1_2", "1_5", "1_10", "2_3_3", "2_5_5", "single_0", "single_14", "single_29"]
    else:
        configs_to_test = args.config.split(",")

    # Always run CPU reference first
    cpu_ref = None
    if "4" in configs_to_test or args.config == "all":
        print("\n" + "="*60)
        print("CONFIG 4: Full CPU (reference baseline)")
        print("="*60)
        start = time.time()
        cpu_output = forward_cpu(transformer, latent_input, timestep, prompt_embeds)
        cpu_time = time.time() - start
        cpu_ref = cpu_output.cpu().float()
        print(f"  Time: {cpu_time:.1f}s")
        print(f"  Output shape: {cpu_ref.shape}")
        print(f"  Output std: {cpu_ref.std().item():.4f}, mean: {cpu_ref.mean().item():.6f}")
        print(f"  Output range: [{cpu_ref.min().item():.4f}, {cpu_ref.max().item():.4f}]")
        results["config4_cpu"] = {
            "time": cpu_time,
            "std": cpu_ref.std().item(),
            "mean": cpu_ref.mean().item(),
        }
        configs_to_test = [c for c in configs_to_test if c != "4"]

    # We need CPU ref for comparisons
    if cpu_ref is None:
        print("\nRunning CPU reference for comparison...")
        start = time.time()
        cpu_output = forward_cpu(transformer, latent_input, timestep, prompt_embeds)
        cpu_time = time.time() - start
        cpu_ref = cpu_output.cpu().float()
        print(f"  CPU ref time: {cpu_time:.1f}s, std: {cpu_ref.std().item():.4f}")

    # Test each config
    for cfg in configs_to_test:
        print("\n" + "="*60)

        if cfg == "tt":
            print("CONFIG TT: Full TT (existing approach)")
            print("="*60)
            # Move transformer to TT
            tt_device = xm.xla_device()
            start = time.time()
            tt_output = forward_tt(transformer, latent_input, timestep, prompt_embeds)
            tt_time = time.time() - start
            tt_out_cpu = tt_output.cpu().float()
            metrics = compare_tensors("TT vs CPU", tt_out_cpu, cpu_ref)
            metrics["time"] = tt_time
            results["config_tt"] = metrics

        elif cfg == "5":
            print("CONFIG 5: TT with f32 upcast between blocks")
            print("="*60)
            start = time.time()
            f32_output = forward_tt_f32_interblock(transformer, latent_input, timestep, prompt_embeds)
            f32_time = time.time() - start
            f32_out_cpu = f32_output.cpu().float()
            metrics = compare_tensors("TT+f32 vs CPU", f32_out_cpu, cpu_ref)
            metrics["time"] = f32_time
            results["config5_f32upcast"] = metrics

        elif cfg.startswith("1_"):
            n = int(cfg.split("_")[1])
            print(f"CONFIG 1: Every-N on CPU (N_tt={n})")
            print("="*60)
            start = time.time()
            hybrid_output = forward_hybrid_every_n(transformer, latent_input, timestep, prompt_embeds, n_tt=n)
            hybrid_time = time.time() - start
            hybrid_cpu = hybrid_output.cpu().float()
            metrics = compare_tensors(f"Hybrid-N{n} vs CPU", hybrid_cpu, cpu_ref)
            metrics["time"] = hybrid_time
            results[f"config1_every{n}"] = metrics

        elif cfg.startswith("2_"):
            parts = cfg.split("_")
            first, last = int(parts[1]), int(parts[2])
            print(f"CONFIG 2: First {first} + Last {last} on CPU")
            print("="*60)
            start = time.time()
            hybrid_output = forward_hybrid_first_last(
                transformer, latent_input, timestep, prompt_embeds,
                cpu_first=first, cpu_last=last,
            )
            hybrid_time = time.time() - start
            hybrid_cpu = hybrid_output.cpu().float()
            metrics = compare_tensors(f"First{first}Last{last} vs CPU", hybrid_cpu, cpu_ref)
            metrics["time"] = hybrid_time
            results[f"config2_first{first}_last{last}"] = metrics

        elif cfg.startswith("single_"):
            # Run ONE specific block on TT, all others on CPU
            block_idx = int(cfg.split("_")[1])
            print(f"DIAGNOSTIC: Only block {block_idx} on TT, rest on CPU")
            print("="*60)
            start = time.time()
            # Use first/last config with only that one block on TT
            hybrid_output = forward_hybrid_single_tt(
                transformer, latent_input, timestep, prompt_embeds,
                tt_block_idx=block_idx,
            )
            hybrid_time = time.time() - start
            hybrid_cpu = hybrid_output.cpu().float()
            metrics = compare_tensors(f"Single-TT-block-{block_idx} vs CPU", hybrid_cpu, cpu_ref)
            metrics["time"] = hybrid_time
            results[f"single_tt_block_{block_idx}"] = metrics

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Config':<30} {'Cosine':>8} {'MaxDiff':>10} {'MeanDiff':>10} {'Time':>8}")
    print("-" * 70)
    for name, m in sorted(results.items()):
        cos = m.get("cosine_sim", 1.0)
        maxd = m.get("max_diff", 0.0)
        meand = m.get("mean_diff", 0.0)
        t = m.get("time", 0.0)
        print(f"{name:<30} {cos:>8.4f} {maxd:>10.4f} {meand:>10.6f} {t:>7.1f}s")

    # Save results
    results_path = "/workspace/tt-xla/docs/wan_hybrid_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
