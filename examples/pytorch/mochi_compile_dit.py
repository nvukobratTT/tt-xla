#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi T2V with torch.compile(backend="tt", fullgraph=True) + 4-chip TP.

Full pipeline: T5 (CPU) → DiT (TT, compiled) → VAE (CPU) → video

Usage:
    # Generate video at 480p/5s
    python3 mochi_compile_dit.py --prompt "A cat playing piano"

    # Quick test at small resolution
    python3 mochi_compile_dit.py --prompt "A cat" --height 64 --width 64 --num-frames 7 --steps 4

    # Compile-only (no denoising, no VAE)
    python3 mochi_compile_dit.py --compile-only
"""

import argparse
import os
import time
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


# =============================================================================
# VIDEO EXPORT
# =============================================================================


def export_to_video(frames: np.ndarray, output_path: str, fps: int = 24):
    """Export uint8 frames [T, H, W, 3] to mp4."""
    try:
        import imageio.v3 as iio
        iio.imwrite(output_path, frames, fps=fps, codec="libx264", plugin="pyav")
    except ImportError:
        import cv2
        h, w = frames.shape[1], frames.shape[2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    print(f"  Saved: {output_path} ({frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]})")


# =============================================================================
# PATCHES — required for single-graph compilation (fullgraph=True)
# =============================================================================


def patch_unflatten_for_dynamo():
    """PATCH 1: Tensor.unflatten → view-based (super() graph break fix)."""
    def _unflatten_via_view(self, dim, sizes):
        if dim < 0:
            dim = self.ndim + dim
        shape = list(self.shape)
        new_shape = shape[:dim] + list(sizes) + shape[dim + 1:]
        return self.view(new_shape)

    torch.Tensor.unflatten = _unflatten_via_view
    print("  [PATCH 1] Tensor.unflatten → view-based")


def patch_attention_processors(model):
    """PATCH 2: MochiAttnProcessor2_0 → MochiAttnProcessorTT (torch.nonzero removal)."""
    from mochi_tt_compat import MochiAttnProcessorTT

    count = 0
    for block in model.transformer_blocks:
        if hasattr(block, "attn1") and hasattr(block.attn1, "processor"):
            block.attn1.processor = MochiAttnProcessorTT()
            count += 1
    print(f"  [PATCH 2] {count} attention processors → MochiAttnProcessorTT")
    return count


def patch_rope_autocast(model):
    """PATCH 3: _create_rope without autocast context manager."""
    def _create_rope_no_autocast(self, freqs, pos):
        freqs = torch.einsum("nd,dhf->nhf", pos.to(torch.float32), freqs.to(torch.float32))
        return torch.cos(freqs), torch.sin(freqs)

    model._create_rope = types.MethodType(_create_rope_no_autocast, model)
    print("  [PATCH 3] _create_rope → removed autocast")


def patch_swiglu_for_tp(model):
    """
    PATCH 4: Split GEGLU fused proj into separate gate+up for correct TP sharding.

    Problem: SwiGLU uses a single Linear(in, 2*hidden) and splits output via chunk(2, -1).
    With column-parallel TP on dim 0, slice indices on the sharded output don't correspond
    to the global gate/up boundary — each chip gets mixed gate+up rows.

    Fix: Pre-split into separate gate_proj and up_proj, each independently sharded.
    """
    import torch.nn.functional as F

    count = 0
    for block in model.transformer_blocks:
        for ff in [block.ff, block.ff_context]:
            if ff is None:
                continue
            geglu = ff.net[0]
            if not hasattr(geglu, 'proj'):
                continue

            W = geglu.proj.weight  # [2*hidden, in_features]
            half = W.shape[0] // 2

            # SwiGLU.forward: chunk(2,-1) -> [up, gate]; return up * silu(gate)
            # First half = up (no activation), second half = gate (gets silu)
            geglu.up_proj = nn.Linear(W.shape[1], half, bias=False, dtype=W.dtype)
            geglu.gate_proj = nn.Linear(W.shape[1], half, bias=False, dtype=W.dtype)
            geglu.up_proj.weight = nn.Parameter(W[:half].clone())
            geglu.gate_proj.weight = nn.Parameter(W[half:].clone())
            del geglu.proj

            def _swiglu_forward(self, x):
                return self.up_proj(x) * F.silu(self.gate_proj(x))
            geglu.forward = types.MethodType(_swiglu_forward, geglu)
            count += 1

    print(f"  [PATCH 4] {count} SwiGLU split (fused proj → gate_proj + up_proj)")
    return count


# =============================================================================
# TENSOR PARALLELISM
# =============================================================================


def apply_tp_sharding(model, mesh):
    """Megatron-style TP sharding for MochiTransformer3DModel (48 blocks)."""
    shard_specs = {}

    for block in model.transformer_blocks:
        # Attention Q/K/V — column-parallel
        shard_specs[block.attn1.to_q.weight] = ("model", None)
        shard_specs[block.attn1.to_k.weight] = ("model", None)
        shard_specs[block.attn1.to_v.weight] = ("model", None)

        if hasattr(block.attn1, 'add_q_proj') and block.attn1.add_q_proj is not None:
            shard_specs[block.attn1.add_q_proj.weight] = ("model", None)
        if hasattr(block.attn1, 'add_k_proj') and block.attn1.add_k_proj is not None:
            shard_specs[block.attn1.add_k_proj.weight] = ("model", None)
        if hasattr(block.attn1, 'add_v_proj') and block.attn1.add_v_proj is not None:
            shard_specs[block.attn1.add_v_proj.weight] = ("model", None)

        # Attention output — row-parallel
        shard_specs[block.attn1.to_out[0].weight] = (None, "model")
        if block.attn1.to_out[0].bias is not None:
            shard_specs[block.attn1.to_out[0].bias] = (None,)

        if hasattr(block.attn1, 'to_add_out') and block.attn1.to_add_out is not None:
            shard_specs[block.attn1.to_add_out.weight] = (None, "model")
            if block.attn1.to_add_out.bias is not None:
                shard_specs[block.attn1.to_add_out.bias] = (None,)

        # FFN — column/row parallel (split gate+up from PATCH 4)
        for ff in [block.ff, block.ff_context]:
            if ff is None:
                continue
            geglu = ff.net[0]
            if hasattr(geglu, 'gate_proj'):
                shard_specs[geglu.gate_proj.weight] = ("model", None)
                shard_specs[geglu.up_proj.weight] = ("model", None)
            elif hasattr(geglu, 'proj'):
                shard_specs[geglu.proj.weight] = ("model", None)
            shard_specs[ff.net[2].weight] = (None, "model")
            if hasattr(ff.net[2], 'bias') and ff.net[2].bias is not None:
                shard_specs[ff.net[2].bias] = (None,)

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    print(f"  TP sharding: {len(model.transformer_blocks)} blocks, {len(shard_specs)} tensors")


# =============================================================================
# PIPELINE
# =============================================================================


class MochiCompilePipeline:
    """Mochi T2V with torch.compile + SPMD TP."""

    def __init__(self, model_id="genmo/mochi-1-preview", height=480, width=848,
                 num_frames=121, opt_level=1):
        self.model_id = model_id
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.opt_level = opt_level

        assert (num_frames - 1) % 6 == 0, "(num_frames-1) must be divisible by 6"

    def setup(self):
        """Initialize SPMD, load all models, apply patches and TP, compile."""
        # --- SPMD ---
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.set_device_type("TT")
        xr.use_spmd()

        self.num_devices = xr.global_runtime_device_count()
        print(f"SPMD: {self.num_devices} devices")

        torch_xla.set_custom_compile_options({"optimization_level": self.opt_level})
        self.device = torch_xla.device()

        self.mesh = Mesh(
            np.array(range(self.num_devices)),
            (1, self.num_devices),
            ("batch", "model"),
        )

        # --- Patches ---
        print("\nPatches:")
        patch_unflatten_for_dynamo()

        # --- Load models ---
        from diffusers import AutoencoderKLMochi, FlowMatchEulerDiscreteScheduler
        from diffusers.models import MochiTransformer3DModel
        from transformers import T5EncoderModel, T5TokenizerFast

        print("\nLoading models:")

        # T5 text encoder — CPU
        print("  T5-XXL (CPU)...", flush=True)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        ).eval()

        # VAE — CPU
        print("  VAE (CPU)...", flush=True)
        self.vae = AutoencoderKLMochi.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=torch.float32,
        ).eval()

        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

        # DiT — TT device
        print("  DiT (TT)...", flush=True)
        t0 = time.time()
        self.transformer = MochiTransformer3DModel.from_pretrained(
            self.model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        ).eval()

        patch_attention_processors(self.transformer)
        patch_rope_autocast(self.transformer)
        patch_swiglu_for_tp(self.transformer)

        self.transformer = self.transformer.to(self.device)

        print("\nTP sharding:")
        apply_tp_sharding(self.transformer, self.mesh)

        # Compile
        print(f"\ntorch.compile(backend='tt', fullgraph=True)...")
        self.compiled_dit = torch.compile(self.transformer, backend="tt", fullgraph=True)

        print(f"DiT ready in {time.time() - t0:.1f}s")

    def encode_prompt(self, prompt: str, negative_prompt: str = "", max_seq_len: int = 256):
        """Encode text prompt with T5-XXL on CPU."""
        print(f"\nEncoding prompt: '{prompt}'")

        tokens = self.tokenizer(
            prompt, max_length=max_seq_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            embeds = self.text_encoder(
                tokens.input_ids, attention_mask=tokens.attention_mask,
            ).last_hidden_state.to(torch.bfloat16)
        mask = tokens.attention_mask.to(torch.bfloat16)

        neg_embeds, neg_mask = None, None
        if negative_prompt is not None:
            neg_tokens = self.tokenizer(
                negative_prompt, max_length=max_seq_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                neg_embeds = self.text_encoder(
                    neg_tokens.input_ids, attention_mask=neg_tokens.attention_mask,
                ).last_hidden_state.to(torch.bfloat16)
            neg_mask = neg_tokens.attention_mask.to(torch.bfloat16)

        print(f"  Embeddings: {list(embeds.shape)}")
        return embeds, mask, neg_embeds, neg_mask

    def prepare_latents(self, generator=None):
        """Prepare initial noise latents."""
        latent_frames = (self.num_frames - 1) // 6 + 1
        latents = torch.randn(
            1, 12, latent_frames, self.height // 8, self.width // 8,
            dtype=torch.bfloat16, generator=generator,
        )
        return latents

    def generate(self, prompt: str, negative_prompt: str = "",
                 num_inference_steps: int = 28, guidance_scale: float = 4.5,
                 seed: Optional[int] = None) -> np.ndarray:
        """Full T2V generation pipeline."""
        # Encode prompt
        embeds, mask, neg_embeds, neg_mask = self.encode_prompt(prompt, negative_prompt)

        # Prepare latents
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        latents = self.prepare_latents(generator)

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        timesteps = self.scheduler.timesteps

        do_cfg = guidance_scale > 1.0 and neg_embeds is not None

        print(f"\nDenoising ({num_inference_steps} steps, CFG={do_cfg}, "
              f"guidance={guidance_scale})...")
        print(f"  Latents: {list(latents.shape)}")
        loop_start = time.time()

        for i, t in enumerate(timesteps):
            step_start = time.time()
            timestep = t.unsqueeze(0).to(torch.long)

            # DiT forward on TT
            noise_pred = self.compiled_dit(
                hidden_states=latents.to(self.device),
                encoder_hidden_states=embeds.to(self.device),
                timestep=timestep.to(self.device),
                encoder_attention_mask=mask.to(self.device),
            )
            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample
            noise_pred = noise_pred.to("cpu", dtype=torch.float32)

            # CFG
            if do_cfg:
                noise_pred_uncond = self.compiled_dit(
                    hidden_states=latents.to(self.device),
                    encoder_hidden_states=neg_embeds.to(self.device),
                    timestep=timestep.to(self.device),
                    encoder_attention_mask=neg_mask.to(self.device),
                )
                if hasattr(noise_pred_uncond, "sample"):
                    noise_pred_uncond = noise_pred_uncond.sample
                noise_pred_uncond = noise_pred_uncond.to("cpu", dtype=torch.float32)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # Scheduler step (CPU)
            latents = self.scheduler.step(noise_pred, t, latents.float(), return_dict=False)[0]
            latents = latents.to(torch.bfloat16)

            step_time = time.time() - step_start
            if i == 0:
                print(f"  Step {i+1}/{num_inference_steps} (t={t.item():.2f}) — "
                      f"{step_time:.1f}s (includes compilation)")
            else:
                print(f"  Step {i+1}/{num_inference_steps} (t={t.item():.2f}) — {step_time:.2f}s")

        loop_time = time.time() - loop_start
        avg_step = loop_time / num_inference_steps
        # Exclude first step (compilation) for throughput estimate
        if num_inference_steps > 1:
            avg_cached = (loop_time - (time.time() - loop_start)) / max(num_inference_steps - 1, 1)
        print(f"Denoising done in {loop_time:.1f}s ({avg_step:.1f}s/step avg)")

        # VAE decode on CPU
        print("\nDecoding with VAE...", flush=True)
        decode_start = time.time()
        with torch.no_grad():
            video = self.vae.decode(latents.float(), return_dict=False)[0]
        print(f"  VAE decode: {time.time() - decode_start:.1f}s")

        # Post-process to uint8 frames [T, H, W, 3]
        video = (video.float().cpu().clamp(-1, 1) + 1) / 2 * 255
        video = video[0].permute(1, 2, 3, 0).to(torch.uint8).numpy()
        return video

    def compile_only(self):
        """Run one forward pass to trigger compilation without full pipeline."""
        h, w, nf = self.height, self.width, self.num_frames
        latent_frames = (nf - 1) // 6 + 1

        print(f"\nCompile-only: {h}x{w}, {nf} frames")
        print(f"  Latent: [1, 12, {latent_frames}, {h//8}, {w//8}]")

        t0 = time.time()
        with torch.no_grad():
            output = self.compiled_dit(
                hidden_states=torch.randn(
                    1, 12, latent_frames, h // 8, w // 8,
                    dtype=torch.bfloat16, device=self.device),
                encoder_hidden_states=torch.randn(
                    1, 256, 4096, dtype=torch.bfloat16, device=self.device),
                timestep=torch.tensor([500], dtype=torch.long, device=self.device),
                encoder_attention_mask=torch.ones(
                    1, 256, dtype=torch.bfloat16, device=self.device),
            )
        torch_xla.sync()
        print(f"Compile done in {time.time() - t0:.1f}s")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Mochi T2V with torch.compile + TP")
    parser.add_argument("--prompt", type=str,
                        default="A serene lake surrounded by mountains at sunset, "
                                "with gentle ripples on the water surface")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num-frames", type=int, default=121,
                        help="(n-1) must be divisible by 6. 121 = 5s at 24fps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimization-level", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: mochi_compile_<WxH>.mp4)")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile DiT, skip T5/VAE/generation")
    args = parser.parse_args()

    pipe = MochiCompilePipeline(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        opt_level=args.optimization_level,
    )
    pipe.setup()

    if args.compile_only:
        pipe.compile_only()
        return

    total_start = time.time()
    video = pipe.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    # Save video
    output_path = args.output or f"mochi_compile_{args.width}x{args.height}.mp4"
    export_to_video(video, output_path, fps=24)

    total_time = time.time() - total_start
    print(f"\nTotal: {total_time:.1f}s "
          f"({video.shape[0]} frames at {video.shape[1]}x{video.shape[2]})")


if __name__ == "__main__":
    main()
