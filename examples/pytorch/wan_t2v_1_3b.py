# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan2.1-T2V-1.3B Text-to-Video Pipeline for Tenstorrent Blackhole hardware.

The 1.3B model is much smaller than the 14B:
- 12 attention heads (vs 40), 1536 inner_dim (vs 5120)
- 30 layers (vs 40), ffn_dim 8960 (vs 13824)
- Attention shape: (1, 12, seq, 128) — should compile in reasonable time

Strategy: graph breaks between blocks (like the 14B approach) but compilation
should be fast enough per-block that we can finish the full pipeline.
Falls back to TP if single-device is still too slow.
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
import torch_xla.runtime as xr


def patch_rope_for_tt():
    """
    Fix RoPE implementation for TT hardware.

    The default diffusers WanAttnProcessor uses torch.empty_like() + strided
    slice assignment (out[..., 0::2] = ...) which is broken on tt-xla.
    The strided scatter write does not lower correctly through StableHLO→TTIR.

    Fix: Replace with torch.stack() + flatten() which uses concat/reshape ops
    that compile correctly on TT.
    """
    import diffusers.models.transformers.transformer_wan as wan_module

    original_call = wan_module.WanAttnProcessor.__call__

    def patched_call(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, rotary_emb=None):
        # Apply the TT-safe RoPE inside the processor
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
                """TT-safe RoPE: uses stack+flatten instead of broken strided assignment."""
                x = hidden_states.unflatten(-1, (-1, 2))
                x1, x2 = x[..., 0], x[..., 1]
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out1 = x1 * cos - x2 * sin
                out2 = x1 * sin + x2 * cos
                return torch.stack([out1, out2], dim=-1).flatten(-2).type_as(hidden_states)

            query = apply_rotary_emb_tt_safe(query, *rotary_emb)
            key = apply_rotary_emb_tt_safe(key, *rotary_emb)

        # I2V task
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
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = wan_module.dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False, backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    wan_module.WanAttnProcessor.__call__ = patched_call
    print("  Patched WanAttnProcessor with TT-safe RoPE (stack+flatten instead of strided assignment)")


def export_to_video(frames: np.ndarray, output_path: str, fps: int = 16):
    """Export frames to video."""
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
    raise RuntimeError("No video export backend available.")


def patch_transformer_with_graph_breaks(transformer):
    """
    Monkey-patch forward to:
    1. Pre-processing on CPU (rope, patch_embedding, condition_embedder)
    2. Transfer to TT device
    3. Run blocks with mark_step() between each for incremental compilation
    """
    from diffusers.models.transformers.transformer_wan import Transformer2DModelOutput

    def forward_with_graph_breaks(
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

        # Move preprocessing modules back to TT
        self.rope = self.rope.to(tt_device)
        self.patch_embedding = self.patch_embedding.to(tt_device)
        self.condition_embedder = self.condition_embedder.to(tt_device)

        # === Phase 2: Transfer to TT ===
        hidden_states = hidden_states.to(dtype=torch.bfloat16, device=tt_device)
        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.bfloat16, device=tt_device)
        timestep_proj = timestep_proj.to(dtype=torch.bfloat16, device=tt_device)
        if isinstance(rotary_emb, (tuple, list)):
            rotary_emb = tuple(r.to(dtype=torch.bfloat16, device=tt_device) for r in rotary_emb)
        else:
            rotary_emb = rotary_emb.to(dtype=torch.bfloat16, device=tt_device)
        xm.mark_step()

        # === Phase 3: Transformer blocks with graph breaks ===
        for i, block in enumerate(self.blocks):
            block_start = time.time()
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            xm.mark_step()
            block_time = time.time() - block_start
            print(f"      Block {i+1}/{len(self.blocks)} done ({block_time:.1f}s)", flush=True)

        # === Phase 4: Output projection ===
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

    transformer.forward = types.MethodType(forward_with_graph_breaks, transformer)
    print(f"  Patched transformer forward with graph breaks ({len(transformer.blocks)} blocks)")


class WanT2V1_3BPipeline:
    """Wan2.1-T2V-1.3B pipeline for TT hardware."""

    MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    def __init__(self, height=480, width=832, num_frames=81):
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def load_models(self):
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.models import WanTransformer3DModel
        from transformers import T5TokenizerFast, UMT5EncoderModel

        print(f"Loading models from {self.MODEL_ID}...")
        start = time.time()

        # Text encoder — UMT5 encoder-only (not the full encoder-decoder)
        print("  Loading text encoder...")
        self.tokenizer = T5TokenizerFast.from_pretrained(self.MODEL_ID, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        ).to("cpu")
        self.text_encoder.eval()

        # VAE
        print("  Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
        ).to("cpu")
        self.vae.eval()

        # Scheduler
        print("  Loading scheduler...")
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.MODEL_ID, subfolder="scheduler")

        # Transformer (1.3B — 30 layers, 12 heads)
        print("  Loading transformer (1.3B)...")
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
        )
        self.transformer.eval()

        patch_rope_for_tt()
        patch_transformer_with_graph_breaks(self.transformer)
        print("  Moving transformer to XLA device...")
        self.transformer = self.transformer.to(xm.xla_device())

        elapsed = time.time() - start
        print(f"Models loaded in {elapsed:.1f}s")

    def encode_prompt(self, prompt, negative_prompt="", max_sequence_length=512):
        print("Encoding prompt...")
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_attention_mask=True, return_tensors="pt",
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
                negative_prompt, padding="max_length", max_length=max_sequence_length,
                truncation=True, return_attention_mask=True, return_tensors="pt",
            )
            with torch.no_grad():
                negative_prompt_embeds = self.text_encoder(
                    uncond_inputs.input_ids.to("cpu"),
                    attention_mask=uncond_inputs.attention_mask.to("cpu"),
                )[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=torch.bfloat16)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels, height, width, num_frames, generator=None):
        latent_num_frames = (num_frames - 1) // 4 + 1
        latent_height = height // 8
        latent_width = width // 8
        shape = (batch_size, num_channels, latent_num_frames, latent_height, latent_width)
        return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def generate(
        self, prompt, negative_prompt="", num_inference_steps=50,
        guidance_scale=5.0, seed=None, num_frames=None, height=None, width=None,
    ):
        num_frames = num_frames or self.num_frames
        height = height or self.height
        width = width or self.width

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt)

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        latents = self.prepare_latents(1, 16, height, width, num_frames, generator)

        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        timesteps = self.scheduler.timesteps
        latents = latents * self.scheduler.init_noise_sigma

        tt_device = xm.xla_device()
        tt_cast = lambda x: x.to(dtype=torch.bfloat16, device=tt_device)
        cpu_cast = lambda x: x.to(device="cpu", dtype=torch.float32)

        print(f"Running denoising loop ({num_inference_steps} steps)...", flush=True)
        loop_start = time.time()

        for i, t in enumerate(timesteps):
            step_start = time.time()
            print(f"  Step {i+1}/{num_inference_steps} (t={t.item():.2f})", flush=True)

            latent_model_input = latents.to(dtype=torch.bfloat16)
            # 1.3B uses scalar timestep (expand_timesteps=False)
            timestep = t.expand(latents.shape[0])

            noise_pred = self.transformer(
                hidden_states=tt_cast(latent_model_input),
                timestep=tt_cast(timestep),
                encoder_hidden_states=tt_cast(prompt_embeds),
            )
            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample
            noise_pred = cpu_cast(noise_pred)

            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                noise_pred_uncond = self.transformer(
                    hidden_states=tt_cast(latent_model_input),
                    timestep=tt_cast(timestep),
                    encoder_hidden_states=tt_cast(negative_prompt_embeds),
                )
                if hasattr(noise_pred_uncond, "sample"):
                    noise_pred_uncond = noise_pred_uncond.sample
                noise_pred_uncond = cpu_cast(noise_pred_uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            print(f"    Step took {time.time() - step_start:.2f}s", flush=True)

        loop_time = time.time() - loop_start
        print(f"Denoising complete in {loop_time:.1f}s ({loop_time/num_inference_steps:.1f}s/step)")

        # VAE decode
        print("Decoding latents with VAE...", flush=True)
        decode_start = time.time()
        latents = latents.to(dtype=self.vae.dtype)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        print(f"VAE decode took {time.time() - decode_start:.1f}s")

        video = video.float().cpu()
        video = (video.clamp(-1, 1) + 1) / 2 * 255
        return video[0].permute(1, 2, 3, 0).to(torch.uint8).numpy()


def run_1_3b_pipeline(
    prompt="A serene lake surrounded by mountains at sunset",
    negative_prompt="",
    num_inference_steps=50,
    guidance_scale=5.0,
    height=480,
    width=832,
    num_frames=81,
    seed=42,
    output_path="output_video_1_3b.mp4",
    optimization_level=1,
):
    torch_xla.set_custom_compile_options({"optimization_level": optimization_level})

    pipeline = WanT2V1_3BPipeline(height=height, width=width, num_frames=num_frames)
    pipeline.load_models()

    video_frames = pipeline.generate(
        prompt=prompt, negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, seed=seed,
    )

    print(f"Generated {video_frames.shape[0]} frames at {video_frames.shape[1]}x{video_frames.shape[2]}")
    export_to_video(video_frames, output_path, fps=16)
    print(f"Video saved to: {output_path}")
    return output_path


def test_wan_1_3b():
    """Quick test: 1 step at minimum resolution."""
    xr.set_device_type("TT")
    output_path = "test_wan_1_3b.mp4"
    try:
        run_1_3b_pipeline(
            prompt="a cat", num_inference_steps=1,
            height=256, width=256, num_frames=9, output_path=output_path,
        )
        assert Path(output_path).exists()
        print(f"Test passed: {output_path}")
    finally:
        if Path(output_path).exists():
            Path(output_path).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B on Blackhole")
    parser.add_argument("--prompt", type=str, default="A serene lake surrounded by mountains at sunset")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="generated_videos")
    parser.add_argument("--optimization_level", type=int, default=1)
    args = parser.parse_args()

    xr.set_device_type("TT")
    os.makedirs(args.output_dir, exist_ok=True)

    filename = args.prompt[:50].lower().replace(" ", "_")
    filename = "".join(c if c.isalnum() or c == "_" else "" for c in filename)
    output_path = os.path.join(args.output_dir, f"{filename}_1_3b.mp4")

    run_1_3b_pipeline(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        height=args.height, width=args.width, num_frames=args.num_frames,
        seed=args.seed, output_path=output_path, optimization_level=args.optimization_level,
    )
