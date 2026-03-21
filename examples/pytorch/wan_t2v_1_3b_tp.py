# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan2.1-T2V-1.3B with Tensor Parallelism across 4 Blackhole chips.

12 attention heads / 4 devices = 3 heads per device.
Megatron-style TP: column-parallel QKV/FFN-up, row-parallel O/FFN-down.
"""

import argparse
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


def apply_tp_sharding_1_3b(transformer, mesh):
    """Apply Megatron-style TP sharding to all blocks of the 1.3B transformer."""
    shard_specs = {}

    for block in transformer.blocks:
        # Self-attention (attn1)
        shard_specs[block.attn1.to_q.weight] = ("model", None)
        shard_specs[block.attn1.to_q.bias] = ("model",)
        shard_specs[block.attn1.to_k.weight] = ("model", None)
        shard_specs[block.attn1.to_k.bias] = ("model",)
        shard_specs[block.attn1.to_v.weight] = ("model", None)
        shard_specs[block.attn1.to_v.bias] = ("model",)
        shard_specs[block.attn1.to_out[0].weight] = (None, "model")
        shard_specs[block.attn1.to_out[0].bias] = (None,)
        shard_specs[block.attn1.norm_q.weight] = ("model",)
        shard_specs[block.attn1.norm_k.weight] = ("model",)

        # Cross-attention (attn2) — no add_k_proj/add_v_proj on 1.3B
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

        # FFN
        shard_specs[block.ffn.net[0].proj.weight] = ("model", None)
        shard_specs[block.ffn.net[0].proj.bias] = ("model",)
        shard_specs[block.ffn.net[2].weight] = (None, "model")
        shard_specs[block.ffn.net[2].bias] = (None,)

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    print(f"  Applied TP sharding to {len(transformer.blocks)} blocks ({len(shard_specs)} tensors)")


def patch_transformer_with_tp(transformer, mesh):
    """Monkey-patch forward: CPU pre-processing + TP blocks with graph breaks."""
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

        tt_device = torch_xla.device()

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

        self.rope = self.rope.to(tt_device)
        self.patch_embedding = self.patch_embedding.to(tt_device)
        self.condition_embedder = self.condition_embedder.to(tt_device)

        # === Phase 2: Transfer to TT (replicated) ===
        hidden_states = hidden_states.to(dtype=torch.bfloat16, device=tt_device)
        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.bfloat16, device=tt_device)
        timestep_proj = timestep_proj.to(dtype=torch.bfloat16, device=tt_device)
        if isinstance(rotary_emb, (tuple, list)):
            rotary_emb = tuple(r.to(dtype=torch.bfloat16, device=tt_device) for r in rotary_emb)
        else:
            rotary_emb = rotary_emb.to(dtype=torch.bfloat16, device=tt_device)

        # Mark inputs as replicated
        xs.mark_sharding(hidden_states, mesh, (None, None, None))
        xs.mark_sharding(encoder_hidden_states, mesh, (None, None, None))
        if timestep_proj.ndim == 3:
            xs.mark_sharding(timestep_proj, mesh, (None, None, None))
        elif timestep_proj.ndim == 4:
            xs.mark_sharding(timestep_proj, mesh, (None, None, None, None))
        if isinstance(rotary_emb, (tuple, list)):
            for r in rotary_emb:
                xs.mark_sharding(r, mesh, tuple(None for _ in range(r.ndim)))

        xm.mark_step()

        # === Phase 3: Blocks with graph breaks ===
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

    transformer.forward = types.MethodType(forward_with_tp, transformer)
    print(f"  Patched transformer forward with TP + graph breaks ({len(transformer.blocks)} blocks)")


class WanT2V1_3BTPPipeline:
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
        self.mesh = None
        self.num_devices = None

    def load_models(self):
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.models import WanTransformer3DModel
        from transformers import T5TokenizerFast, UMT5EncoderModel

        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()

        self.num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(self.num_devices))
        self.mesh = Mesh(device_ids, (1, self.num_devices), ("batch", "model"))

        num_heads = 12
        assert num_heads % self.num_devices == 0, f"{num_heads} heads not divisible by {self.num_devices} devices"
        print(f"SPMD enabled: {self.num_devices} devices, {num_heads // self.num_devices} heads/device")

        print(f"\nLoading models from {self.MODEL_ID}...")
        start = time.time()

        print("  Loading text encoder...")
        self.tokenizer = T5TokenizerFast.from_pretrained(self.MODEL_ID, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        ).to("cpu")
        self.text_encoder.eval()

        print("  Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
        ).to("cpu")
        self.vae.eval()

        print("  Loading scheduler...")
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.MODEL_ID, subfolder="scheduler")

        print("  Loading transformer (1.3B)...")
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
        )
        self.transformer.eval()

        print("  Moving transformer to XLA device...")
        device = torch_xla.device()
        self.transformer = self.transformer.to(device)

        print("  Applying tensor-parallel sharding...")
        apply_tp_sharding_1_3b(self.transformer, self.mesh)
        patch_transformer_with_tp(self.transformer, self.mesh)

        print(f"Models loaded and sharded in {time.time() - start:.1f}s")

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

        tt_device = torch_xla.device()
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


def run_pipeline(
    prompt="A serene lake surrounded by mountains at sunset",
    negative_prompt="", num_inference_steps=50, guidance_scale=5.0,
    height=480, width=832, num_frames=81, seed=42,
    output_path="output_video_1_3b_tp.mp4", optimization_level=1,
    cache_dir=None,
):
    # Enable runtime program cache — avoids recompilation within a run
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

    # Enable persistent cache if path provided (write-only for now, issue #498)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        xr.initialize_cache(cache_dir)
        print(f"Persistent cache enabled at: {cache_dir}")

    torch_xla.set_custom_compile_options({"optimization_level": optimization_level})

    pipeline = WanT2V1_3BTPPipeline(height=height, width=width, num_frames=num_frames)
    pipeline.load_models()

    video_frames = pipeline.generate(
        prompt=prompt, negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, seed=seed,
    )

    print(f"Generated {video_frames.shape[0]} frames at {video_frames.shape[1]}x{video_frames.shape[2]}")
    export_to_video(video_frames, output_path, fps=16)
    print(f"Video saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B TP on Blackhole")
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
    parser.add_argument("--cache_dir", type=str, default=None, help="Persistent cache directory")
    args = parser.parse_args()

    xr.set_device_type("TT")
    os.makedirs(args.output_dir, exist_ok=True)

    filename = args.prompt[:50].lower().replace(" ", "_")
    filename = "".join(c if c.isalnum() or c == "_" else "" for c in filename)
    output_path = os.path.join(args.output_dir, f"{filename}_1_3b_tp.mp4")

    run_pipeline(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        height=args.height, width=args.width, num_frames=args.num_frames,
        seed=args.seed, output_path=output_path, optimization_level=args.optimization_level,
        cache_dir=args.cache_dir,
    )
