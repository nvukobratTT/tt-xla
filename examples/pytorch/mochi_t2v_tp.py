# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi 1 Text-to-Video Pipeline with Tensor Parallelism across 4 Blackhole chips.

Strategy: Megatron-style TP for the ~10B transformer.
- 24 attention heads / 4 devices = 6 heads per device
- QKV projections: column-parallel (shard output dim)
- Output projection: row-parallel (shard input dim) + all-reduce
- FFN up: column-parallel
- FFN down: row-parallel + all-reduce
- Pre-processing (RoPE, patch_embed, time_embed) on CPU
- VAE and text encoder on CPU

Architecture:
- 48 MochiTransformerBlocks
- Joint attention: video tokens + text tokens (MochiAttention)
- Each block has: attn1 (joint), ff (video), ff_context (text, except last block)
- inner_dim = 24 * 128 = 3072
- FFN inner_dim = (4 * 3072 * 2) / 3 = 8192
- pooled_projection_dim = 1536 (for text/context stream)
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


def export_to_video(frames: np.ndarray, output_path: str, fps: int = 24):
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


def apply_tp_sharding_mochi_block(block, mesh, num_devices):
    """
    Apply Megatron-style tensor parallel sharding to a single MochiTransformerBlock.

    Block structure:
        attn1 (MochiAttention - joint attention):
            to_q [3072, 3072], to_k [3072, 3072], to_v [3072, 3072]  (no bias)
            add_q_proj [3072, 1536], add_k_proj [3072, 1536], add_v_proj [3072, 1536]  (no bias)
            to_out.0 [3072, 3072] (with bias)
            to_add_out [1536, 3072] (with bias, except last block)
            norm_q, norm_k [128] (per-head dim norms - RMSNorm with weight)
            norm_added_q, norm_added_k [128] (same)
        ff.net.0.proj [8192, 3072] (SwiGLU up, no bias)
        ff.net.2 [3072, 8192] (down, no bias)
        ff_context (same structure as ff, but dim=1536, inner=4096, except last block)

    Sharding strategy:
        Q/K/V: column-parallel — shard output dim (heads)
        Output: row-parallel — shard input dim
        FFN up: column-parallel
        FFN down: row-parallel
        Per-head norms: replicated (they operate on dim_head=128, not sharded)
    """
    shard_specs = {}

    # === Joint Attention (attn1) ===
    # Video stream Q/K/V — column-parallel (no bias)
    shard_specs[block.attn1.to_q.weight] = ("model", None)
    shard_specs[block.attn1.to_k.weight] = ("model", None)
    shard_specs[block.attn1.to_v.weight] = ("model", None)

    # Context stream Q/K/V — column-parallel (no bias)
    if hasattr(block.attn1, 'add_q_proj') and block.attn1.add_q_proj is not None:
        shard_specs[block.attn1.add_q_proj.weight] = ("model", None)
    if hasattr(block.attn1, 'add_k_proj') and block.attn1.add_k_proj is not None:
        shard_specs[block.attn1.add_k_proj.weight] = ("model", None)
    if hasattr(block.attn1, 'add_v_proj') and block.attn1.add_v_proj is not None:
        shard_specs[block.attn1.add_v_proj.weight] = ("model", None)

    # Output projection — row-parallel
    shard_specs[block.attn1.to_out[0].weight] = (None, "model")
    if block.attn1.to_out[0].bias is not None:
        shard_specs[block.attn1.to_out[0].bias] = (None,)  # replicated

    # Context output projection (exists in all blocks except last)
    if hasattr(block.attn1, 'to_add_out') and block.attn1.to_add_out is not None:
        shard_specs[block.attn1.to_add_out.weight] = (None, "model")
        if block.attn1.to_add_out.bias is not None:
            shard_specs[block.attn1.to_add_out.bias] = (None,)  # replicated

    # Per-head norms — these are dim_head=128 dimensional, NOT sharded
    # (RMSNorm operates per-head after unflatten, so weight is always 128-dim)
    # Leave them replicated (don't mark sharding)

    # === Video FFN ===
    # ff.net.0.proj (SwiGLU up) — column-parallel
    shard_specs[block.ff.net[0].proj.weight] = ("model", None)
    if hasattr(block.ff.net[0].proj, 'bias') and block.ff.net[0].proj.bias is not None:
        shard_specs[block.ff.net[0].proj.bias] = ("model",)

    # ff.net.2 (down) — row-parallel
    shard_specs[block.ff.net[2].weight] = (None, "model")
    if hasattr(block.ff.net[2], 'bias') and block.ff.net[2].bias is not None:
        shard_specs[block.ff.net[2].bias] = (None,)  # replicated

    # === Context FFN (exists in all blocks except last) ===
    if block.ff_context is not None:
        shard_specs[block.ff_context.net[0].proj.weight] = ("model", None)
        if hasattr(block.ff_context.net[0].proj, 'bias') and block.ff_context.net[0].proj.bias is not None:
            shard_specs[block.ff_context.net[0].proj.bias] = ("model",)

        shard_specs[block.ff_context.net[2].weight] = (None, "model")
        if hasattr(block.ff_context.net[2], 'bias') and block.ff_context.net[2].bias is not None:
            shard_specs[block.ff_context.net[2].bias] = (None,)

    # === Norm layers with linear projections ===
    # norm1 (MochiRMSNormZero) — has a linear that projects to 4*dim
    # This linear goes from dim → 4*dim, producing scale_msa, gate_msa, scale_mlp, gate_mlp
    # These are replicated (they modulate the hidden states, not the heads)

    return shard_specs


def apply_tp_sharding_mochi_transformer(transformer, mesh, num_devices):
    """
    Apply tensor-parallel sharding to all 48 blocks of MochiTransformer3DModel.

    Pre-processing modules (patch_embed, time_embed, rope, pos_frequencies)
    stay on CPU and are not sharded.
    """
    all_specs = {}

    for i, block in enumerate(transformer.transformer_blocks):
        block_specs = apply_tp_sharding_mochi_block(block, mesh, num_devices)
        all_specs.update(block_specs)

    # Apply all sharding annotations
    for tensor, spec in all_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    print(f"  Applied TP sharding to {len(transformer.transformer_blocks)} blocks ({len(all_specs)} tensors)")
    return all_specs


def patch_transformer_with_tp(transformer, mesh, num_devices):
    """
    Monkey-patch MochiTransformer3DModel forward for TP execution:
    1. Pre-processing on CPU (RoPE, patch_embed, time_embed)
    2. Transfer to TT device — inputs replicated across all devices
    3. Run blocks with SPMD — XLA handles sharded matmuls + all-reduces
    4. Insert xm.mark_step() between blocks for incremental compilation
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    def forward_with_tp(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.config.patch_size

        post_patch_height = height // p
        post_patch_width = width // p

        tt_device = xm.xla_device()

        # === Phase 1: Pre-processing on CPU ===
        hidden_cpu = hidden_states.to("cpu", dtype=torch.bfloat16)
        encoder_cpu = encoder_hidden_states.to("cpu", dtype=torch.bfloat16)
        timestep_cpu = timestep.to("cpu")
        mask_cpu = encoder_attention_mask.to("cpu", dtype=torch.bfloat16)

        # Move preprocessing modules to CPU
        self.time_embed = self.time_embed.to("cpu")
        self.patch_embed = self.patch_embed.to("cpu")
        self.rope = self.rope.to("cpu")
        self.pos_frequencies = self.pos_frequencies.to("cpu")

        with torch.no_grad():
            # Time embedding + text conditioning
            temb, encoder_hidden_states_out = self.time_embed(
                timestep_cpu,
                encoder_cpu,
                mask_cpu,
                hidden_dtype=torch.bfloat16,
            )

            # Patch embedding
            hidden_states_2d = hidden_cpu.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states_patched = self.patch_embed(hidden_states_2d)
            hidden_states_patched = hidden_states_patched.unflatten(0, (batch_size, -1)).flatten(1, 2)

            # RoPE
            image_rotary_emb = self.rope(
                self.pos_frequencies,
                num_frames,
                post_patch_height,
                post_patch_width,
                device="cpu",
                dtype=torch.float32,
            )

        print(f"    Pre-processing done on CPU. hidden={hidden_states_patched.shape}, "
              f"temb={temb.shape}, encoder={encoder_hidden_states_out.shape}", flush=True)

        # Move preprocessing back to TT
        self.time_embed = self.time_embed.to(tt_device)
        self.patch_embed = self.patch_embed.to(tt_device)
        self.rope = self.rope.to(tt_device)
        self.pos_frequencies = self.pos_frequencies.to(tt_device)

        # === Phase 2: Transfer to TT device (replicated across all chips) ===
        hidden_states = hidden_states_patched.to(dtype=torch.bfloat16, device=tt_device)
        encoder_hidden_states = encoder_hidden_states_out.to(dtype=torch.bfloat16, device=tt_device)
        temb = temb.to(dtype=torch.bfloat16, device=tt_device)
        encoder_attention_mask = mask_cpu.to(dtype=torch.bfloat16, device=tt_device)

        # RoPE embeddings
        if isinstance(image_rotary_emb, (tuple, list)):
            image_rotary_emb = tuple(
                r.to(dtype=torch.bfloat16, device=tt_device) for r in image_rotary_emb
            )
        else:
            image_rotary_emb = image_rotary_emb.to(dtype=torch.bfloat16, device=tt_device)

        # Mark inputs as replicated
        xs.mark_sharding(hidden_states, mesh, (None, None, None))
        xs.mark_sharding(encoder_hidden_states, mesh, (None, None, None))
        xs.mark_sharding(temb, mesh, tuple(None for _ in range(temb.ndim)))
        xs.mark_sharding(encoder_attention_mask, mesh, tuple(None for _ in range(encoder_attention_mask.ndim)))
        if isinstance(image_rotary_emb, (tuple, list)):
            for r in image_rotary_emb:
                xs.mark_sharding(r, mesh, tuple(None for _ in range(r.ndim)))

        xm.mark_step()

        # === Phase 3: Transformer blocks with graph breaks ===
        for i, block in enumerate(self.transformer_blocks):
            block_start = time.time()
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                encoder_attention_mask=encoder_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            xm.mark_step()
            block_time = time.time() - block_start
            print(f"      Block {i+1}/{len(self.transformer_blocks)} done ({block_time:.1f}s)", flush=True)

        # === Phase 4: Output projection ===
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        xm.mark_step()

        # Reshape back to video dimensions
        hidden_states = hidden_states.reshape(
            batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1
        )
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    transformer.forward = types.MethodType(forward_with_tp, transformer)
    print(f"  Patched transformer forward with TP + graph breaks ({len(transformer.transformer_blocks)} blocks)")


class MochiT2VTPConfig:
    """Configuration for Mochi T2V pipeline with tensor parallelism."""

    def __init__(
        self,
        model_id: str = "genmo/mochi-1-preview",
        height: int = 480,
        width: int = 848,
        num_frames: int = 19,
    ):
        self.model_id = model_id
        self.height = height
        self.width = width
        self.num_frames = num_frames


class MochiT2VTPPipeline:
    """
    Mochi 1 T2V pipeline with tensor parallelism across multiple Blackhole chips.
    """

    def __init__(self, config: MochiT2VTPConfig):
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
        from diffusers import AutoencoderKLMochi, FlowMatchEulerDiscreteScheduler
        from diffusers.models import MochiTransformer3DModel
        from transformers import T5EncoderModel, T5TokenizerFast

        # Setup SPMD mesh
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()

        self.num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(self.num_devices))
        self.mesh = Mesh(device_ids, (1, self.num_devices), ("batch", "model"))

        print(f"SPMD enabled with {self.num_devices} devices")
        print(f"Mesh: {self.mesh}")

        # Validate head count divisibility
        num_heads = 24  # Mochi has 24 attention heads
        if num_heads % self.num_devices != 0:
            raise ValueError(
                f"Number of attention heads ({num_heads}) must be divisible by "
                f"number of devices ({self.num_devices}) for head-parallel TP."
            )
        print(f"  {num_heads} heads / {self.num_devices} devices = {num_heads // self.num_devices} heads per device")

        print(f"\nLoading models from {self.config.model_id}...")
        start = time.time()

        # 1. Text encoder — CPU
        print("  Loading text encoder (T5-XXL)...")
        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.config.model_id, subfolder="tokenizer"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to("cpu")
        self.text_encoder.eval()

        # 2. VAE — CPU
        print("  Loading VAE...")
        self.vae = AutoencoderKLMochi.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to("cpu")
        self.vae.eval()
        self.vae.enable_tiling()

        # 3. Scheduler
        print("  Loading scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model_id, subfolder="scheduler"
        )

        # 4. Transformer — load then move to XLA device with TP sharding
        print("  Loading transformer (~10B)...")
        self.transformer = MochiTransformer3DModel.from_pretrained(
            self.config.model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.transformer.eval()

        # Move to XLA device
        print("  Moving transformer to XLA device...")
        device = torch_xla.device()
        self.transformer = self.transformer.to(device)

        # Apply TP sharding annotations
        print("  Applying tensor-parallel sharding...")
        apply_tp_sharding_mochi_transformer(self.transformer, self.mesh, self.num_devices)

        # Patch forward for CPU pre-processing + graph breaks
        patch_transformer_with_tp(self.transformer, self.mesh, self.num_devices)

        elapsed = time.time() - start
        print(f"Models loaded and sharded in {elapsed:.1f}s")

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        max_sequence_length: int = 256,
    ):
        """Encode text prompt using T5-XXL."""
        print("Encoding prompt...")

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_inputs.input_ids.to("cpu"),
                attention_mask=text_inputs.attention_mask.to("cpu"),
            )[0]
        prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)
        attention_mask = text_inputs.attention_mask.to(dtype=torch.bfloat16)

        negative_prompt_embeds = None
        negative_attention_mask = None
        if negative_prompt:
            uncond_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                negative_prompt_embeds = self.text_encoder(
                    uncond_inputs.input_ids.to("cpu"),
                    attention_mask=uncond_inputs.attention_mask.to("cpu"),
                )[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=torch.bfloat16)
            negative_attention_mask = uncond_inputs.attention_mask.to(dtype=torch.bfloat16)

        return prompt_embeds, attention_mask, negative_prompt_embeds, negative_attention_mask

    def prepare_latents(self, batch_size, num_channels, height, width, num_frames, generator=None):
        """Prepare random latent noise."""
        # Mochi VAE compression: temporal 6x, spatial 8x
        vae_temporal_compression = 6
        vae_spatial_compression = 8

        latent_num_frames = (num_frames - 1) // vae_temporal_compression + 1
        latent_height = height // vae_spatial_compression
        latent_width = width // vae_spatial_compression

        shape = (batch_size, num_channels, latent_num_frames, latent_height, latent_width)
        latents = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cpu")

        return latents

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        seed: Optional[int] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a video from text prompt using TP."""
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width

        # Encode prompt
        prompt_embeds, attention_mask, negative_prompt_embeds, negative_attention_mask = \
            self.encode_prompt(prompt, negative_prompt)

        # Prepare latents
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        num_channels = 12  # Mochi latent channels
        latents = self.prepare_latents(
            batch_size=1, num_channels=num_channels,
            height=height, width=width, num_frames=num_frames,
            generator=generator,
        )

        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        timesteps = self.scheduler.timesteps
        latents = latents * self.scheduler.init_noise_sigma

        tt_device = torch_xla.device()
        tt_cast = lambda x: x.to(dtype=torch.bfloat16, device=tt_device)
        cpu_cast = lambda x: x.to(device="cpu", dtype=torch.float32)

        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        print(f"Running denoising loop ({num_inference_steps} steps, CFG={do_cfg})...", flush=True)
        loop_start = time.time()

        for i, t in enumerate(timesteps):
            step_start = time.time()
            print(f"  Step {i+1}/{num_inference_steps} (t={t.item():.2f})", flush=True)

            latent_model_input = latents.to(dtype=torch.bfloat16)
            timestep = t.unsqueeze(0).to(torch.long)

            # Run transformer on TT
            noise_pred = self.transformer(
                hidden_states=tt_cast(latent_model_input),
                encoder_hidden_states=tt_cast(prompt_embeds),
                timestep=timestep.to(tt_device),
                encoder_attention_mask=tt_cast(attention_mask),
            )
            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample
            noise_pred = cpu_cast(noise_pred)

            # CFG
            if do_cfg:
                noise_pred_uncond = self.transformer(
                    hidden_states=tt_cast(latent_model_input),
                    encoder_hidden_states=tt_cast(negative_prompt_embeds),
                    timestep=timestep.to(tt_device),
                    encoder_attention_mask=tt_cast(negative_attention_mask),
                )
                if hasattr(noise_pred_uncond, "sample"):
                    noise_pred_uncond = noise_pred_uncond.sample
                noise_pred_uncond = cpu_cast(noise_pred_uncond)

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # Scheduler step on CPU
            latents = self.scheduler.step(noise_pred, t, latents.float(), return_dict=False)[0]

            step_time = time.time() - step_start
            print(f"    Step took {step_time:.2f}s", flush=True)

        loop_time = time.time() - loop_start
        print(f"Denoising complete in {loop_time:.1f}s ({loop_time/num_inference_steps:.1f}s/step)")

        # VAE decode on CPU
        print("Decoding latents with VAE...", flush=True)
        decode_start = time.time()

        latents = latents.to(dtype=torch.float32)
        video = self.vae.decode(latents, return_dict=False)[0]

        decode_time = time.time() - decode_start
        print(f"VAE decode took {decode_time:.1f}s")

        # Post-process
        video = video.float().cpu()
        video = (video.clamp(-1, 1) + 1) / 2 * 255
        video = video[0].permute(1, 2, 3, 0).to(torch.uint8).numpy()

        return video


def run_mochi_tp_pipeline(
    prompt: str = "A serene lake surrounded by mountains at sunset",
    negative_prompt: str = "",
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    height: int = 480,
    width: int = 848,
    num_frames: int = 19,
    seed: Optional[int] = 42,
    output_path: str = "output_mochi_tp.mp4",
    optimization_level: int = 1,
):
    """Run the Mochi T2V pipeline with tensor parallelism."""
    torch_xla.set_custom_compile_options({"optimization_level": optimization_level})

    config = MochiT2VTPConfig(
        height=height,
        width=width,
        num_frames=num_frames,
    )
    pipeline = MochiT2VTPPipeline(config)
    pipeline.load_models()

    video_frames = pipeline.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    print(f"Generated {video_frames.shape[0]} frames at {video_frames.shape[1]}x{video_frames.shape[2]}")
    export_to_video(video_frames, output_path, fps=24)
    print(f"Video saved to: {output_path}")

    return output_path


def test_mochi_t2v_tp():
    """Quick test: 1 denoising step at minimum resolution to verify TP compilation works."""
    xr.set_device_type("TT")

    output_path = "test_mochi_tp_output.mp4"
    if Path(output_path).exists():
        Path(output_path).unlink()

    try:
        run_mochi_tp_pipeline(
            prompt="a cat",
            num_inference_steps=1,
            guidance_scale=1.0,  # No CFG for speed
            height=64,
            width=64,
            num_frames=7,
            output_path=output_path,
        )
        assert Path(output_path).exists(), f"Output video {output_path} was not created"
        print(f"Test passed: video created at {output_path}")
    finally:
        if Path(output_path).exists():
            Path(output_path).unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mochi T2V with Tensor Parallelism on Blackhole")
    parser.add_argument("--prompt", type=str, default="A serene lake surrounded by mountains at sunset")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="generated_videos")
    parser.add_argument("--optimization_level", type=int, default=1)
    parser.add_argument("--test", action="store_true", help="Run quick test")

    args = parser.parse_args()

    xr.set_device_type("TT")

    if args.test:
        test_mochi_t2v_tp()
    else:
        os.makedirs(args.output_dir, exist_ok=True)

        filename = args.prompt[:50].lower().replace(" ", "_")
        filename = "".join(c if c.isalnum() or c == "_" else "" for c in filename)
        output_path = os.path.join(args.output_dir, f"{filename}_mochi_tp.mp4")

        run_mochi_tp_pipeline(
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
        )
