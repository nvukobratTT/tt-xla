# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan2.1-T2V-14B Text-to-Video Pipeline for Tenstorrent Blackhole hardware.

Key design: The 14B transformer is too large to compile as a single MLIR graph.
We monkey-patch the transformer's forward method to insert xm.mark_step() calls
between each of the 40 transformer blocks, forcing the XLA compiler to compile
each block as a separate graph (~7 min each on first run, cached thereafter).
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


def patch_transformer_with_graph_breaks(transformer):
    """
    Monkey-patch the WanTransformer3DModel forward to:
    1. Run patch_embedding, rope, and condition_embedder on CPU (avoids reshape segfault in tt-mlir)
    2. Transfer intermediates to TT device for the block loop
    3. Insert xm.mark_step() between each block for incremental compilation
    4. Run output norm/projection on TT, then transfer back
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
        # Move inputs to CPU for rope + patch_embedding + condition_embedder
        # This avoids the tt-mlir segfault in reshape during condition_embedder compilation
        hidden_cpu = hidden_states.to("cpu")
        timestep_cpu = timestep.to("cpu")
        encoder_cpu = encoder_hidden_states.to("cpu")

        # These modules need to be on CPU temporarily
        rope_device = next(self.rope.parameters()).device if list(self.rope.parameters()) else None
        patch_device = next(self.patch_embedding.parameters()).device
        cond_device = next(self.condition_embedder.parameters()).device

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

        # === Phase 2: Transfer to TT and run blocks in eager mode ===
        from torch_xla.experimental.eager import eager_mode_context

        hidden_states = hidden_states.to(dtype=torch.bfloat16, device=tt_device)
        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.bfloat16, device=tt_device)
        timestep_proj = timestep_proj.to(dtype=torch.bfloat16, device=tt_device)
        # rotary_emb is a tuple of tensors from WanRotaryPosEmbed
        if isinstance(rotary_emb, (tuple, list)):
            rotary_emb = tuple(r.to(dtype=torch.bfloat16, device=tt_device) for r in rotary_emb)
        else:
            rotary_emb = rotary_emb.to(dtype=torch.bfloat16, device=tt_device)
        xm.mark_step()

        # 4. Transformer blocks — use eager mode to compile each op separately
        # This avoids the massive graph compilation that takes hours
        with eager_mode_context(True):
            for i, block in enumerate(self.blocks):
                block_start = time.time()
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                block_time = time.time() - block_start
                print(f"      Block {i+1}/{len(self.blocks)} done ({block_time:.1f}s)", flush=True)

        # === Phase 3: Output projection (also in eager mode) ===
        with eager_mode_context(True):
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


class WanT2VConfig:
    """Configuration for Wan2.1-T2V pipeline."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        device: str = "cpu",
        transformer_on_tt: bool = True,
    ):
        self.model_id = model_id
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.device = device
        self.transformer_on_tt = transformer_on_tt


class WanT2VPipeline:
    """
    Manual pipeline for Wan2.1 Text-to-Video generation on TT hardware.
    """

    def __init__(self, config: WanT2VConfig):
        self.config = config
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def load_models(self):
        """Load all model components."""
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.models import WanTransformer3DModel
        from transformers import T5TokenizerFast, UMT5EncoderModel

        print(f"Loading models from {self.config.model_id}...")
        start = time.time()

        # 1. Text encoder (UMT5-XXL) — always on CPU
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

        # 2. VAE — on CPU
        print("  Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to("cpu")
        self.vae.eval()

        # 3. Scheduler
        print("  Loading scheduler...")
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.config.model_id, subfolder="scheduler"
        )

        # 4. Transformer
        print("  Loading transformer (14B)...")
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.config.model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.transformer.eval()

        if self.config.transformer_on_tt:
            # Patch forward to run pre-processing on CPU (avoids tt-mlir reshape crash)
            # and use eager mode for block execution (avoids massive graph compilation)
            patch_transformer_with_graph_breaks(self.transformer)
            print("  Moving transformer to XLA device...")
            self.transformer = self.transformer.to(xm.xla_device())

        elapsed = time.time() - start
        print(f"Models loaded in {elapsed:.1f}s")

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        max_sequence_length: int = 512,
    ):
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

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        num_frames: int,
        generator: Optional[torch.Generator] = None,
    ):
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
        """Generate a video from text prompt."""
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width

        # 1. Encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt
        )

        # 2. Prepare latents
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        num_channels = 16
        latents = self.prepare_latents(
            batch_size=1,
            num_channels=num_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
        )

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        timesteps = self.scheduler.timesteps

        latents = latents * self.scheduler.init_noise_sigma

        mask = torch.ones_like(latents)

        tt_device = xm.xla_device() if self.config.transformer_on_tt else "cpu"
        tt_cast = lambda x: x.to(dtype=torch.bfloat16, device=tt_device)
        cpu_cast = lambda x: x.to(device="cpu", dtype=torch.float32)

        # 4. Denoising loop
        print(f"Running denoising loop ({num_inference_steps} steps)...", flush=True)
        loop_start = time.time()

        for i, t in enumerate(timesteps):
            step_start = time.time()
            print(f"  Step {i+1}/{num_inference_steps} (t={t.item():.2f})", flush=True)

            latent_model_input = latents.to(dtype=torch.bfloat16)

            # Wan uses expanded timesteps
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

            # Move inputs to TT device
            latent_input_tt = tt_cast(latent_model_input)
            timestep_tt = tt_cast(timestep)
            prompt_embeds_tt = tt_cast(prompt_embeds)

            # Forward pass through patched transformer (with graph breaks)
            noise_pred = self.transformer(
                hidden_states=latent_input_tt,
                timestep=timestep_tt,
                encoder_hidden_states=prompt_embeds_tt,
            )
            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample
            noise_pred = cpu_cast(noise_pred)

            # Classifier-free guidance
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

                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            step_time = time.time() - step_start
            print(f"    Step took {step_time:.2f}s", flush=True)

        loop_time = time.time() - loop_start
        print(f"Denoising complete in {loop_time:.1f}s ({loop_time/num_inference_steps:.1f}s/step)")

        # 5. VAE decode
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

        # 6. Post-process
        video = video.float().cpu()
        video = (video.clamp(-1, 1) + 1) / 2 * 255
        video = video[0].permute(1, 2, 3, 0).to(torch.uint8).numpy()

        return video


def run_wan_pipeline(
    prompt: str = "A serene lake surrounded by mountains at sunset",
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    seed: Optional[int] = 42,
    output_path: str = "output_video.mp4",
    optimization_level: int = 1,
):
    """Run the Wan T2V pipeline end-to-end."""
    torch_xla.set_custom_compile_options({"optimization_level": optimization_level})

    config = WanT2VConfig(
        height=height,
        width=width,
        num_frames=num_frames,
        device="cpu",
        transformer_on_tt=True,
    )
    pipeline = WanT2VPipeline(config)
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


def test_wan_t2v_pipeline():
    """Test Wan T2V pipeline generates valid output."""
    xr.set_device_type("TT")

    output_path = "test_wan_output.mp4"
    if Path(output_path).exists():
        Path(output_path).unlink()

    try:
        run_wan_pipeline(
            prompt="a cat",
            num_inference_steps=1,
            height=480,
            width=832,
            num_frames=17,
            output_path=output_path,
        )
        assert Path(output_path).exists(), f"Output video {output_path} was not created"
        print(f"Test passed: video created at {output_path}")
    finally:
        if Path(output_path).exists():
            Path(output_path).unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-14B on Tenstorrent Blackhole")
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
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    xr.set_device_type("TT")

    os.makedirs(args.output_dir, exist_ok=True)

    filename = args.prompt[:50].lower().replace(" ", "_")
    filename = "".join(c if c.isalnum() or c == "_" else "" for c in filename)
    output_path = os.path.join(args.output_dir, f"{filename}.mp4")

    run_wan_pipeline(
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
