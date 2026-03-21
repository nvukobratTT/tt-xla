# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi 1 Text-to-Video Pipeline with Tensor Parallelism across 4 Blackhole chips.

Clean approach: entire transformer compiled and executed on TT device.
No CPU preprocessing workarounds, no graph breaks between blocks.

Strategy: Megatron-style TP for the ~10B transformer.
- 24 attention heads / 4 devices = 6 heads per device
- QKV projections: column-parallel (shard output dim)
- Output projection: row-parallel (shard input dim) + all-reduce
- FFN up: column-parallel
- FFN down: row-parallel + all-reduce

Architecture:
- 48 MochiTransformerBlocks with joint attention (video + text tokens)
- inner_dim = 24 * 128 = 3072
- FFN inner_dim = (4 * 3072 * 2) / 3 = 8192
- pooled_projection_dim = 1536 (context/text stream)
"""

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Optional

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


def apply_tp_sharding_mochi_block(block, mesh):
    """
    Apply Megatron-style tensor parallel sharding to a single MochiTransformerBlock.

    MochiAttention structure:
        to_q, to_k, to_v [3072, 3072] — no bias
        add_q_proj, add_k_proj, add_v_proj [3072, 1536] — no bias (context stream)
        to_out.0 [3072, 3072] — with bias
        to_add_out [1536, 3072] — with bias (except last block)

    FFN (SwiGLU):
        ff.net.0.proj [8192, 3072] — no bias
        ff.net.2 [3072, 8192] — no bias
        ff_context same structure but 1536/4096 dims (except last block)
    """
    shard_specs = {}

    # --- Joint Attention (attn1) ---
    # Video stream Q/K/V — column-parallel
    shard_specs[block.attn1.to_q.weight] = ("model", None)
    shard_specs[block.attn1.to_k.weight] = ("model", None)
    shard_specs[block.attn1.to_v.weight] = ("model", None)

    # Context stream Q/K/V — column-parallel
    if hasattr(block.attn1, 'add_q_proj') and block.attn1.add_q_proj is not None:
        shard_specs[block.attn1.add_q_proj.weight] = ("model", None)
    if hasattr(block.attn1, 'add_k_proj') and block.attn1.add_k_proj is not None:
        shard_specs[block.attn1.add_k_proj.weight] = ("model", None)
    if hasattr(block.attn1, 'add_v_proj') and block.attn1.add_v_proj is not None:
        shard_specs[block.attn1.add_v_proj.weight] = ("model", None)

    # Output projection — row-parallel
    shard_specs[block.attn1.to_out[0].weight] = (None, "model")
    if block.attn1.to_out[0].bias is not None:
        shard_specs[block.attn1.to_out[0].bias] = (None,)

    # Context output projection (all blocks except last)
    if hasattr(block.attn1, 'to_add_out') and block.attn1.to_add_out is not None:
        shard_specs[block.attn1.to_add_out.weight] = (None, "model")
        if block.attn1.to_add_out.bias is not None:
            shard_specs[block.attn1.to_add_out.bias] = (None,)

    # --- Video FFN (SwiGLU) ---
    shard_specs[block.ff.net[0].proj.weight] = ("model", None)  # up — column-parallel
    if hasattr(block.ff.net[0].proj, 'bias') and block.ff.net[0].proj.bias is not None:
        shard_specs[block.ff.net[0].proj.bias] = ("model",)
    shard_specs[block.ff.net[2].weight] = (None, "model")  # down — row-parallel
    if hasattr(block.ff.net[2], 'bias') and block.ff.net[2].bias is not None:
        shard_specs[block.ff.net[2].bias] = (None,)

    # --- Context FFN (all blocks except last) ---
    if block.ff_context is not None:
        shard_specs[block.ff_context.net[0].proj.weight] = ("model", None)
        if hasattr(block.ff_context.net[0].proj, 'bias') and block.ff_context.net[0].proj.bias is not None:
            shard_specs[block.ff_context.net[0].proj.bias] = ("model",)
        shard_specs[block.ff_context.net[2].weight] = (None, "model")
        if hasattr(block.ff_context.net[2], 'bias') and block.ff_context.net[2].bias is not None:
            shard_specs[block.ff_context.net[2].bias] = (None,)

    return shard_specs


def apply_tp_sharding_mochi_transformer(transformer, mesh):
    """Apply tensor-parallel sharding to all 48 blocks of MochiTransformer3DModel."""
    all_specs = {}

    for i, block in enumerate(transformer.transformer_blocks):
        block_specs = apply_tp_sharding_mochi_block(block, mesh)
        all_specs.update(block_specs)

    for tensor, spec in all_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    print(f"  Applied TP sharding to {len(transformer.transformer_blocks)} blocks ({len(all_specs)} tensors)")
    return all_specs


class MochiT2VTPPipeline:
    """
    Mochi 1 T2V pipeline with tensor parallelism.

    Components:
    - Text encoder (T5-XXL): CPU
    - Transformer (~10B): TT device, compiled, 4-chip TP
    - VAE decoder: CPU
    - Scheduler: CPU
    """

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

        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.mesh = None
        self.num_devices = None

    def setup_spmd(self):
        """Initialize SPMD mesh for tensor parallelism."""
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()

        self.num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(self.num_devices))
        self.mesh = Mesh(device_ids, (1, self.num_devices), ("batch", "model"))

        print(f"SPMD enabled with {self.num_devices} devices")
        print(f"Mesh: {self.mesh}")

        num_heads = 24
        assert num_heads % self.num_devices == 0, \
            f"Heads ({num_heads}) must be divisible by devices ({self.num_devices})"
        print(f"  {num_heads} heads / {self.num_devices} devices = {num_heads // self.num_devices} heads/device")

    def load_models(self):
        """Load all model components."""
        from diffusers import AutoencoderKLMochi, FlowMatchEulerDiscreteScheduler
        from diffusers.models import MochiTransformer3DModel
        from transformers import T5EncoderModel, T5TokenizerFast

        self.setup_spmd()

        print(f"\nLoading models from {self.model_id}...")
        start = time.time()

        # Text encoder — CPU
        print("  Loading text encoder (T5-XXL)...")
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        ).to("cpu").eval()

        # VAE — CPU
        print("  Loading VAE...")
        self.vae = AutoencoderKLMochi.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=torch.float32,
        ).to("cpu").eval()
        self.vae.enable_tiling()

        # Scheduler
        print("  Loading scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

        # Transformer — load, move to XLA, shard, compile
        print("  Loading transformer (~10B)...")
        self.transformer = MochiTransformer3DModel.from_pretrained(
            self.model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        ).eval()

        print("  Moving transformer to XLA device...")
        device = torch_xla.device()
        self.transformer = self.transformer.to(device)

        # Patch attention processor for TT compatibility
        # (removes torch.nonzero, dynamic indexing, strided slicing)
        from mochi_tt_compat import patch_mochi_for_tt
        patch_mochi_for_tt(self.transformer)

        print("  Applying tensor-parallel sharding...")
        apply_tp_sharding_mochi_transformer(self.transformer, self.mesh)

        elapsed = time.time() - start
        print(f"Models loaded in {elapsed:.1f}s")

    def encode_prompt(self, prompt: str, negative_prompt: str = "", max_seq_len: int = 256):
        """Encode text prompt using T5-XXL on CPU."""
        print("Encoding prompt...")
        inputs = self.tokenizer(
            prompt, padding="max_length", max_length=max_seq_len,
            truncation=True, return_tensors="pt",
        )

        with torch.no_grad():
            embeds = self.text_encoder(
                inputs.input_ids.to("cpu"),
                attention_mask=inputs.attention_mask.to("cpu"),
            )[0].to(dtype=torch.bfloat16)

        mask = inputs.attention_mask.to(dtype=torch.bfloat16)

        neg_embeds, neg_mask = None, None
        if negative_prompt:
            neg_inputs = self.tokenizer(
                negative_prompt, padding="max_length", max_length=max_seq_len,
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                neg_embeds = self.text_encoder(
                    neg_inputs.input_ids.to("cpu"),
                    attention_mask=neg_inputs.attention_mask.to("cpu"),
                )[0].to(dtype=torch.bfloat16)
            neg_mask = neg_inputs.attention_mask.to(dtype=torch.bfloat16)

        return embeds, mask, neg_embeds, neg_mask

    def prepare_latents(self, height, width, num_frames, generator=None):
        """Prepare random latent noise matching Mochi's VAE compression."""
        latent_frames = (num_frames - 1) // 6 + 1  # temporal 6x
        latent_h = height // 8  # spatial 8x
        latent_w = width // 8

        latents = torch.randn(
            1, 12, latent_frames, latent_h, latent_w,
            generator=generator, dtype=torch.bfloat16, device="cpu",
        )
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
        """Generate video from text prompt."""
        num_frames = num_frames or self.num_frames
        height = height or self.height
        width = width or self.width

        embeds, mask, neg_embeds, neg_mask = self.encode_prompt(prompt, negative_prompt)

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        latents = self.prepare_latents(height, width, num_frames, generator)

        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        timesteps = self.scheduler.timesteps
        # FlowMatch scheduler doesn't use init_noise_sigma — latents are used as-is

        device = torch_xla.device()
        do_cfg = guidance_scale > 1.0 and neg_embeds is not None

        print(f"\nDenoising ({num_inference_steps} steps, CFG={do_cfg})...", flush=True)
        loop_start = time.time()

        for i, t in enumerate(timesteps):
            step_start = time.time()

            timestep = t.unsqueeze(0).to(torch.long)

            # Forward pass — whole model on TT
            noise_pred = self.transformer(
                hidden_states=latents.to(device),
                encoder_hidden_states=embeds.to(device),
                timestep=timestep.to(device),
                encoder_attention_mask=mask.to(device),
            )
            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample
            noise_pred = noise_pred.to("cpu", dtype=torch.float32)

            # CFG
            if do_cfg:
                noise_pred_uncond = self.transformer(
                    hidden_states=latents.to(device),
                    encoder_hidden_states=neg_embeds.to(device),
                    timestep=timestep.to(device),
                    encoder_attention_mask=neg_mask.to(device),
                )
                if hasattr(noise_pred_uncond, "sample"):
                    noise_pred_uncond = noise_pred_uncond.sample
                noise_pred_uncond = noise_pred_uncond.to("cpu", dtype=torch.float32)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents.float(), return_dict=False)[0]
            latents = latents.to(torch.bfloat16)

            step_time = time.time() - step_start
            print(f"  Step {i+1}/{num_inference_steps} (t={t.item():.2f}) — {step_time:.2f}s", flush=True)

        loop_time = time.time() - loop_start
        print(f"Denoising done in {loop_time:.1f}s ({loop_time/num_inference_steps:.1f}s/step)")

        # VAE decode on CPU
        print("Decoding with VAE...", flush=True)
        decode_start = time.time()
        video = self.vae.decode(latents.float(), return_dict=False)[0]
        print(f"VAE decode: {time.time() - decode_start:.1f}s")

        # Post-process to uint8 frames
        video = (video.float().cpu().clamp(-1, 1) + 1) / 2 * 255
        video = video[0].permute(1, 2, 3, 0).to(torch.uint8).numpy()
        return video


def test_mochi_t2v_tp():
    """Minimal test: 1 step at small resolution to verify compilation."""
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    pipe = MochiT2VTPPipeline(height=64, width=64, num_frames=7)
    pipe.load_models()

    video = pipe.generate(
        prompt="a cat",
        num_inference_steps=1,
        guidance_scale=1.0,
        seed=42,
    )

    output_path = "test_mochi_tp.mp4"
    export_to_video(video, output_path)
    assert Path(output_path).exists()
    print(f"Test passed: {video.shape[0]} frames at {video.shape[1]}x{video.shape[2]}")
    Path(output_path).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mochi T2V with Tensor Parallelism")
    parser.add_argument("--prompt", type=str, default="A serene lake surrounded by mountains at sunset")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="generated_videos")
    parser.add_argument("--optimization_level", type=int, default=1)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": args.optimization_level})

    if args.test:
        test_mochi_t2v_tp()
    else:
        os.makedirs(args.output_dir, exist_ok=True)

        pipe = MochiT2VTPPipeline(
            height=args.height, width=args.width, num_frames=args.num_frames,
        )
        pipe.load_models()

        video = pipe.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )

        filename = args.prompt[:50].lower().replace(" ", "_")
        filename = "".join(c if c.isalnum() or c == "_" else "" for c in filename)
        output_path = os.path.join(args.output_dir, f"{filename}_mochi_tp.mp4")

        print(f"Generated {video.shape[0]} frames at {video.shape[1]}x{video.shape[2]}")
        export_to_video(video, output_path)
        print(f"Video saved to: {output_path}")
