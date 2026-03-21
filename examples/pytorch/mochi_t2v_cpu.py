# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi 1 Text-to-Video CPU Baseline.

Runs the full MochiPipeline on CPU to establish reference output
for later comparison with TT hardware execution.

Model: genmo/mochi-1-preview (~10B transformer + VAE + T5-XXL)
Output: reference video + saved latents for PCC comparison.
"""

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch


def export_to_video(frames, output_path: str, fps: int = 24):
    """Export frames to video file."""
    try:
        from diffusers.utils import export_to_video as diffusers_export
        if isinstance(frames, np.ndarray):
            frames = list(frames)
        diffusers_export(frames, output_path, fps=fps)
        return
    except Exception:
        pass

    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.numpy()
            writer.append_data(frame)
        writer.close()
        return
    except ImportError:
        pass

    raise RuntimeError("No video export backend available. Install imageio-ffmpeg.")


def run_mochi_cpu_baseline(
    prompt: str = "A serene lake surrounded by mountains at sunset",
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    height: int = 480,
    width: int = 848,
    num_frames: int = 19,  # Mochi default, generates ~0.8s video at 24fps
    seed: int = 42,
    output_dir: str = "mochi_reference",
    save_latents: bool = True,
):
    """Run Mochi pipeline on CPU and save reference outputs."""
    from diffusers import MochiPipeline
    from diffusers.utils import export_to_video as diffusers_export

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Mochi pipeline...")
    start = time.time()
    pipe = MochiPipeline.from_pretrained(
        "genmo/mochi-1-preview",
        torch_dtype=torch.bfloat16,
    )
    load_time = time.time() - start
    print(f"Pipeline loaded in {load_time:.1f}s")

    # Enable memory-efficient features for CPU
    pipe.enable_vae_tiling()

    # Move to CPU explicitly
    pipe = pipe.to("cpu")

    print(f"\nGenerating video:")
    print(f"  Prompt: {prompt}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Seed: {seed}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    gen_start = time.time()
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    gen_time = time.time() - gen_start

    frames = output.frames[0]
    print(f"\nGeneration complete in {gen_time:.1f}s ({gen_time/num_inference_steps:.1f}s/step)")
    print(f"Output: {len(frames)} frames")

    # Save video
    video_path = os.path.join(output_dir, "mochi_cpu_reference.mp4")
    diffusers_export(frames, video_path, fps=24)
    print(f"Video saved to: {video_path}")

    # Save metadata
    metadata = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "seed": seed,
        "generation_time_s": gen_time,
        "load_time_s": load_time,
        "num_output_frames": len(frames),
    }

    import json
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    return video_path


def run_mochi_cpu_component_test(
    output_dir: str = "mochi_reference",
):
    """
    Run each Mochi component individually on CPU and save intermediate tensors.
    Useful for component-by-component PCC comparison with TT output.
    """
    from diffusers import MochiPipeline
    from diffusers.models import MochiTransformer3DModel, AutoencoderKLMochi
    from transformers import T5EncoderModel, T5TokenizerFast

    os.makedirs(output_dir, exist_ok=True)

    model_id = "genmo/mochi-1-preview"
    prompt = "A serene lake surrounded by mountains at sunset"
    seed = 42

    # 1. Text encoding
    print("=== Text Encoding ===")
    tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to("cpu")
    text_encoder.eval()

    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=256,
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeds = text_encoder(
            text_inputs.input_ids, attention_mask=text_inputs.attention_mask
        )[0]
    print(f"  Text embeddings: {text_embeds.shape} (dtype={text_embeds.dtype})")
    torch.save(text_embeds, os.path.join(output_dir, "text_embeds_cpu.pt"))
    torch.save(text_inputs.attention_mask, os.path.join(output_dir, "attention_mask.pt"))

    del text_encoder
    gc.collect()

    # 2. Transformer single-step test
    print("\n=== Transformer (single step) ===")
    transformer = MochiTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cpu")
    transformer.eval()

    # Create test inputs matching pipeline dimensions
    # For minimum test: small spatial dims
    test_h, test_w = 64, 64  # Minimum viable
    test_frames = 7  # Small number of frames
    latent_channels = 12

    generator = torch.Generator(device="cpu").manual_seed(seed)
    test_latents = torch.randn(
        1, latent_channels, test_frames, test_h // 2, test_w // 2,  # patch_size=2
        generator=generator, dtype=torch.bfloat16
    )
    test_timestep = torch.tensor([500], dtype=torch.long)

    print(f"  Test latents: {test_latents.shape}")
    print(f"  Text embeds: {text_embeds.shape}")

    with torch.no_grad():
        transformer_out = transformer(
            hidden_states=test_latents,
            encoder_hidden_states=text_embeds,
            timestep=test_timestep,
            encoder_attention_mask=text_inputs.attention_mask.to(torch.bfloat16),
        )
    if hasattr(transformer_out, "sample"):
        transformer_out_tensor = transformer_out.sample
    else:
        transformer_out_tensor = transformer_out[0]

    print(f"  Transformer output: {transformer_out_tensor.shape}")
    torch.save(transformer_out_tensor, os.path.join(output_dir, "transformer_out_cpu.pt"))
    torch.save(test_latents, os.path.join(output_dir, "test_latents.pt"))

    # Save transformer config info
    print(f"  Config: {transformer.config.num_attention_heads} heads, "
          f"{transformer.config.attention_head_dim} dim_head, "
          f"{transformer.config.num_layers} layers")

    del transformer
    gc.collect()

    # 3. VAE decoder test
    print("\n=== VAE Decoder ===")
    vae = AutoencoderKLMochi.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    ).to("cpu")
    vae.eval()
    vae.enable_tiling()

    # Small test latent for VAE
    vae_latent = torch.randn(1, 12, 2, 12, 12, dtype=torch.float32)

    with torch.no_grad():
        vae_out = vae.decode(vae_latent).sample
    print(f"  VAE output: {vae_out.shape}")
    torch.save(vae_out, os.path.join(output_dir, "vae_out_cpu.pt"))

    del vae
    gc.collect()

    print(f"\nAll reference tensors saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mochi T2V CPU Baseline")
    parser.add_argument("--prompt", type=str, default="A serene lake surrounded by mountains at sunset")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=19)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="mochi_reference")
    parser.add_argument("--component-test", action="store_true",
                        help="Run component-by-component test instead of full pipeline")

    args = parser.parse_args()

    if args.component_test:
        run_mochi_cpu_component_test(args.output_dir)
    else:
        run_mochi_cpu_baseline(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            output_dir=args.output_dir,
        )
