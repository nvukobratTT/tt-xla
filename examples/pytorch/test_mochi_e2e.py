#!/usr/bin/env python3
"""Mochi T2V end-to-end: text → transformer on TT → VAE on CPU → video."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

xr.set_device_type("TT")
torch_xla.set_custom_compile_options({"optimization_level": 0})
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
num_devices = xr.global_runtime_device_count()
mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
device = torch_xla.device()

from diffusers import AutoencoderKLMochi, FlowMatchEulerDiscreteScheduler
from diffusers.models import MochiTransformer3DModel
from transformers import T5EncoderModel, T5TokenizerFast
from mochi_tt_compat import MochiAttnProcessorTT
from mochi_t2v_tp import apply_tp_sharding_mochi_transformer

MODEL_ID = "genmo/mochi-1-preview"
# Small dims that fit DRAM
HEIGHT, WIDTH, NUM_FRAMES = 16, 16, 3
NUM_STEPS = 1  # Start with 1 step

print("=== Loading models ===", flush=True)
start = time.time()

# T5 text encoder — CPU
print("  Text encoder...", flush=True)
tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(
    MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cpu").eval()

# VAE — CPU
print("  VAE...", flush=True)
vae = AutoencoderKLMochi.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=torch.float32).to("cpu").eval()
vae.enable_tiling()

# Scheduler
print("  Scheduler...", flush=True)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# Transformer — TT device
print("  Transformer...", flush=True)
transformer = MochiTransformer3DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16).eval()
for block in transformer.transformer_blocks:
    block.attn1.processor = MochiAttnProcessorTT()
transformer = transformer.to(device)
apply_tp_sharding_mochi_transformer(transformer, mesh)
compiled_transformer = torch.compile(transformer, backend="tt")

print(f"Models loaded in {time.time()-start:.1f}s", flush=True)

# === Encode prompt ===
print("\n=== Encoding prompt ===", flush=True)
prompt = "a cat walking on a beach"
inputs = tokenizer(prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
with torch.no_grad():
    prompt_embeds = text_encoder(inputs.input_ids, attention_mask=inputs.attention_mask)[0]
prompt_embeds = prompt_embeds.to(torch.bfloat16)
attention_mask = inputs.attention_mask.to(torch.bfloat16)
print(f"  Prompt embeddings: {prompt_embeds.shape}", flush=True)

# === Prepare latents ===
latent_frames = (NUM_FRAMES - 1) // 6 + 1
latent_h = HEIGHT // 8
latent_w = WIDTH // 8
# Mochi needs spatial dims divisible by patch_size=2
# 16//8=2, which is fine
generator = torch.Generator("cpu").manual_seed(42)
latents = torch.randn(1, 12, latent_frames, latent_h, latent_w, generator=generator, dtype=torch.bfloat16)
print(f"  Latents: {latents.shape}", flush=True)

# === Denoising loop ===
print(f"\n=== Denoising ({NUM_STEPS} steps) ===", flush=True)
scheduler.set_timesteps(NUM_STEPS, device="cpu")

for i, t in enumerate(scheduler.timesteps):
    step_start = time.time()
    timestep = t.unsqueeze(0).long()

    # Run transformer on TT
    with torch.no_grad():
        noise_pred = compiled_transformer(
            hidden_states=latents.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            timestep=timestep.to(device),
            encoder_attention_mask=attention_mask.to(device),
        )
    if hasattr(noise_pred, "sample"):
        noise_pred = noise_pred.sample
    noise_pred = noise_pred.cpu().float()

    # Scheduler step
    latents = scheduler.step(noise_pred, t, latents.float(), return_dict=False)[0]
    latents = latents.to(torch.bfloat16)

    print(f"  Step {i+1}/{NUM_STEPS}: {time.time()-step_start:.1f}s", flush=True)

# === VAE decode ===
print("\n=== VAE decode ===", flush=True)
with torch.no_grad():
    video = vae.decode(latents.float(), return_dict=False)[0]
video = (video.float().clamp(-1, 1) + 1) / 2 * 255
video = video[0].permute(1, 2, 3, 0).to(torch.uint8).numpy()
print(f"  Video: {video.shape} (frames, H, W, C)", flush=True)

# === Export ===
output_path = "mochi_e2e_test.mp4"
try:
    from diffusers.utils import export_to_video
    export_to_video(list(video), output_path, fps=24)
    print(f"\nVideo saved to {output_path}", flush=True)
except Exception as e:
    print(f"\nVideo export failed (expected with tiny dims): {e}", flush=True)
    # Save frames as images instead
    from PIL import Image
    os.makedirs("mochi_e2e_frames", exist_ok=True)
    for i, frame in enumerate(video):
        Image.fromarray(frame).save(f"mochi_e2e_frames/frame_{i:03d}.png")
    print(f"  Saved {len(video)} frames to mochi_e2e_frames/", flush=True)

print("\n=== E2E PIPELINE SUCCESS ===", flush=True)
