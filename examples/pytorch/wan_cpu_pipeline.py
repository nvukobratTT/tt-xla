"""
Wan2.1-T2V-1.3B: Full CPU pipeline to verify correct video output.
This validates the model + pipeline produces working video, proving
any quality issues are purely from TT device precision.
"""

import argparse
import os
import time
import types
import numpy as np
import torch
import torch_xla
import torch_xla.runtime as xr

# Need to init XLA even for CPU-only (model loaded with XLA context)
xr.set_device_type("TT")


def patch_rope_for_tt():
    import diffusers.models.transformers.transformer_wan as wan_module

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


def patch_transformer_cpu_forward(transformer):
    """CPU-only forward pass."""
    from diffusers.models.transformers.transformer_wan import Transformer2DModelOutput

    def forward_cpu(
        self,
        hidden_states, timestep, encoder_hidden_states,
        encoder_hidden_states_image=None, return_dict=True, attention_kwargs=None,
    ):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Everything on CPU
        self.to("cpu")
        hidden_states = hidden_states.to("cpu")
        timestep = timestep.to("cpu")
        encoder_hidden_states = encoder_hidden_states.to("cpu")

        with torch.no_grad():
            rotary_emb = self.rope(hidden_states)
            hidden_states = self.patch_embedding(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)

            if timestep.ndim == 2:
                ts_seq_len = timestep.shape[1]
                timestep_flat = timestep.flatten()
            else:
                ts_seq_len = None
                timestep_flat = timestep

            temb, timestep_proj, encoder_hidden_states, enc_img = self.condition_embedder(
                timestep_flat, encoder_hidden_states,
                encoder_hidden_states_image.to("cpu") if encoder_hidden_states_image is not None else None,
                timestep_seq_len=ts_seq_len,
            )
            if ts_seq_len is not None:
                timestep_proj = timestep_proj.unflatten(2, (6, -1))
            else:
                timestep_proj = timestep_proj.unflatten(1, (6, -1))

            if enc_img is not None:
                encoder_hidden_states = torch.concat([enc_img, encoder_hidden_states], dim=1)

        hidden_states = hidden_states.to(dtype=torch.bfloat16)
        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.bfloat16)
        timestep_proj = timestep_proj.to(dtype=torch.bfloat16)
        rotary_emb = tuple(r.to(dtype=torch.bfloat16) for r in rotary_emb)

        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            if (i + 1) % 10 == 0:
                print(f"      Block {i+1}/{len(self.blocks)} done", flush=True)

        # Output projection
        temb = temb.to(dtype=torch.bfloat16)
        sst = self.scale_shift_table.data
        if temb.ndim == 3:
            shift, scale = (sst.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (sst + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).to(torch.bfloat16)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    transformer.forward = types.MethodType(forward_cpu, transformer)


def export_to_video(frames, output_path, fps=16):
    try:
        from diffusers.utils import export_to_video as dexp
        dexp(list(frames), output_path, fps=fps)
        return
    except Exception:
        pass
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except ImportError:
        # Save frames as images
        from PIL import Image
        os.makedirs(output_path + "_frames", exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(f"{output_path}_frames/frame_{i:04d}.png")
        print(f"Saved {len(frames)} frames to {output_path}_frames/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A cat walking in a garden")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output_cpu.mp4")
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from diffusers.models import WanTransformer3DModel
    from transformers import T5TokenizerFast, UMT5EncoderModel

    MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    print("Loading models...")
    start = time.time()

    tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).to("cpu").eval()

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
    ).to("cpu").eval()

    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()

    patch_rope_for_tt()
    patch_transformer_cpu_forward(transformer)
    transformer.to("cpu")
    print(f"Models loaded in {time.time() - start:.1f}s")

    # Encode prompt
    print("Encoding prompt...")
    text_inputs = tokenizer(
        args.prompt, padding="max_length", max_length=512,
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_inputs.input_ids, attention_mask=text_inputs.attention_mask,
        )[0].to(dtype=torch.bfloat16)

    neg_inputs = tokenizer(
        "", padding="max_length", max_length=512,
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )
    with torch.no_grad():
        neg_embeds = text_encoder(
            neg_inputs.input_ids, attention_mask=neg_inputs.attention_mask,
        )[0].to(dtype=torch.bfloat16)

    # Prepare latents
    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_h = args.height // 8
    latent_w = args.width // 8
    shape = (1, 16, latent_frames, latent_h, latent_w)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")

    scheduler.set_timesteps(args.steps, device="cpu")
    timesteps = scheduler.timesteps
    latents = latents * scheduler.init_noise_sigma

    # Denoising loop
    print(f"Running {args.steps}-step denoising ({args.height}x{args.width}x{args.num_frames})...")
    loop_start = time.time()

    for i, t in enumerate(timesteps):
        step_start = time.time()
        print(f"  Step {i+1}/{args.steps} (t={t.item():.2f})", flush=True)

        latent_input = latents.to(dtype=torch.bfloat16)
        ts = t.expand(1)

        noise_pred = transformer(
            hidden_states=latent_input, timestep=ts,
            encoder_hidden_states=prompt_embeds,
        )
        if hasattr(noise_pred, "sample"):
            noise_pred = noise_pred.sample
        noise_pred = noise_pred.float()

        if args.guidance_scale > 1.0:
            noise_pred_uncond = transformer(
                hidden_states=latent_input, timestep=ts,
                encoder_hidden_states=neg_embeds,
            )
            if hasattr(noise_pred_uncond, "sample"):
                noise_pred_uncond = noise_pred_uncond.sample
            noise_pred_uncond = noise_pred_uncond.float()
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        print(f"    Step took {time.time() - step_start:.2f}s", flush=True)

    loop_time = time.time() - loop_start
    print(f"Denoising done in {loop_time:.1f}s ({loop_time / args.steps:.1f}s/step)")

    # VAE decode
    print("VAE decoding...")
    vae_start = time.time()
    latents_decode = latents.to(dtype=vae.dtype)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents_decode.device, latents_decode.dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents_decode.device, latents_decode.dtype)
    latents_decode = latents_decode / latents_std + latents_mean
    video = vae.decode(latents_decode, return_dict=False)[0]
    print(f"VAE decode took {time.time() - vae_start:.1f}s")

    video = video.float().cpu()
    video = (video.clamp(-1, 1) + 1) / 2 * 255
    frames = video[0].permute(1, 2, 3, 0).to(torch.uint8).numpy()

    print(f"Output: {frames.shape[0]} frames at {frames.shape[1]}x{frames.shape[2]}")
    export_to_video(frames, args.output, fps=16)
    print(f"Video saved: {args.output}")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
