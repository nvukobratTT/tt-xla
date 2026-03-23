# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi T2V experiments:
1. opt_level=1 at various dims (perf comparison vs opt_level=0)
2. Output quality validation at 64x64/3f with multiple steps
3. BFP8 weight investigation — softmax constraint analysis

Run inside tt-xla-dev container.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def run_experiment(name, opt_level, height, width, num_frames, steps, guidance_scale, seed, 
                   bfp8=False, prompt="a cat walking on a beach"):
    """Run a single Mochi experiment and return results dict."""
    import torch_xla
    import torch_xla.runtime as xr

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  opt_level={opt_level}, dims={height}x{width}, frames={num_frames}")
    print(f"  steps={steps}, CFG={guidance_scale}, bfp8={bfp8}")
    print(f"{'='*60}\n")

    compile_opts = {"optimization_level": opt_level}
    if bfp8:
        compile_opts["experimental_weight_dtype"] = "bfp8"
    torch_xla.set_custom_compile_options(compile_opts)

    from mochi_t2v_tp import MochiT2VTPPipeline, export_to_video

    result = {
        "name": name,
        "opt_level": opt_level,
        "height": height,
        "width": width, 
        "num_frames": num_frames,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "bfp8": bfp8,
        "prompt": prompt,
    }

    try:
        pipe = MochiT2VTPPipeline(height=height, width=width, num_frames=num_frames)
        
        load_start = time.time()
        pipe.load_models()
        result["load_time"] = time.time() - load_start

        gen_start = time.time()
        video = pipe.generate(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        result["gen_time"] = time.time() - gen_start
        result["total_time"] = result["load_time"] + result["gen_time"]
        result["video_shape"] = list(video.shape)
        result["status"] = "PASS"

        # Save video
        out_dir = Path("generated_videos/experiments")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{name}.mp4")
        export_to_video(video, out_path)
        result["output_path"] = out_path
        print(f"\n✅ {name}: PASS — {video.shape[0]} frames @ {video.shape[1]}x{video.shape[2]}, gen={result['gen_time']:.1f}s")

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"\n❌ {name}: FAIL — {e}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["opt1_48x48_3f", "opt1_32x32_7f", "opt1_64x64_3f",
                                "quality_64x64", "bfp8_32x32_3f", "bfp8_48x48_3f",
                                "all_opt1", "all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch_xla.runtime as xr
    xr.set_device_type("TT")

    results = []

    if args.experiment in ("opt1_48x48_3f", "all_opt1", "all"):
        results.append(run_experiment(
            "opt1_48x48_3f", opt_level=1, height=48, width=48, num_frames=3,
            steps=1, guidance_scale=1.0, seed=args.seed,
        ))

    if args.experiment in ("opt1_32x32_7f", "all_opt1", "all"):
        results.append(run_experiment(
            "opt1_32x32_7f", opt_level=1, height=32, width=32, num_frames=7,
            steps=1, guidance_scale=1.0, seed=args.seed,
        ))

    if args.experiment in ("opt1_64x64_3f", "all_opt1", "all"):
        results.append(run_experiment(
            "opt1_64x64_3f", opt_level=1, height=64, width=64, num_frames=3,
            steps=1, guidance_scale=1.0, seed=args.seed,
        ))

    if args.experiment in ("quality_64x64", "all"):
        results.append(run_experiment(
            "quality_64x64_28step", opt_level=0, height=64, width=64, num_frames=3,
            steps=28, guidance_scale=4.5, seed=args.seed,
            prompt="a cat walking on a beach, high quality, cinematic",
        ))

    if args.experiment in ("bfp8_32x32_3f", "all"):
        results.append(run_experiment(
            "bfp8_32x32_3f", opt_level=0, height=32, width=32, num_frames=3,
            steps=1, guidance_scale=1.0, seed=args.seed, bfp8=True,
        ))

    if args.experiment in ("bfp8_48x48_3f", "all"):
        results.append(run_experiment(
            "bfp8_48x48_3f", opt_level=0, height=48, width=48, num_frames=3,
            steps=1, guidance_scale=1.0, seed=args.seed, bfp8=True,
        ))

    # Save results
    out_dir = Path("generated_videos/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = str(out_dir / f"results_{args.experiment}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = r["status"]
        if status == "PASS":
            print(f"  ✅ {r['name']}: gen={r['gen_time']:.1f}s, total={r['total_time']:.1f}s, shape={r['video_shape']}")
        else:
            print(f"  ❌ {r['name']}: {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
