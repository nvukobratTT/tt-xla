# Continued experiments: 64x64/7f retry, 96x96/3f, quality validation
import os, time, json, traceback, sys, signal
import numpy as np
import torch
import torch_xla
import torch_xla.runtime as xr
from pathlib import Path

xr.set_device_type("TT")

results = []

def run_one(name, opt_level, height, width, num_frames, steps, guidance_scale, seed, prompt, timeout_s=300):
    torch_xla.set_custom_compile_options({"optimization_level": opt_level})
    from mochi_t2v_tp import MochiT2VTPPipeline, export_to_video
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  opt={opt_level}, {height}x{width}, {num_frames}f, {steps} steps, CFG={guidance_scale}")
    print(f"  timeout={timeout_s}s")
    print(f"{'='*60}\n")
    
    result = {"name": name, "opt_level": opt_level, "height": height, "width": width,
              "num_frames": num_frames, "steps": steps}
    
    try:
        pipe = MochiT2VTPPipeline(height=height, width=width, num_frames=num_frames)
        pipe.load_models()
        
        # Set alarm for timeout
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout_s}s")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_s)
        
        start = time.time()
        video = pipe.generate(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, seed=seed)
        elapsed = time.time() - start
        signal.alarm(0)  # Cancel alarm
        
        result["gen_time"] = elapsed
        result["shape"] = list(video.shape)
        result["status"] = "PASS"
        
        out_dir = Path("generated_videos/experiments")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{name}.mp4")
        export_to_video(video, out_path)
        result["output_path"] = out_path
        print(f"\n✅ {name}: PASS — {elapsed:.1f}s, shape={video.shape}")
        
    except TimeoutError as e:
        signal.alarm(0)
        result["status"] = "TIMEOUT"
        result["error"] = str(e)
        print(f"\n⏰ {name}: TIMEOUT — {e}")
    except Exception as e:
        signal.alarm(0)
        result["status"] = "FAIL"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"\n❌ {name}: FAIL — {e}")
    
    results.append(result)
    
    # Save after each test
    with open("generated_videos/experiments/results_continued.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    try:
        del pipe
        import gc; gc.collect()
    except:
        pass

test_idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1

if test_idx in (0, -1):
    # Test 0: 64x64/7f retry with 5min timeout
    run_one("opt1_64x64_7f_retry", opt_level=1, height=64, width=64, num_frames=7,
            steps=1, guidance_scale=1.0, seed=42, prompt="a cat walking on a beach", timeout_s=300)

if test_idx in (1, -1):
    # Test 1: 96x96/3f — bigger spatial
    run_one("opt1_96x96_3f", opt_level=1, height=96, width=96, num_frames=3,
            steps=1, guidance_scale=1.0, seed=42, prompt="a cat walking on a beach", timeout_s=300)

if test_idx in (2, -1):
    # Test 2: Quality validation — 64x64/3f with 28 steps + CFG
    run_one("quality_64x64_28step", opt_level=1, height=64, width=64, num_frames=3,
            steps=28, guidance_scale=4.5, seed=42, 
            prompt="a cat walking on a beach, high quality, cinematic",
            timeout_s=600)

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
for r in results:
    s = r["status"]
    if s == "PASS":
        print(f"  ✅ {r['name']}: {r['gen_time']:.1f}s, shape={r['shape']}")
    elif s == "TIMEOUT":
        print(f"  ⏰ {r['name']}: {r['error']}")
    else:
        print(f"  ❌ {r['name']}: {r.get('error', 'unknown')[:100]}")
