# Test opt_level=1 at dims that OOM'd with opt_level=0
import os, time, json, traceback, sys
import numpy as np
import torch
import torch_xla
import torch_xla.runtime as xr

xr.set_device_type("TT")
torch_xla.set_custom_compile_options({"optimization_level": 1})

from mochi_t2v_tp import MochiT2VTPPipeline, export_to_video
from pathlib import Path

configs = [
    # (name, height, width, num_frames) — ascending token count
    ("opt1_96x96_3f", 96, 96, 3),     # 2336 tokens — OOM at opt0
    ("opt1_48x48_7f", 48, 48, 7),     # 1184 tokens — OOM at opt0
    ("opt1_64x64_7f", 64, 64, 7),     # 2080 tokens — OOM at opt0
]

# Allow selecting specific test
if len(sys.argv) > 1:
    idx = int(sys.argv[1])
    configs = [configs[idx]]

results = []
for name, h, w, nf in configs:
    print(f"\n{'='*60}")
    print(f"Testing {name}: {h}x{w}, {nf} frames, opt_level=1")
    print(f"{'='*60}")
    
    try:
        pipe = MochiT2VTPPipeline(height=h, width=w, num_frames=nf)
        pipe.load_models()
        
        start = time.time()
        video = pipe.generate(prompt="a cat walking on a beach", num_inference_steps=1, guidance_scale=1.0, seed=42)
        elapsed = time.time() - start
        
        out_dir = Path("generated_videos/experiments")
        out_dir.mkdir(parents=True, exist_ok=True)
        export_to_video(video, str(out_dir / f"{name}.mp4"))
        
        result = {"name": name, "status": "PASS", "gen_time": elapsed, "shape": list(video.shape)}
        print(f"\n✅ {name}: PASS — {elapsed:.1f}s, shape={video.shape}")
    except Exception as e:
        result = {"name": name, "status": "FAIL", "error": str(e)}
        print(f"\n❌ {name}: FAIL — {e}")
        traceback.print_exc()
    
    results.append(result)
    
    # Clean up for next test
    try:
        del pipe
        import gc; gc.collect()
    except:
        pass

print(f"\n{'='*60}")
print("SUMMARY")
for r in results:
    if r["status"] == "PASS":
        print(f"  ✅ {r['name']}: {r['gen_time']:.1f}s, shape={r['shape']}")
    else:
        print(f"  ❌ {r['name']}: {r['error'][:100]}")

# Save results
with open("generated_videos/experiments/results_opt1_scale.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
