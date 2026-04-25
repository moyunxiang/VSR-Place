"""bigblue2 / bigblue4 memory profiling for the supplement.

Goal: produce evidence that the VSR repair operator is NOT what causes the
80GB OOM on bigblue2 (23,084 macros) and bigblue4 (8,170 macros) -- the
ChipDiffusion attention-GNN backbone is. We separately measure peak VRAM for:

  (a) loading the model + cond  (no sampling)
  (b) sampling step 1            (incremental)
  (c) full guided_sample call    (where it OOMs)
  (d) just running VSR-post on a synthetic baseline -- this should fit fine

Output: results/ispd2005/mem_profile.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion" / "diffusion"))
sys.path.insert(0, str(REPO / "scripts"))

_mp = types.ModuleType("moviepy")
_mp_ed = types.ModuleType("moviepy.editor")
_mp_ed.ImageSequenceClip = lambda *a, **k: None
sys.modules["moviepy"] = _mp
sys.modules["moviepy.editor"] = _mp_ed

import torch  # noqa: E402

from vsr_place.backbone.adapter import ChipDiffusionAdapter  # noqa: E402
from vsr_place.verifier.verifier import Verifier  # noqa: E402
from vsr_place.renoising.local_repair import local_repair_loop  # noqa: E402

NAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
         "bigblue1", "bigblue2", "bigblue3", "bigblue4"]


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def profile_circuit(checkpoint, cond, x_in, name, dtype="fp32",
                    grad_checkpoint=False):
    rec = {"circuit": name, "n_macros": int(cond.x.shape[0]),
           "n_edges": int(cond.edge_index.shape[1]), "dtype": dtype,
           "grad_checkpoint": grad_checkpoint}
    try:
        # (a) Load adapter
        reset_peak()
        ad = ChipDiffusionAdapter.from_checkpoint(
            checkpoint, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
        )
        rec["mem_load_mb"] = peak_mb()

        # (b) Single forward pass through model
        reset_peak()
        cond_cu = cond.to("cuda")
        x_t = torch.randn(1, cond.x.shape[0], 2, device="cuda")
        t = torch.tensor([0.5], device="cuda")
        if dtype == "bf16":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = ad.model(x_t, cond_cu, t)
        elif dtype == "fp16":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = ad.model(x_t, cond_cu, t)
        else:
            _ = ad.model(x_t, cond_cu, t)
        rec["mem_forward_mb"] = peak_mb()

        # (c) Full guided_sample (this is where bigblue2/4 OOM)
        reset_peak()
        try:
            t0 = time.time()
            pl = ad.guided_sample(cond_cu, num_samples=1).squeeze(0)
            rec["mem_full_sample_mb"] = peak_mb()
            rec["full_sample_time"] = time.time() - t0
            rec["sample_succeeded"] = True
        except torch.cuda.OutOfMemoryError:
            rec["mem_full_sample_mb"] = peak_mb()
            rec["sample_succeeded"] = False
            print(f"  full sample OOM at peak {rec['mem_full_sample_mb']:.0f}MB", flush=True)

        # (d) VSR-post on synthetic baseline (independent of backbone OOM)
        reset_peak()
        cw, ch = ad.get_canvas_size(cond)
        synthetic_centers = torch.rand(cond.x.shape[0], 2) * 0.8 + 0.1
        synthetic_centers[:, 0] *= cw; synthetic_centers[:, 1] *= ch
        sz_cpu = (cond.x.cpu() / 2.0)
        sz_cpu[:, 0] *= cw; sz_cpu[:, 1] *= ch
        ver = Verifier(canvas_width=cw, canvas_height=ch)
        fb = ver(synthetic_centers, sz_cpu)
        mask = fb.severity_vector > 0
        try:
            t0 = time.time()
            _ = local_repair_loop(synthetic_centers, sz_cpu, cw, ch, num_steps=100,
                                  step_size=0.3, only_mask=mask,
                                  edge_index=cond.edge_index.cpu(), hpwl_weight=2.0)
            rec["vsr_post_only_time"] = time.time() - t0
            rec["vsr_post_only_succeeded"] = True
        except Exception as e:
            rec["vsr_post_only_succeeded"] = False
            rec["vsr_post_only_error"] = str(e)[:200]
        del ad
    except torch.cuda.OutOfMemoryError as e:
        rec["error"] = "OOM during setup"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/large-v2/large-v2.ckpt")
    p.add_argument("--circuits", type=int, nargs="+", default=[1, 5, 7],  # adaptec2, bigblue2, bigblue4
                   help="Indices into NAMES.")
    p.add_argument("--out", default="results/ispd2005/mem_profile.json")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    if args.mock:
        print("=== mem_profile preflight ===", flush=True)
        print(f"  target circuits: {[NAMES[i] for i in args.circuits]}", flush=True)
        sample = {"circuit": "fakecircuit", "n_macros": 0, "n_edges": 0,
                  "dtype": "fp32", "grad_checkpoint": False,
                  "mem_load_mb": 0.0, "mem_forward_mb": 0.0,
                  "mem_full_sample_mb": 0.0, "sample_succeeded": False,
                  "vsr_post_only_succeeded": True}
        json.dumps(sample)
        print("  schema valid", flush=True)
        print("=== preflight OK ===", flush=True)
        return

    from run_vsr import load_benchmark_data, filter_macros_only
    val = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for ci in args.circuits:
        name = NAMES[ci]
        x_in, cond = val[ci]
        print(f"\n=== {name} (n_macros={cond.x.shape[0]}, n_edges={cond.edge_index.shape[1]}) ===", flush=True)

        for dtype in ["fp32", "bf16", "fp16"]:
            print(f" -- dtype={dtype} --", flush=True)
            rec = profile_circuit(args.checkpoint, cond, x_in, name, dtype=dtype)
            rows.append(rec)
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"   {rec}", flush=True)

    print(f"\nDone. Wrote {out_path}.")


if __name__ == "__main__":
    main()
