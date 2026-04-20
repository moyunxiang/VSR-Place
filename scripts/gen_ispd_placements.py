#!/usr/bin/env python3
"""Pre-generate guided placements for ISPD2005 circuits.

Creates many (circuit, seed) placement pairs, saves as pickle for training.

Usage:
    python scripts/gen_ispd_placements.py \
        --checkpoint checkpoints/large-v2/large-v2.ckpt \
        --circuits 0 2 4 \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --output data/ispd_placements.pkl
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "chipdiffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "chipdiffusion" / "diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from run_vsr import load_benchmark_data, filter_macros_only  # noqa: E402
from vsr_place.backbone.adapter import ChipDiffusionAdapter  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--circuits", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output", type=str, default="data/ispd_placements.pkl")
    args = parser.parse_args()

    val_set = load_benchmark_data("ispd2005")
    val_set = [filter_macros_only(x, cond) for x, cond in val_set]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing if any (to resume)
    placements = []
    if out_path.exists():
        with open(out_path, "rb") as f:
            placements = pickle.load(f)
        done = {(p["circuit_idx"], p["seed"]) for p in placements}
        print(f"Resuming: {len(placements)} existing placements.", flush=True)
    else:
        done = set()

    # Reuse adapter across seeds for same circuit (avoids reloading 73MB checkpoint)
    for i in args.circuits:
        x_in, cond = val_set[i]
        n_macros = cond.x.shape[0]
        print(f"\n=== circuit {i} ({n_macros} macros) ===", flush=True)

        torch.cuda.empty_cache()
        try:
            adapter = ChipDiffusionAdapter.from_checkpoint(
                args.checkpoint, device="cuda",
                input_shape=tuple(x_in.shape), guidance="opt",
            )
            cw, ch = adapter.get_canvas_size(cond)
            cond_gpu = cond.to("cuda")
        except torch.cuda.OutOfMemoryError:
            print(f"  Loading adapter OOM, skip circuit {i}", flush=True)
            continue
        except Exception as e:
            print(f"  Adapter load ERROR {type(e).__name__}: {e}", flush=True)
            continue

        for seed in args.seeds:
            if (i, seed) in done:
                continue
            torch.cuda.empty_cache()
            torch.manual_seed(seed)
            try:
                t0 = time.time()
                placement = adapter.guided_sample(cond_gpu, num_samples=1).squeeze(0)
                centers, sizes = adapter.decode_placement(placement, cond)
                elapsed = time.time() - t0

                placements.append({
                    "circuit_idx": i,
                    "seed": seed,
                    "centers_bad": centers.cpu(),
                    "sizes": sizes.cpu(),
                    "edge_index": cond.edge_index.cpu(),
                    "edge_attr": cond.edge_attr.cpu() if cond.edge_attr is not None else None,
                    "canvas_w": cw,
                    "canvas_h": ch,
                })
                print(f"  seed={seed}: done ({elapsed:.1f}s, {len(placements)} total)", flush=True)
                with open(out_path, "wb") as f:
                    pickle.dump(placements, f)
            except torch.cuda.OutOfMemoryError:
                print(f"  seed={seed}: OOM, skip", flush=True)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  seed={seed}: ERROR {type(e).__name__}: {e}", flush=True)

        # Free adapter before moving to next circuit
        del adapter
        torch.cuda.empty_cache()

    print(f"\nDone. {len(placements)} placements saved to {out_path}")


if __name__ == "__main__":
    main()
