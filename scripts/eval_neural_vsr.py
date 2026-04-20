#!/usr/bin/env python3
"""Evaluate NeuralVSR on ISPD2005 circuits.

Compares:
- Guided baseline (no repair)
- Hand-crafted VSR (repulsive force, 100 iters)
- NeuralVSR (trained GNN, K iters)

Usage:
    python scripts/eval_neural_vsr.py \
        --checkpoint checkpoints/large-v2/large-v2.ckpt \
        --neural checkpoints/neural_vsr/best.pt \
        --circuits 0 2 4 \
        --seeds 42 123 300
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "chipdiffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "chipdiffusion" / "diffusion"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from scripts.run_vsr import load_benchmark_data, filter_macros_only  # noqa: E402
from vsr_place.backbone.adapter import ChipDiffusionAdapter  # noqa: E402
from vsr_place.verifier.verifier import Verifier  # noqa: E402
from vsr_place.renoising.local_repair import local_repair_loop  # noqa: E402
from vsr_place.neural.train import load_model  # noqa: E402
from vsr_place.neural.infer import neural_repair_loop  # noqa: E402


CIRCUIT_NAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
                 "bigblue1", "bigblue2", "bigblue3", "bigblue4"]


def eval_one(ckpt, neural_ckpt, circuit_idx, seed, num_steps_neural=10, device="cuda"):
    """Evaluate all 3 methods on one circuit + seed."""
    val_set = load_benchmark_data("ispd2005")
    val_set = [filter_macros_only(x, cond) for x, cond in val_set]

    name = CIRCUIT_NAMES[circuit_idx]
    x_in, cond = val_set[circuit_idx]

    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    adapter = ChipDiffusionAdapter.from_checkpoint(
        ckpt, device=device,
        input_shape=tuple(x_in.shape), guidance="opt",
    )
    canvas_w, canvas_h = adapter.get_canvas_size(cond)
    verifier = Verifier(canvas_width=canvas_w, canvas_height=canvas_h)

    # Guided sampling
    t0 = time.time()
    placement = adapter.guided_sample(cond.to(device), num_samples=1).squeeze(0)
    t_sample = time.time() - t0

    centers, sizes = adapter.decode_placement(placement, cond)
    centers_cpu = centers.cpu()
    sizes_cpu = sizes.cpu()

    # Extract edges (for NeuralVSR)
    edge_index = cond.edge_index.cpu()
    edge_attr = cond.edge_attr.cpu() if cond.edge_attr is not None else None

    # Baseline
    fb_base = verifier(centers_cpu, sizes_cpu)
    mask = fb_base.severity_vector > 0

    row = {
        "circuit": name,
        "n_macros": cond.x.shape[0],
        "seed": seed,
        "t_sample": t_sample,
        "baseline_violations": fb_base.global_stats["total_violations"],
        "baseline_overlap": fb_base.global_stats["num_overlap_violations"],
        "baseline_boundary": fb_base.global_stats["num_boundary_violations"],
        "n_offending": int(mask.sum().item()),
    }

    # Hand-crafted
    if mask.any():
        t0 = time.time()
        centers_hand = local_repair_loop(
            centers_cpu, sizes_cpu, canvas_w, canvas_h,
            num_steps=100, step_size=0.3, only_mask=mask,
        )
        t_hand = time.time() - t0
        fb_hand = verifier(centers_hand, sizes_cpu)
        row["hand_violations"] = fb_hand.global_stats["total_violations"]
        row["t_hand"] = t_hand
    else:
        row["hand_violations"] = row["baseline_violations"]
        row["t_hand"] = 0.0

    # NeuralVSR
    if neural_ckpt and Path(neural_ckpt).exists():
        neural_model = load_model(neural_ckpt, device=device)
        t0 = time.time()
        if mask.any():
            centers_neural = neural_repair_loop(
                centers_cpu, sizes_cpu, canvas_w, canvas_h,
                model=neural_model,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_steps=num_steps_neural,
                step_size=1.0,
                only_mask=mask,
                device=device,
            )
        else:
            centers_neural = centers_cpu
        t_neural = time.time() - t0
        fb_neural = verifier(centers_neural, sizes_cpu)
        row["neural_violations"] = fb_neural.global_stats["total_violations"]
        row["t_neural"] = t_neural
    else:
        row["neural_violations"] = None
        row["t_neural"] = None

    # Improvements
    b = row["baseline_violations"]
    if b > 0:
        row["hand_improvement"] = (b - row["hand_violations"]) / b * 100
        if row["neural_violations"] is not None:
            row["neural_improvement"] = (b - row["neural_violations"]) / b * 100

    del adapter
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="ChipDiffusion checkpoint")
    parser.add_argument("--neural", type=str, default=None, help="NeuralVSR checkpoint (optional)")
    parser.add_argument("--circuits", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-steps-neural", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/neural_vsr/eval.json")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i in args.circuits:
        name = CIRCUIT_NAMES[i]
        print(f"\n{'='*60}")
        print(f"  {name} (idx={i})")
        print(f"{'='*60}", flush=True)

        for seed in args.seeds:
            try:
                row = eval_one(args.checkpoint, args.neural, i, seed,
                               num_steps_neural=args.num_steps_neural)
                all_results.append(row)
                b = row["baseline_violations"]
                h = row["hand_violations"]
                n = row.get("neural_violations")
                n_str = f"{n:6d}" if n is not None else "  N/A "
                print(f"  seed={seed}: baseline={b:6d} hand={h:6d} neural={n_str}  "
                      f"(hand {row['hand_improvement']:+.1f}%, "
                      f"neural {row.get('neural_improvement', float('nan')):+.1f}%)",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                all_results.append({"circuit": name, "seed": seed, "error": "OOM"})
                print(f"  seed={seed}: OOM", flush=True)
            except Exception as e:
                all_results.append({"circuit": name, "seed": seed,
                                    "error": f"{type(e).__name__}: {e}"})
                print(f"  seed={seed}: {type(e).__name__}: {e}", flush=True)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Circuit':<10} {'Baseline':<10} {'Hand':<10} {'Neural':<10} "
          f"{'hand%':<10} {'neural%':<10}")
    print("-" * 70)

    by_circuit = {}
    for r in all_results:
        if "error" in r:
            continue
        c = r["circuit"]
        by_circuit.setdefault(c, []).append(r)

    for c in CIRCUIT_NAMES:
        rows = by_circuit.get(c, [])
        if not rows:
            continue
        b = sum(r["baseline_violations"] for r in rows) / len(rows)
        h = sum(r["hand_violations"] for r in rows) / len(rows)
        neural_rows = [r for r in rows if r.get("neural_violations") is not None]
        if neural_rows:
            n = sum(r["neural_violations"] for r in neural_rows) / len(neural_rows)
            n_str = f"{n:<10.0f}"
            n_imp = (b - n) / b * 100
            n_imp_str = f"{n_imp:+.1f}%"
        else:
            n_str = "N/A"
            n_imp_str = "N/A"
        h_imp = (b - h) / b * 100
        print(f"{c:<10} {b:<10.0f} {h:<10.0f} {n_str:<10} "
              f"{h_imp:+.1f}%{'':<5} {n_imp_str}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
