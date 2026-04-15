#!/usr/bin/env python3
"""Run VSR-Place experiments.

Usage:
    # Dry run (print config)
    python scripts/run_vsr.py --dry-run

    # Run VSR-Place on synthetic data
    python scripts/run_vsr.py --checkpoint checkpoints/large-v2/step_250000.ckpt --task v1.61

    # Run on IBM benchmarks
    python scripts/run_vsr.py --checkpoint checkpoints/large-v2/step_250000.ckpt --task ibm.cluster512.v1

    # Baseline (no VSR)
    python scripts/run_vsr.py --checkpoint checkpoints/large-v2/step_250000.ckpt --task v1.61 --no-vsr

    # Custom config
    python scripts/run_vsr.py --config configs/methods/vsr_mask_only.yaml --checkpoint ... --task ...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root and ChipDiffusion to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHIPDIFFUSION_ROOT = PROJECT_ROOT / "third_party" / "chipdiffusion"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(CHIPDIFFUSION_ROOT))

from vsr_place.verifier.verifier import Verifier
from vsr_place.renoising.selector import (
    GlobalSelector, ThresholdSelector, TopKSelector, AdaptiveThresholdSelector,
)
from vsr_place.renoising.strength import (
    FixedStrength, SeverityAdaptiveStrength, ScheduledStrength,
)
from vsr_place.loop.vsr_loop import VSRLoop
from vsr_place.loop.budget import RepairBudget
from vsr_place.metrics.legality import compute_legality_metrics, compute_pass_rate
from vsr_place.metrics.violations import violation_reduction_rate


def build_selector(cfg: dict):
    sel_type = cfg.get("type", "threshold")
    if sel_type == "global":
        return GlobalSelector()
    elif sel_type == "threshold":
        return ThresholdSelector(threshold=cfg.get("threshold", 0.0))
    elif sel_type == "top_k":
        return TopKSelector(k=cfg.get("k", 5))
    elif sel_type == "adaptive":
        return AdaptiveThresholdSelector(percentile=cfg.get("percentile", 50.0))
    else:
        raise ValueError(f"Unknown selector type: {sel_type}")


def build_strength(cfg: dict):
    str_type = cfg.get("type", "fixed")
    if str_type == "fixed":
        return FixedStrength(alpha=cfg.get("alpha", 0.3))
    elif str_type == "severity_adaptive":
        return SeverityAdaptiveStrength(
            alpha_min=cfg.get("alpha_min", 0.1),
            alpha_max=cfg.get("alpha_max", 0.5),
        )
    elif str_type == "scheduled":
        return ScheduledStrength(alpha_schedule=cfg.get("schedule", [0.5, 0.3, 0.2, 0.1]))
    else:
        raise ValueError(f"Unknown strength type: {str_type}")


def load_benchmark_data(task: str):
    """Load benchmark data using ChipDiffusion's data pipeline.

    Args:
        task: Dataset task name (e.g., 'v1.61', 'ibm.cluster512.v1', 'ispd2005').

    Returns:
        List of (x, cond) tuples from the validation set.
    """
    # Change CWD to ChipDiffusion root so relative paths work
    original_cwd = os.getcwd()
    os.chdir(str(CHIPDIFFUSION_ROOT))

    try:
        from diffusion import utils as cd_utils
        _, val_set = cd_utils.load_graph_data_with_config(task)
        print(f"Loaded {len(val_set)} validation samples from task '{task}'")
        return val_set
    finally:
        os.chdir(original_cwd)


def run_single_sample(adapter, verifier, loop, x_in, cond, sample_idx, device, legalize=False):
    """Run VSR-Place on a single benchmark sample.

    Args:
        adapter: ChipDiffusionAdapter.
        verifier: Verifier instance.
        loop: VSRLoop instance (or None for baseline).
        x_in: (V, 2) ground truth placement (for reference).
        cond: PyG Data object.
        sample_idx: Sample index for logging.
        device: Torch device.

    Returns:
        Dict of metrics for this sample.
    """
    cond = cond.to(device)

    if loop is not None:
        # VSR-Place mode
        result = loop.run(cond=cond)
        placement = result.placement
        metrics = result.metrics.copy()
        metrics["method"] = "vsr_place"
    else:
        # Baseline mode: just sample
        placement = adapter.sample(cond, num_samples=1)
        placement = placement.squeeze(0)  # (V, 2)
        # Verify the baseline result
        centers, sizes = adapter.decode_placement(placement, cond)
        feedback = verifier(centers, sizes)
        metrics = {
            "method": "baseline",
            **feedback.global_stats,
            "num_repair_loops": 0,
        }

    # Optional: apply ChipDiffusion's legalizer after VSR/baseline
    if legalize:
        placement = adapter.legalize(placement, cond, mode="opt",
                                     step_size=0.2, grad_descent_steps=5000)

    # Decode final placement and compute verification metrics
    centers, sizes = adapter.decode_placement(placement, cond)
    final_feedback = verifier(centers, sizes)
    legality_metrics = compute_legality_metrics(final_feedback)

    metrics.update({
        "sample_idx": sample_idx,
        "legalized": legalize,
        **legality_metrics,
    })

    return metrics, placement


def run_experiment(cfg: dict, seed: int = 42):
    """Run a full VSR-Place experiment across all benchmark samples."""
    torch.manual_seed(seed)

    device = cfg.get("backbone", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    checkpoint_path = cfg.get("backbone", {}).get("checkpoint")
    if checkpoint_path is None:
        print("ERROR: No checkpoint specified. Use --checkpoint <path>")
        return None

    task = cfg.get("benchmark", {}).get("task", "v1.61")

    # Load model
    from vsr_place.backbone.adapter import ChipDiffusionAdapter
    print(f"Loading checkpoint: {checkpoint_path}")
    adapter = ChipDiffusionAdapter.from_checkpoint(checkpoint_path, device=device)

    # Load benchmark data
    print(f"Loading benchmark: {task}")
    val_set = load_benchmark_data(task)

    num_samples = min(
        cfg.get("eval", {}).get("num_samples", len(val_set)),
        len(val_set),
    )

    # Build verifier — canvas size from first sample's cond
    # ChipDiffusion decodes to absolute coords, so verifier needs actual canvas dims
    first_x, first_cond = val_set[0]
    canvas_w, canvas_h = adapter.get_canvas_size(first_cond)
    print(f"Canvas size: {canvas_w} x {canvas_h}")

    verifier_cfg = cfg.get("verifier", {})
    verifier = Verifier(
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        min_spacing=verifier_cfg.get("min_spacing", 0.0),
        check_spacing=verifier_cfg.get("check_spacing", False),
    )

    # Build VSR loop (or None for baseline)
    vsr_cfg = cfg.get("vsr", {})
    loop = None

    if vsr_cfg.get("enabled", True):
        selector = build_selector(vsr_cfg.get("selector", {}))
        strength = build_strength(vsr_cfg.get("strength", {}))
        budget = RepairBudget(
            max_loops=vsr_cfg.get("budget", {}).get("max_loops", 4),
            early_stop_on_legal=vsr_cfg.get("budget", {}).get("early_stop_on_legal", True),
            min_improvement=vsr_cfg.get("budget", {}).get("min_improvement", 0.0),
        )
        loop = VSRLoop(
            backend=adapter,
            verifier=verifier,
            selector=selector,
            strength=strength,
            budget=budget,
            denoise_steps=vsr_cfg.get("denoise_steps", 50),
            save_intermediates=False,
        )
        method_name = "vsr_place"
    else:
        method_name = "baseline"

    legalize = cfg.get("legalize", False)

    # Run on each sample
    print(f"\nRunning {method_name} on {num_samples} samples (seed={seed})...")
    all_metrics = []
    t_start = time.time()

    for i in range(num_samples):
        x_in, cond = val_set[i]
        print(f"  Sample {i+1}/{num_samples}...", end=" ", flush=True)

        t_sample = time.time()
        metrics, placement = run_single_sample(
            adapter, verifier, loop, x_in, cond, i, device,
            legalize=legalize,
        )
        elapsed = time.time() - t_sample

        metrics["seed"] = seed
        metrics["task"] = task
        metrics["sample_time"] = elapsed
        all_metrics.append(metrics)

        legal_str = "LEGAL" if metrics["is_legal"] else f"violations={metrics['total_violations']}"
        print(f"{legal_str} ({elapsed:.1f}s)")

    total_time = time.time() - t_start

    # Summary
    num_legal = sum(1 for m in all_metrics if m["is_legal"])
    pass_rate = num_legal / len(all_metrics)
    avg_violations = sum(m["total_violations"] for m in all_metrics) / len(all_metrics)

    print(f"\n{'='*50}")
    print(f"Method: {method_name} | Task: {task} | Seed: {seed}")
    print(f"Pass rate: {num_legal}/{len(all_metrics)} ({pass_rate:.1%})")
    print(f"Avg violations: {avg_violations:.2f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*50}\n")

    return {
        "method": method_name,
        "task": task,
        "seed": seed,
        "pass_rate": pass_rate,
        "avg_violations": avg_violations,
        "total_time": total_time,
        "per_sample": all_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Run VSR-Place experiments")
    parser.add_argument("--config", type=str, default="configs/defaults.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to ChipDiffusion checkpoint")
    parser.add_argument("--task", type=str, default=None, help="Benchmark task (e.g. v1.61, ibm.cluster512.v1)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-vsr", action="store_true", help="Run baseline without VSR")
    parser.add_argument("--legalize", action="store_true", help="Apply legalizer after placement")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.checkpoint:
        cfg.setdefault("backbone", {})["checkpoint"] = args.checkpoint
    if args.task:
        cfg.setdefault("benchmark", {})["task"] = args.task
    if args.no_vsr:
        cfg.setdefault("vsr", {})["enabled"] = False
    if args.legalize:
        cfg["legalize"] = True

    if args.dry_run:
        print(yaml.dump(cfg, default_flow_style=False))
        return

    seeds = [args.seed] if args.seed else cfg.get("eval", {}).get("seeds", [42])

    for seed in seeds:
        result = run_experiment(cfg, seed=seed)
        if result is None:
            continue

        # Save results
        output_dir = Path(cfg.get("logging", {}).get("output_dir", "results/runs"))
        run_dir = output_dir / f"{result['method']}_{result['task']}_{seed}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
