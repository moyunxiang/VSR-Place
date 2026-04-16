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
sys.path.insert(0, str(CHIPDIFFUSION_ROOT / "diffusion"))

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
    """Load benchmark data from ChipDiffusion's dataset directory.

    Supports both old format (separate graph*.pickle + output*.pickle) and
    new format (single numbered pickle files from data-gen).

    Args:
        task: Dataset task name (e.g., 'v1.61', 'ibm.cluster512.v1', 'ispd2005').

    Returns:
        List of (x, cond) tuples from the validation set.
    """
    import pickle
    import re
    from pathlib import Path

    import torch
    from torch_geometric.data import Data

    # Find dataset directory
    dataset_dir = CHIPDIFFUSION_ROOT / "datasets" / "graph" / task
    if not dataset_dir.exists():
        # Try data-gen outputs
        dataset_dir = CHIPDIFFUSION_ROOT / "data-gen" / "outputs" / task
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset '{task}' not found in datasets/graph/ or data-gen/outputs/")

    # Load config
    config_path = dataset_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        n_train = cfg.get("train_samples", cfg.get("num_train_samples", 0))
        n_val = cfg.get("val_samples", cfg.get("num_val_samples", 0))
    else:
        n_train = 0
        n_val = None  # Use all available

    # Find pickle files (excluding config/checkpoint files)
    pickle_files = sorted(
        [p for p in dataset_dir.glob("*.pickle") if p.name != "config.yaml"],
        key=lambda p: int(re.search(r'\d+', p.stem).group()) if re.search(r'\d+', p.stem) else 0
    )

    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in {dataset_dir}")

    # Check if it's old format (graph*.pickle) or new format (numbered)
    has_graph_files = any(p.name.startswith("graph") for p in pickle_files)

    val_set = []
    if has_graph_files:
        # Old format: separate graph + output files
        graph_files = sorted([p for p in pickle_files if p.name.startswith("graph")])
        output_files = sorted([p for p in pickle_files if p.name.startswith("output")])
        val_graphs = graph_files[n_train:n_train + (n_val or len(graph_files))]

        for gf in val_graphs:
            with open(gf, "rb") as f:
                cond = pickle.load(f)
            # Find matching output
            idx = re.search(r'\d+', gf.stem).group()
            of = dataset_dir / f"output{idx}.pickle"
            if of.exists():
                with open(of, "rb") as f:
                    x = torch.tensor(pickle.load(f), dtype=torch.float32)
            else:
                x = torch.zeros(cond.x.shape[0], 2)
            val_set.append((x, cond))
    else:
        # New format: numbered pickle files from data-gen (contain both graph and placement)
        val_files = pickle_files[n_train:n_train + (n_val or len(pickle_files))]

        for pf in val_files:
            with open(pf, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                cond = data.get("graph", data.get("cond"))
                x = data.get("placement", data.get("x"))
                if isinstance(x, torch.Tensor):
                    pass
                else:
                    x = torch.tensor(x, dtype=torch.float32)
            elif isinstance(data, (list, tuple)) and len(data) == 2:
                x, cond = data[0], data[1]
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple):
                # List of (x, cond) tuples — batch of samples in one file
                for x_item, cond_item in data:
                    if not isinstance(x_item, torch.Tensor):
                        x_item = torch.tensor(x_item, dtype=torch.float32)
                    val_set.append((x_item, cond_item))
                continue  # Already appended all samples from this file
            elif hasattr(data, "x"):
                cond = data
                x = torch.zeros(cond.x.shape[0], 2)
            else:
                raise ValueError(f"Unknown pickle format in {pf}: {type(data)}")
            val_set.append((x, cond))

    # Trim to requested val count
    if n_val is not None and len(val_set) > n_val:
        val_set = val_set[:n_val]

    print(f"Loaded {len(val_set)} validation samples from task '{task}' ({dataset_dir})")
    return val_set


def filter_macros_only(x: torch.Tensor, cond) -> tuple:
    """Filter to macro-only subgraph (drop standard cells).

    Required for ISPD2005 where circuits have 200K+ nodes but only
    a few hundred to thousands of macros. Running pairwise overlap
    on all nodes would be infeasible (N^2 memory).

    Args:
        x: (V, 2) placement coordinates.
        cond: PyG Data object with is_macros attribute.

    Returns:
        Tuple of (x_macro, cond_macro) with only macro nodes.
    """
    if not hasattr(cond, "is_macros") or cond.is_macros is None:
        return x, cond  # No macro info, return as-is

    macro_mask = cond.is_macros.bool()
    n_macros = macro_mask.sum().item()

    if n_macros == 0 or n_macros == len(macro_mask):
        return x, cond  # All or none are macros

    # Map old indices to new indices
    old_to_new = torch.full((len(macro_mask),), -1, dtype=torch.long)
    old_to_new[macro_mask] = torch.arange(n_macros)

    # Filter node features
    x_macro = x[macro_mask]
    cond_x_macro = cond.x[macro_mask]

    # Filter edges (keep only edges between macros)
    edge_index = cond.edge_index
    src_is_macro = macro_mask[edge_index[0]]
    dst_is_macro = macro_mask[edge_index[1]]
    edge_mask = src_is_macro & dst_is_macro

    edge_index_macro = old_to_new[edge_index[:, edge_mask]]
    edge_attr_macro = cond.edge_attr[edge_mask] if cond.edge_attr is not None else None

    from torch_geometric.data import Data
    cond_macro = Data(
        x=cond_x_macro,
        edge_index=edge_index_macro,
        edge_attr=edge_attr_macro,
        is_ports=torch.zeros(n_macros, dtype=torch.bool),
        is_macros=torch.ones(n_macros, dtype=torch.bool),
    )

    # Preserve chip_size
    if hasattr(cond, "chip_size"):
        cond_macro.chip_size = cond.chip_size

    return x_macro, cond_macro


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

    # Filter to macro-only if requested (required for large circuits like ISPD2005)
    macro_only = cfg.get("macro_only", False)
    if macro_only:
        print("Filtering to macro-only subgraph...")
        val_set = [filter_macros_only(x, cond) for x, cond in val_set]
        first_x, first_cond = val_set[0]
        print(f"  Macros per circuit: {[cond.x.shape[0] for _, cond in val_set]}")

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
    parser.add_argument("--macro-only", action="store_true", help="Filter to macro-only subgraph (for ISPD2005)")
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
    if args.macro_only:
        cfg["macro_only"] = True

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
