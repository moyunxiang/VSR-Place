#!/usr/bin/env python3
"""Run VSR-Place experiments.

Usage:
    python scripts/run_vsr.py --config configs/defaults.yaml
    python scripts/run_vsr.py --checkpoint path/to/model.pt --benchmark iccad04
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vsr_place.verifier.verifier import Verifier
from vsr_place.renoising.selector import (
    GlobalSelector,
    ThresholdSelector,
    TopKSelector,
    AdaptiveThresholdSelector,
)
from vsr_place.renoising.strength import (
    FixedStrength,
    SeverityAdaptiveStrength,
    ScheduledStrength,
)
from vsr_place.loop.vsr_loop import VSRLoop
from vsr_place.loop.budget import RepairBudget
from vsr_place.conditioning.mask_only import MaskOnlyRepair
from vsr_place.metrics.legality import compute_legality_metrics


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


def run_experiment(cfg: dict, seed: int = 42):
    """Run a single VSR-Place experiment with given config and seed."""
    torch.manual_seed(seed)

    device = cfg.get("backbone", {}).get("device", "cpu")

    # Check GPU availability
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    checkpoint_path = cfg.get("backbone", {}).get("checkpoint")
    if checkpoint_path is None:
        print("ERROR: No checkpoint specified. Set backbone.checkpoint in config.")
        print("This script requires a GPU and pretrained ChipDiffusion checkpoint.")
        print("Exiting.")
        return None

    # Import adapter (requires ChipDiffusion dependencies)
    from vsr_place.backbone.adapter import ChipDiffusionAdapter

    # Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    adapter = ChipDiffusionAdapter.from_checkpoint(checkpoint_path, device=device)

    # Build verifier
    verifier_cfg = cfg.get("verifier", {})
    verifier = Verifier(
        canvas_width=verifier_cfg.get("canvas_width", 2.0),
        canvas_height=verifier_cfg.get("canvas_height", 2.0),
        min_spacing=verifier_cfg.get("min_spacing", 0.0),
        check_spacing=verifier_cfg.get("check_spacing", False),
    )

    vsr_cfg = cfg.get("vsr", {})

    if vsr_cfg.get("enabled", True):
        # Build VSR components
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
            save_intermediates=cfg.get("eval", {}).get("save_placements", True),
        )

        print(f"Running VSR-Place (seed={seed})...")
        # TODO: Load actual benchmark data via adapter.load_benchmark()
        # result = loop.run(cond=benchmark_data)
        print("Benchmark data loading requires GPU environment. Skipping.")
        return None
    else:
        # Baseline: just run ChipDiffusion sampling
        print(f"Running baseline (seed={seed})...")
        print("Benchmark data loading requires GPU environment. Skipping.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run VSR-Place experiments")
    parser.add_argument("--config", type=str, default="configs/defaults.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.dry_run:
        print(yaml.dump(cfg, default_flow_style=False))
        return

    seeds = [args.seed] if args.seed else cfg.get("eval", {}).get("seeds", [42])

    for seed in seeds:
        result = run_experiment(cfg, seed=seed)
        if result is not None:
            # Save results
            output_dir = Path(cfg.get("logging", {}).get("output_dir", "results/runs"))
            run_dir = output_dir / f"seed_{seed}_{int(time.time())}"
            run_dir.mkdir(parents=True, exist_ok=True)

            with open(run_dir / "config.yaml", "w") as f:
                yaml.dump(cfg, f)
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(result.metrics, f, indent=2)

            print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
