#!/usr/bin/env python3
"""Run full ablation suite for VSR-Place.

Generates configs for each ablation variant and runs them sequentially.

Usage:
    python scripts/run_ablations.py --checkpoint <path> --task v1.61 --seed 42
    python scripts/run_ablations.py --checkpoint <path> --task v1.61 --dry-run
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ablation dimensions from proposal Section 8.4
ABLATIONS = {
    # === Re-noising scope ===
    "scope_global": {"vsr": {"selector": {"type": "global"}}},
    "scope_threshold_0.0": {"vsr": {"selector": {"type": "threshold", "threshold": 0.0}}},
    "scope_threshold_0.01": {"vsr": {"selector": {"type": "threshold", "threshold": 0.01}}},
    "scope_threshold_0.05": {"vsr": {"selector": {"type": "threshold", "threshold": 0.05}}},
    "scope_threshold_0.1": {"vsr": {"selector": {"type": "threshold", "threshold": 0.1}}},
    "scope_topk_3": {"vsr": {"selector": {"type": "top_k", "k": 3}}},
    "scope_topk_5": {"vsr": {"selector": {"type": "top_k", "k": 5}}},
    "scope_topk_10": {"vsr": {"selector": {"type": "top_k", "k": 10}}},

    # === Re-noising strength ===
    "strength_fixed_0.1": {"vsr": {"strength": {"type": "fixed", "alpha": 0.1}}},
    "strength_fixed_0.3": {"vsr": {"strength": {"type": "fixed", "alpha": 0.3}}},
    "strength_fixed_0.5": {"vsr": {"strength": {"type": "fixed", "alpha": 0.5}}},
    "strength_fixed_0.7": {"vsr": {"strength": {"type": "fixed", "alpha": 0.7}}},
    "strength_adaptive": {"vsr": {"strength": {"type": "severity_adaptive", "alpha_min": 0.1, "alpha_max": 0.5}}},

    # === Repair budget ===
    "budget_1": {"vsr": {"budget": {"max_loops": 1}}},
    "budget_2": {"vsr": {"budget": {"max_loops": 2}}},
    "budget_4": {"vsr": {"budget": {"max_loops": 4}}},
    "budget_8": {"vsr": {"budget": {"max_loops": 8}}},

    # === Constraint set ===
    "constraints_boundary_only": {"verifier": {"overlap_weight": 0.0, "boundary_weight": 1.0}},
    "constraints_overlap_only": {"verifier": {"overlap_weight": 1.0, "boundary_weight": 0.0}},
    "constraints_both": {"verifier": {"overlap_weight": 1.0, "boundary_weight": 1.0}},
    "constraints_both_spacing": {"verifier": {"overlap_weight": 1.0, "boundary_weight": 1.0, "check_spacing": True, "min_spacing": 0.01}},
}


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning new dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run VSR-Place ablation suite")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="v1.61")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablations", type=str, nargs="*", default=None,
                        help="Specific ablation names to run (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="List available ablations")
    args = parser.parse_args()

    if args.list:
        print("Available ablations:")
        for name in sorted(ABLATIONS.keys()):
            print(f"  {name}")
        return

    # Load base config
    base_config_path = PROJECT_ROOT / "configs" / "methods" / "vsr_mask_only.yaml"
    defaults_path = PROJECT_ROOT / "configs" / "defaults.yaml"

    with open(defaults_path) as f:
        base_cfg = yaml.safe_load(f)
    with open(base_config_path) as f:
        method_cfg = yaml.safe_load(f)
    base_cfg = deep_merge(base_cfg, method_cfg)
    base_cfg["backbone"]["checkpoint"] = args.checkpoint
    base_cfg.setdefault("benchmark", {})["task"] = args.task

    ablation_names = args.ablations or sorted(ABLATIONS.keys())
    total = len(ablation_names)

    print(f"=== VSR-Place Ablation Suite ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Seed: {args.seed}")
    print(f"Ablations: {total}")
    print()

    if args.dry_run:
        for name in ablation_names:
            override = ABLATIONS.get(name)
            if override is None:
                print(f"  SKIP {name} (unknown)")
                continue
            cfg = deep_merge(base_cfg, override)
            print(f"  {name}:")
            # Show only the changed parts
            for k, v in override.items():
                print(f"    {k}: {v}")
        return

    # Import here (needs torch)
    from scripts.run_vsr import run_experiment

    results_dir = PROJECT_ROOT / "results" / "ablations" / f"{args.task}_{args.seed}_{int(time.time())}"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for idx, name in enumerate(ablation_names):
        override = ABLATIONS.get(name)
        if override is None:
            print(f"[{idx+1}/{total}] SKIP {name} (unknown)")
            continue

        cfg = deep_merge(base_cfg, override)
        print(f"\n[{idx+1}/{total}] Running ablation: {name}")

        result = run_experiment(cfg, seed=args.seed)

        if result is not None:
            ablation_dir = results_dir / name
            ablation_dir.mkdir(parents=True, exist_ok=True)

            with open(ablation_dir / "config.yaml", "w") as f:
                yaml.dump(cfg, f)
            with open(ablation_dir / "metrics.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

            summary[name] = {
                "pass_rate": result["pass_rate"],
                "avg_violations": result["avg_violations"],
                "total_time": result["total_time"],
            }

    # Save summary
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Ablation suite complete. Results: {results_dir}")
    print(f"\nSummary:")
    print(f"{'Ablation':<30} {'Pass Rate':>10} {'Avg Violations':>15} {'Time':>8}")
    print("-" * 65)
    for name, m in sorted(summary.items()):
        print(f"{name:<30} {m['pass_rate']:>10.1%} {m['avg_violations']:>15.2f} {m['total_time']:>7.1f}s")


if __name__ == "__main__":
    main()
