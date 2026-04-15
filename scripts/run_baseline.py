#!/usr/bin/env python3
"""Run ChipDiffusion baselines directly using its eval pipeline.

This calls ChipDiffusion's own eval.py for ground-truth baseline numbers,
ensuring our comparisons are fair (same code, same eval protocol).

Usage:
    # Unguided baseline
    python scripts/run_baseline.py --checkpoint checkpoints/large-v2/step_250000.ckpt \
        --task v1.61 --mode unguided

    # Guided baseline
    python scripts/run_baseline.py --checkpoint checkpoints/large-v2/step_250000.ckpt \
        --task ibm.cluster512.v1 --mode guided

    # Guided + legalizer
    python scripts/run_baseline.py --checkpoint checkpoints/large-v2/step_250000.ckpt \
        --task ibm.cluster512.v1 --mode guided_legalized
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHIPDIFFUSION_ROOT = PROJECT_ROOT / "third_party" / "chipdiffusion"


def build_command(args):
    """Build the ChipDiffusion eval command."""
    base_cmd = [
        sys.executable, "diffusion/eval.py",
        f"task={args.task}",
        f"from_checkpoint={args.checkpoint}",
        f"num_output_samples={args.num_samples}",
        f"seed={args.seed}",
        "logger.wandb=False",
    ]

    if args.mode == "unguided":
        base_cmd.extend([
            "method=eval",
            "legalizer=none",
            "guidance=none",
        ])
    elif args.mode == "guided":
        base_cmd.extend([
            "method=eval_guided",
        ])
    elif args.mode == "guided_legalized":
        base_cmd.extend([
            "method=eval_guided",
            "legalizer=opt-adam",
        ])
    elif args.mode == "macro_only":
        base_cmd.extend([
            "method=eval_macro_only",
            "legalizer=opt-adam",
            "guidance=opt",
            "model.grad_descent_steps=20",
            "model.hpwl_guidance_weight=16e-4",
            "legalization.alpha_lr=8e-3",
            "legalization.hpwl_weight=12e-5",
            "legalization.legality_potential_target=0",
            "legalization.grad_descent_steps=20000",
            "macros_only=True",
        ])
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return base_cmd


def main():
    parser = argparse.ArgumentParser(description="Run ChipDiffusion baselines")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="v1.61",
                        help="Dataset task (v1.61, ibm.cluster512.v1, ispd2005)")
    parser.add_argument("--mode", type=str, default="unguided",
                        choices=["unguided", "guided", "guided_legalized", "macro_only"])
    parser.add_argument("--num-samples", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = build_command(args)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CHIPDIFFUSION_ROOT)

    full_cmd = " ".join(cmd)
    print(f"Working directory: {CHIPDIFFUSION_ROOT}")
    print(f"Command: {full_cmd}")

    if args.dry_run:
        return

    print(f"\n{'='*50}")
    print(f"Running {args.mode} baseline on {args.task} (seed={args.seed})")
    print(f"{'='*50}\n")

    result = subprocess.run(
        cmd,
        cwd=str(CHIPDIFFUSION_ROOT),
        env=env,
    )

    if result.returncode != 0:
        print(f"\nERROR: Baseline exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nBaseline complete. Check logs/ directory in {CHIPDIFFUSION_ROOT}")


if __name__ == "__main__":
    main()
