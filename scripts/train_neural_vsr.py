#!/usr/bin/env python3
"""Train NeuralVSR on synthetic data.

Usage:
    # Quick sanity check (small dataset, few epochs)
    python scripts/train_neural_vsr.py --smoke

    # Full training
    python scripts/train_neural_vsr.py \
        --output checkpoints/neural_vsr/best.pt \
        --num-train 100000 --num-val 2000 \
        --batch-size 32 --epochs 20

    # Custom architecture
    python scripts/train_neural_vsr.py \
        --hidden-dim 96 --num-layers 4 --heads 4
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="checkpoints/neural_vsr/best.pt")
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-val", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-min", type=int, default=30)
    parser.add_argument("--n-max", type=int, default=300)
    parser.add_argument("--canvas-side", type=float, default=10.0)
    parser.add_argument("--perturb-scale", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test with tiny dataset")
    args = parser.parse_args()

    if args.smoke:
        # Override with tiny settings
        args.num_train = 50
        args.num_val = 10
        args.batch_size = 4
        args.epochs = 2
        args.device = "cpu"
        print("=== SMOKE TEST MODE ===")

    from vsr_place.neural.train import train
    train(
        output_path=args.output,
        num_train=args.num_train,
        num_val=args.num_val,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        device=args.device,
        n_range=(args.n_min, args.n_max),
        canvas_side=args.canvas_side,
        perturb_scale=args.perturb_scale,
    )


if __name__ == "__main__":
    main()
