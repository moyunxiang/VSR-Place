#!/usr/bin/env python3
"""Train NeuralVSR on REAL ISPD2005 placements (not synthetic).

Closes the sim-to-real gap: model trained directly on the distribution
of placements it will see at inference.

Usage:
    python scripts/train_neural_vsr_real.py \
        --data data/ispd_placements.pkl \
        --output checkpoints/neural_vsr/real_v1.pt \
        --epochs 50 --batch-size 4 --lr 1e-3

Augmentation: each training epoch adds random perturbation to centers,
so we get many effective samples from a small set of real placements.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def collate(batch):
    """PyG-style concatenation of variable-size graphs."""
    centers = []
    node_features = []
    edge_indices = []
    edge_attrs = []
    target_deltas = []
    canvas_info = []

    offset = 0
    for item in batch:
        n = item["centers"].shape[0]
        centers.append(item["centers"])
        node_features.append(item["node_features"])
        target_deltas.append(item["target_delta"])
        edge_indices.append(item["edge_index"] + offset)
        edge_attrs.append(item["edge_attr"])
        canvas_info.append((item["canvas_w"], item["canvas_h"], n))
        offset += n

    return {
        "centers": torch.cat(centers, dim=0),
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "target_delta": torch.cat(target_deltas, dim=0),
        "canvas_info": canvas_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="pickle of placements")
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repair-iters", type=int, default=30,
                        help="Hand-crafted iterations for target")
    args = parser.parse_args()

    from vsr_place.neural.model import NeuralVSR
    from vsr_place.neural.real_dataset import RealISPDDataset

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    # Dataset
    print(f"Loading placements from {args.data}...", flush=True)
    full_ds = RealISPDDataset(args.data, repair_iters=args.repair_iters)
    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    # Deterministic split
    torch.manual_seed(0)
    perm = torch.randperm(n_total)
    train_indices = perm[:n_train].tolist()
    val_indices = perm[n_train:].tolist()

    train_ds = [full_ds[i] for i in train_indices]
    val_ds = [full_ds[i] for i in val_indices]

    print(f"  total={n_total}, train={len(train_ds)}, val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Model
    model = NeuralVSR(
        node_feat_dim=5, edge_feat_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, heads=args.heads,
    ).to(device)
    print(f"  params: {model.num_parameters():,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val = float("inf")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t_loss = 0.0
        nb = 0
        t0 = time.time()
        for batch in train_loader:
            centers = batch["centers"].to(device)
            feats = batch["node_features"].to(device)
            eidx = batch["edge_index"].to(device)
            eattr = batch["edge_attr"].to(device)
            target = batch["target_delta"].to(device)
            cs = batch["canvas_info"][0][0]
            pred = model(centers, feats, eidx, eattr, canvas_scale=cs)
            loss = (((pred - target) / cs) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            t_loss += loss.item()
            nb += 1
        t_loss /= max(1, nb)
        sched.step()

        model.eval()
        v_loss = 0.0
        vb = 0
        with torch.no_grad():
            for batch in val_loader:
                centers = batch["centers"].to(device)
                feats = batch["node_features"].to(device)
                eidx = batch["edge_index"].to(device)
                eattr = batch["edge_attr"].to(device)
                target = batch["target_delta"].to(device)
                cs = batch["canvas_info"][0][0]
                pred = model(centers, feats, eidx, eattr, canvas_scale=cs)
                loss = (((pred - target) / cs) ** 2).mean()
                v_loss += loss.item()
                vb += 1
        v_loss /= max(1, vb)

        is_best = v_loss < best_val
        if is_best:
            best_val = v_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "node_feat_dim": 5, "edge_feat_dim": 4,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers, "heads": args.heads,
                },
                "val_loss": v_loss, "epoch": epoch,
            }, out_path)

        marker = " [best]" if is_best else ""
        print(f"epoch {epoch+1:3d}/{args.epochs}  train={t_loss:.6f}  val={v_loss:.6f}  "
              f"time={time.time()-t0:.1f}s{marker}", flush=True)

    print(f"\nBest val loss: {best_val:.6f}  saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
