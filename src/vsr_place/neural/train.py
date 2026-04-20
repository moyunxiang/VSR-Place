"""Training loop for NeuralVSR.

Trains the GNN to predict displacement (legal - bad) given (bad, feedback).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vsr_place.neural.dataset import SyntheticVSRDataset
from vsr_place.neural.model import NeuralVSR
from vsr_place.verifier.verifier import Verifier


def _collate_single(batch):
    """Collate a batch of variable-size graphs into one big graph.

    Uses PyG-style concatenation: stack node tensors, shift edge indices.
    """
    centers = []
    node_features = []
    edge_indices = []
    edge_attrs = []
    target_deltas = []
    offsets = []
    sizes_list = []
    canvas_info = []

    offset = 0
    for item in batch:
        n = item["centers"].shape[0]
        centers.append(item["centers"])
        node_features.append(item["node_features"])
        target_deltas.append(item["target_delta"])
        sizes_list.append(item["sizes"])
        edge_indices.append(item["edge_index"] + offset)
        edge_attrs.append(item["edge_attr"])
        offsets.append(offset)
        canvas_info.append((item["canvas_w"], item["canvas_h"], n))
        offset += n

    return {
        "centers": torch.cat(centers, dim=0),
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "target_delta": torch.cat(target_deltas, dim=0),
        "sizes": torch.cat(sizes_list, dim=0),
        "offsets": offsets,
        "canvas_info": canvas_info,
    }


def train(
    output_path: str,
    num_train: int = 10000,
    num_val: int = 500,
    batch_size: int = 16,
    num_epochs: int = 10,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 3,
    heads: int = 4,
    device: str = "cuda",
    n_range: tuple = (30, 300),
    canvas_side: float = 10.0,
    perturb_scale: float = 0.5,
    val_every: int = 1,
):
    """Train NeuralVSR on synthetic data."""
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"=== NeuralVSR Training ===")
    print(f"  train={num_train} val={num_val} bs={batch_size} epochs={num_epochs}")
    print(f"  arch: hidden={hidden_dim} layers={num_layers} heads={heads}")
    print(f"  device={device}")

    # Datasets (precompute once to avoid per-epoch regeneration)
    train_ds = SyntheticVSRDataset(
        num_samples=num_train, n_range=n_range,
        canvas_side=canvas_side, perturb_scale=perturb_scale, seed=0,
        precompute=True, verbose=True,
    )
    val_ds = SyntheticVSRDataset(
        num_samples=num_val, n_range=n_range,
        canvas_side=canvas_side, perturb_scale=perturb_scale, seed=10_000_000,
        precompute=True, verbose=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate_single, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=_collate_single, num_workers=0)

    # Model
    model = NeuralVSR(
        node_feat_dim=5, edge_feat_dim=4,
        hidden_dim=hidden_dim, num_layers=num_layers, heads=heads,
    ).to(device)
    print(f"  params: {model.num_parameters():,}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)

    best_val_loss = float("inf")
    best_ckpt_path = Path(output_path)
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            centers = batch["centers"].to(device)
            node_feats = batch["node_features"].to(device)
            edge_idx = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            target = batch["target_delta"].to(device)

            pred = model(centers, node_feats, edge_idx, edge_attr)
            loss = ((pred - target) ** 2).mean()

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)
        sched.step()
        train_time = time.time() - t0

        # Validate
        if (epoch + 1) % val_every == 0:
            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    centers = batch["centers"].to(device)
                    node_feats = batch["node_features"].to(device)
                    edge_idx = batch["edge_index"].to(device)
                    edge_attr = batch["edge_attr"].to(device)
                    target = batch["target_delta"].to(device)

                    pred = model(centers, node_feats, edge_idx, edge_attr)
                    loss = ((pred - target) ** 2).mean()
                    val_loss += loss.item()
                    n_val_batches += 1
            val_loss /= max(1, n_val_batches)

            # Save best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save({
                    "model_state": model.state_dict(),
                    "config": {
                        "node_feat_dim": 5, "edge_feat_dim": 4,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers, "heads": heads,
                    },
                    "val_loss": val_loss,
                    "epoch": epoch,
                }, best_ckpt_path)

            marker = " [best]" if is_best else ""
            print(f"epoch {epoch+1:3d}/{num_epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  time={train_time:.1f}s{marker}")
        else:
            print(f"epoch {epoch+1:3d}/{num_epochs}  train_loss={train_loss:.6f}  time={train_time:.1f}s")

    print(f"\nBest val loss: {best_val_loss:.6f}  saved to {best_ckpt_path}")
    return model, best_val_loss


def load_model(checkpoint_path: str, device: str = "cpu") -> NeuralVSR:
    """Load a trained NeuralVSR model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = NeuralVSR(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
