"""Real-data training for NeuralVSR.

Generate training pairs from actual ISPD2005 circuits instead of synthetic:
1. For each circuit, run ChipDiffusion guided sampling with many seeds.
2. Each sample is (guided placement, violations, hand-crafted repair target).
3. Train NeuralVSR on these in-distribution samples.

This closes the sim-to-real gap by training directly on the target distribution.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vsr_place.neural.dataset import compute_violation_features
from vsr_place.renoising.local_repair import local_repair_loop


class RealISPDDataset(Dataset):
    """Dataset built from pre-computed ISPD2005 guided placements.

    Expects a pickle file with list of dicts:
        {
          "centers_bad": (N, 2),
          "sizes": (N, 2),
          "edge_index": (2, E),
          "edge_attr": (E, 4),
          "canvas_w": float,
          "canvas_h": float,
        }

    Args:
        pickle_path: path to placements pickle.
        repair_iters: hand-crafted iterations for target.
        repair_step: hand-crafted step_size.
        augmentations_per_sample: how many perturbed versions to generate
            per real placement (e.g. 30 means 1 real → 31 training points).
        perturb_scale: stddev of perturbation as fraction of canvas (e.g. 0.02).
    """

    def __init__(
        self,
        pickle_path: str,
        repair_iters: int = 30,
        repair_step: float = 0.3,
        augmentations_per_sample: int = 0,
        perturb_scale: float = 0.02,
    ):
        with open(pickle_path, "rb") as f:
            self.placements = pickle.load(f)
        self.repair_iters = repair_iters
        self.repair_step = repair_step

        # Build augmented sample list: each item points to (placement_idx, perturb_seed_or_None)
        # perturb=None means use original, otherwise use seed for reproducible perturbation.
        self._items = []
        for p_idx in range(len(self.placements)):
            self._items.append((p_idx, None))  # original
            for k in range(augmentations_per_sample):
                self._items.append((p_idx, p_idx * 10000 + k))  # augmented

        self.perturb_scale = perturb_scale
        self._cache = {}

        # Eagerly precompute targets for all (uses CPU)
        print(f"Precomputing targets for {len(self._items)} samples "
              f"({len(self.placements)} real + {len(self._items) - len(self.placements)} augmented)...",
              flush=True)
        for i in range(len(self._items)):
            self._compute(i)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(self._items)}", flush=True)

    def __len__(self):
        return len(self._items)

    def _compute(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        p_idx, seed = self._items[idx]
        p = self.placements[p_idx]
        centers = p["centers_bad"].clone()
        sizes = p["sizes"]
        cw = p["canvas_w"]
        ch = p["canvas_h"]

        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
            scale = self.perturb_scale * max(cw, ch)
            noise = torch.randn(centers.shape, generator=g) * scale
            centers = centers + noise

        # Hand-crafted target
        repaired = local_repair_loop(
            centers, sizes, cw, ch,
            num_steps=self.repair_iters, step_size=self.repair_step,
        )
        target = repaired - centers
        node_features = compute_violation_features(centers, sizes, cw, ch)

        # Detach everything and ensure no grad history in cached items
        item = {
            "centers": centers.detach().clone(),
            "node_features": node_features.detach().clone(),
            "edge_index": p["edge_index"].detach().clone(),
            "edge_attr": p["edge_attr"].detach().clone() if p["edge_attr"] is not None else None,
            "target_delta": target.detach().clone(),
            "sizes": sizes.detach().clone() if isinstance(sizes, torch.Tensor) else sizes,
            "canvas_w": cw,
            "canvas_h": ch,
        }
        self._cache[idx] = item
        return item

    def __getitem__(self, idx: int):
        item = self._compute(idx)
        # Return fresh tensor views to avoid autograd entanglement across epochs
        return {
            "centers": item["centers"].clone(),
            "node_features": item["node_features"].clone(),
            "edge_index": item["edge_index"],
            "edge_attr": item["edge_attr"],
            "target_delta": item["target_delta"].clone(),
            "sizes": item["sizes"],
            "canvas_w": item["canvas_w"],
            "canvas_h": item["canvas_h"],
        }
