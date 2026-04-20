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
    """

    def __init__(self, pickle_path: str, repair_iters: int = 30, repair_step: float = 0.3):
        with open(pickle_path, "rb") as f:
            self.placements = pickle.load(f)
        self.repair_iters = repair_iters
        self.repair_step = repair_step

        # Pre-compute targets (hand-crafted repair output) — cheap CPU op
        self._targets = []
        for i, p in enumerate(self.placements):
            centers = p["centers_bad"]
            sizes = p["sizes"]
            cw = p["canvas_w"]
            ch = p["canvas_h"]
            repaired = local_repair_loop(
                centers, sizes, cw, ch,
                num_steps=repair_iters, step_size=repair_step,
            )
            self._targets.append(repaired - centers)

    def __len__(self):
        return len(self.placements)

    def __getitem__(self, idx: int):
        p = self.placements[idx]
        centers = p["centers_bad"]
        sizes = p["sizes"]
        cw = p["canvas_w"]
        ch = p["canvas_h"]

        node_features = compute_violation_features(centers, sizes, cw, ch)

        return {
            "centers": centers,
            "node_features": node_features,
            "edge_index": p["edge_index"],
            "edge_attr": p["edge_attr"],
            "target_delta": self._targets[idx],
            "sizes": sizes,
            "canvas_w": cw,
            "canvas_h": ch,
        }
