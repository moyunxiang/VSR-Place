"""Real-data training for NeuralVSR with trajectory distillation.

Key idea: instead of training to predict the full repair trajectory in
one shot, we train to predict a SINGLE step of hand-crafted repair.
This greatly expands our training set from 3 real placements to
3 × K trajectory pairs, and teaches the model "local physics" rather
than "global repair" — which generalizes much better across circuits.
"""

from __future__ import annotations

import pickle

import torch
from torch.utils.data import Dataset

from vsr_place.neural.dataset import compute_violation_features
from vsr_place.renoising.local_repair import local_repair_loop


class RealISPDDataset(Dataset):
    """Dataset built from ISPD2005 guided placements with trajectory distillation.

    Args:
        pickle_path: list of placements.
        trajectory_steps: how many hand-crafted steps to unroll per sample.
            Each step → one training pair (x_k → x_{k+1}).
        repair_step: hand-crafted step_size.
    """

    def __init__(
        self,
        pickle_path: str,
        trajectory_steps: int = 30,
        repair_step: float = 0.3,
    ):
        with open(pickle_path, "rb") as f:
            self.placements = pickle.load(f)

        print(f"Building trajectory dataset from {len(self.placements)} real placements "
              f"× {trajectory_steps} steps...", flush=True)

        self._items = []  # flat list of (placement_idx, step_idx, x_prev, x_next)
        for p_idx, p in enumerate(self.placements):
            centers = p["centers_bad"]
            sizes = p["sizes"]
            cw = p["canvas_w"]
            ch = p["canvas_h"]

            pairs = local_repair_loop(
                centers, sizes, cw, ch,
                num_steps=trajectory_steps, step_size=repair_step,
                return_trajectory=True,
            )
            for step_idx, (x_prev, x_next) in enumerate(pairs):
                self._items.append({
                    "p_idx": p_idx,
                    "step_idx": step_idx,
                    "centers": x_prev.detach().clone(),
                    "target_next": x_next.detach().clone(),
                })
            if (p_idx + 1) % 10 == 0 or p_idx == len(self.placements) - 1:
                print(f"  {p_idx+1}/{len(self.placements)} ({len(self._items)} pairs)",
                      flush=True)

        # Pre-compute node features per item
        print(f"Computing violation features for {len(self._items)} items...", flush=True)
        for item in self._items:
            p = self.placements[item["p_idx"]]
            item["node_features"] = compute_violation_features(
                item["centers"], p["sizes"], p["canvas_w"], p["canvas_h"],
            ).detach().clone()
            item["target_delta"] = (item["target_next"] - item["centers"]).detach().clone()

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx: int):
        item = self._items[idx]
        p = self.placements[item["p_idx"]]
        return {
            "centers": item["centers"].clone(),
            "node_features": item["node_features"].clone(),
            "edge_index": p["edge_index"],
            "edge_attr": p["edge_attr"],
            "target_delta": item["target_delta"].clone(),
            "sizes": p["sizes"],
            "canvas_w": p["canvas_w"],
            "canvas_h": p["canvas_h"],
        }
