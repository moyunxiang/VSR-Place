"""Ground-truth legal placement dataset for NeuralVSR.

Instead of teacher-distilling hand-crafted (noisy, HPWL-unaware), we use
the ISPD2005 original .pl file placements as targets. These are:
- Actually legal (from industrial placer)
- HPWL-optimized (contest winners)
- No approximation error

Training signal:
  input: guided_sampling_placement (has violations)
  target: ISPD2005_legal_placement - guided_sampling_placement

This should produce a model that learns "move from diffusion-bad toward
industrial-legal", which is exactly what we want at inference time.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vsr_place.neural.dataset import compute_violation_features


class GTLegalDataset(Dataset):
    """Dataset pairing guided placements with ISPD2005 ground-truth legal placements.

    Args:
        guided_pkl: path to pickle with list of guided placements
                    (e.g. data/ispd_placements_full.pkl)
        ispd_data_dir: path to ISPD2005 parsed pickles
                       (third_party/chipdiffusion/datasets/graph/ispd2005/)
        augmentations: number of random perturbations per sample (0 = none)
        perturb_scale: perturbation magnitude as fraction of canvas
    """

    def __init__(
        self,
        guided_pkl: str,
        ispd_data_dir: str,
        augmentations: int = 0,
        perturb_scale: float = 0.02,
    ):
        with open(guided_pkl, "rb") as f:
            self.guided = pickle.load(f)

        ispd_dir = Path(ispd_data_dir)

        # Load ground-truth legal placements per circuit
        self._gt_legal = {}
        for i in range(8):
            out_file = ispd_dir / f"output{i}.pickle"
            if out_file.exists():
                with open(out_file, "rb") as f:
                    gt = pickle.load(f)
                if not isinstance(gt, torch.Tensor):
                    gt = torch.tensor(gt, dtype=torch.float32)
                self._gt_legal[i] = gt

        # For each guided placement, need to extract only the macro rows
        # (guided was already filtered to macro-only; ground truth is full)
        # We match by computing which nodes are macros in the original cond.
        # But we don't have access to the original cond here — we need to load it.
        self._gt_legal_macros = {}
        for i in self._gt_legal:
            graph_file = ispd_dir / f"graph{i}.pickle"
            with open(graph_file, "rb") as f:
                cond = pickle.load(f)
            if hasattr(cond, "is_macros") and cond.is_macros is not None:
                mask = cond.is_macros.bool()
                self._gt_legal_macros[i] = self._gt_legal[i][mask]
            else:
                self._gt_legal_macros[i] = self._gt_legal[i]

        # Build items list
        self._items = []
        for g_idx, g in enumerate(self.guided):
            c_idx = g["circuit_idx"]
            if c_idx not in self._gt_legal_macros:
                continue
            gt_legal = self._gt_legal_macros[c_idx]
            if gt_legal.shape[0] != g["centers_bad"].shape[0]:
                print(f"WARNING: shape mismatch circuit {c_idx}: "
                      f"gt {gt_legal.shape} vs guided {g['centers_bad'].shape}",
                      flush=True)
                continue
            # Original (no perturbation)
            self._items.append((g_idx, None))
            # Augmentations
            for k in range(augmentations):
                self._items.append((g_idx, g_idx * 10000 + k))

        self.perturb_scale = perturb_scale
        self._cache = {}
        print(f"GTLegalDataset: {len(self.guided)} guided + {len(self._items)} total items "
              f"({len(self._gt_legal_macros)} circuits have GT)", flush=True)

    def __len__(self):
        return len(self._items)

    def _compute(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        g_idx, seed = self._items[idx]
        g = self.guided[g_idx]
        c_idx = g["circuit_idx"]

        centers = g["centers_bad"].clone()
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))
            scale = self.perturb_scale * max(g["canvas_w"], g["canvas_h"])
            noise = torch.randn(centers.shape, generator=gen) * scale
            centers = centers + noise

        gt_legal = self._gt_legal_macros[c_idx]
        target_delta = gt_legal - centers

        node_features = compute_violation_features(
            centers, g["sizes"], g["canvas_w"], g["canvas_h"],
        )

        item = {
            "centers": centers.detach().clone(),
            "node_features": node_features.detach().clone(),
            "edge_index": g["edge_index"].detach().clone(),
            "edge_attr": (g["edge_attr"].detach().clone()
                          if g["edge_attr"] is not None else None),
            "target_delta": target_delta.detach().clone(),
            "sizes": g["sizes"],
            "canvas_w": g["canvas_w"],
            "canvas_h": g["canvas_h"],
        }
        self._cache[idx] = item
        return item

    def __getitem__(self, idx: int):
        item = self._compute(idx)
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
