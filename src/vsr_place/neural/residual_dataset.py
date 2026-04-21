"""Residual learning dataset for NeuralVSR.

Strategy: let hand-crafted do 80% of the work (fast, local physics),
train NeuralVSR to predict the residual correction that moves from
hand-crafted output toward ISPD2005 ground-truth legal placement.

Training pair:
  input: (centers_hand, violation_features_at_hand, graph)
  target: x_gt_legal - centers_hand

This is easier than learning "bad → legal" because:
- Hand-crafted output is already mostly legal (small residual)
- Residual targets have smaller magnitude → easier to learn
- Model acts as a fine-tuning refinement, not a replacement
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vsr_place.neural.dataset import compute_violation_features
from vsr_place.renoising.local_repair import local_repair_loop


class ResidualDataset(Dataset):
    """Dataset pairing hand-crafted-repaired placements with GT residuals.

    Args:
        guided_pkl: list of {centers_bad, sizes, edge_index, edge_attr, canvas_w/h, circuit_idx}.
        ispd_data_dir: ISPD2005 parsed pickles (output{i}.pickle has GT legal positions).
        hpwl_weight: hand-crafted hpwl attraction weight.
        repair_steps: hand-crafted iterations.
        repair_step: hand-crafted step size.
        augmentations: random perturbations per sample.
        perturb_scale: perturbation magnitude (fraction of canvas).
    """

    def __init__(
        self,
        guided_pkl: str,
        ispd_data_dir: str,
        hpwl_weight: float = 2.0,
        repair_steps: int = 100,
        repair_step: float = 0.3,
        augmentations: int = 0,
        perturb_scale: float = 0.02,
    ):
        with open(guided_pkl, "rb") as f:
            self.guided = pickle.load(f)

        ispd_dir = Path(ispd_data_dir)

        # Load GT legal per circuit, filter to macros
        self._gt_macros = {}
        for i in range(8):
            out_f = ispd_dir / f"output{i}.pickle"
            graph_f = ispd_dir / f"graph{i}.pickle"
            if not (out_f.exists() and graph_f.exists()):
                continue
            with open(out_f, "rb") as f:
                gt = pickle.load(f)
            if not isinstance(gt, torch.Tensor):
                gt = torch.tensor(gt, dtype=torch.float32)
            with open(graph_f, "rb") as f:
                cond = pickle.load(f)
            if hasattr(cond, "is_macros") and cond.is_macros is not None:
                mask = cond.is_macros.bool()
                self._gt_macros[i] = gt[mask]
            else:
                self._gt_macros[i] = gt

        self.hpwl_weight = hpwl_weight
        self.repair_steps = repair_steps
        self.repair_step = repair_step
        self.perturb_scale = perturb_scale

        # Build items (with optional augmentations)
        self._items = []
        for g_idx, g in enumerate(self.guided):
            c_idx = g["circuit_idx"]
            if c_idx not in self._gt_macros:
                continue
            if self._gt_macros[c_idx].shape[0] != g["centers_bad"].shape[0]:
                continue
            self._items.append((g_idx, None))
            for k in range(augmentations):
                self._items.append((g_idx, g_idx * 10000 + k))

        # Precompute hand-crafted outputs & residual targets (CPU, eager)
        print(f"ResidualDataset: precomputing hand-crafted repair for "
              f"{len(self._items)} items...", flush=True)
        self._cache = {}
        for idx in range(len(self._items)):
            self._compute(idx)
            if (idx + 1) % 20 == 0:
                print(f"  {idx+1}/{len(self._items)}", flush=True)
        print("  done", flush=True)

    def __len__(self):
        return len(self._items)

    def _compute(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        g_idx, seed = self._items[idx]
        g = self.guided[g_idx]
        c_idx = g["circuit_idx"]

        centers_start = g["centers_bad"].clone()
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))
            scale = self.perturb_scale * max(g["canvas_w"], g["canvas_h"])
            centers_start = centers_start + torch.randn(centers_start.shape, generator=gen) * scale

        # Run hand-crafted repair from the (possibly perturbed) start
        # Get offending mask at start
        from vsr_place.verifier.verifier import Verifier
        v = Verifier(canvas_width=g["canvas_w"], canvas_height=g["canvas_h"])
        fb = v(centers_start, g["sizes"])
        offend_mask = fb.severity_vector > 0

        centers_hand = local_repair_loop(
            centers_start, g["sizes"], g["canvas_w"], g["canvas_h"],
            num_steps=self.repair_steps,
            step_size=self.repair_step,
            only_mask=offend_mask,
            edge_index=g["edge_index"],
            hpwl_weight=self.hpwl_weight,
        )

        # Residual target
        gt_legal = self._gt_macros[c_idx]
        target_residual = gt_legal - centers_hand

        # Violation features at hand-crafted state (input to model)
        node_features = compute_violation_features(
            centers_hand, g["sizes"], g["canvas_w"], g["canvas_h"],
        )

        item = {
            "centers": centers_hand.detach().clone(),
            "node_features": node_features.detach().clone(),
            "edge_index": g["edge_index"].detach().clone(),
            "edge_attr": (g["edge_attr"].detach().clone()
                          if g["edge_attr"] is not None else None),
            "target_delta": target_residual.detach().clone(),
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
