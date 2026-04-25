#!/usr/bin/env python3
"""Figure 7 (supplement) — per-iteration convergence curves."""
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/ispd2005/convergence.pkl"
OUT = ROOT / "paper/figures/fig7_convergence.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

data = pickle.loads(IN.read_bytes())

# Group by circuit
bycirc = defaultdict(list)
for (name, seed), traj in data.items():
    bycirc[name].append((seed, traj))

circuits = sorted(bycirc.keys())
fig, axes = plt.subplots(2, len(circuits), figsize=(5 * len(circuits), 6), squeeze=False)

for j, name in enumerate(circuits):
    trajs = bycirc[name]
    # Stack
    V = np.array([t["violations"] for _, t in trajs])  # (n_seeds, T+1)
    H = np.array([t["hpwls"] for _, t in trajs])
    # Normalize to % change
    base_v = V[:, 0:1]
    base_h = H[:, 0:1]
    dv = (V - base_v) / np.maximum(base_v, 1) * 100
    dh = (H - base_h) / np.maximum(base_h, 1e-9) * 100
    iters = np.arange(V.shape[1])

    # Violations
    ax = axes[0, j]
    mean_v = dv.mean(axis=0); std_v = dv.std(axis=0)
    ax.plot(iters, mean_v, color="#c0392b", linewidth=1.5)
    ax.fill_between(iters, mean_v - std_v, mean_v + std_v, color="#c0392b", alpha=0.2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("Δviolations (%)")
    ax.set_title(f"{name} — violations")
    ax.grid(alpha=0.3)

    # HPWL
    ax = axes[1, j]
    mean_h = dh.mean(axis=0); std_h = dh.std(axis=0)
    ax.plot(iters, mean_h, color="#2980b9", linewidth=1.5)
    ax.fill_between(iters, mean_h - std_h, mean_h + std_h, color="#2980b9", alpha=0.2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("ΔHPWL (%)")
    ax.set_title(f"{name} — HPWL")
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=150)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"wrote {OUT}")
