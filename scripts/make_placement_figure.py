#!/usr/bin/env python3
"""Figure 2 — Placement visualizations before/after repair.

Grid: rows = circuits (adaptec1, adaptec3). Cols = baseline, w=0 (viol-only), w=2 (HPWL-aware).
Macros colored red if violating, blue if legal.
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PL_DIR = ROOT / "results/placements"
OUT = ROOT / "paper/figures/fig2_placements.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

CIRCUITS = ["adaptec1", "adaptec3"]
VARIANTS = [
    ("baseline", "Baseline (guided diffusion)"),
    ("w0", r"VSR, $\lambda=0$ (violations only)"),
    ("w2", r"VSR, $\lambda=2$ (HPWL-aware)"),
]


def draw(ax, centers, sizes, severity, canvas_w, canvas_h, title, vmax):
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    ax.add_patch(patches.Rectangle((0, 0), canvas_w, canvas_h,
        linewidth=1, edgecolor="black", facecolor="none"))

    # Colormap: viridis_r (dark = legal, yellow = heavy violation), log scale
    norm = mcolors.LogNorm(vmin=max(1e-3, severity[severity > 0].min() if (severity > 0).any() else 1e-3),
                            vmax=max(vmax, 1e-2))
    cmap = cm.get_cmap("YlOrRd")

    # Draw legal first, then violators on top
    order = np.argsort(severity)
    for i in order:
        cx, cy = centers[i]; w, h = sizes[i]
        s = severity[i]
        if s <= 1e-6:
            fc = "#cfd7e3"; ec = "#627490"; alpha = 0.6
        else:
            rgba = cmap(norm(s))
            fc = rgba; ec = "#5a0000"; alpha = 0.9
        ax.add_patch(patches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            linewidth=0.2, edgecolor=ec, facecolor=fc, alpha=alpha,
        ))
    ax.set_xlim(-0.02 * canvas_w, 1.02 * canvas_w)
    ax.set_ylim(-0.02 * canvas_h, 1.02 * canvas_h)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


fig, axes = plt.subplots(len(CIRCUITS), len(VARIANTS), figsize=(12, 8))
if len(CIRCUITS) == 1:
    axes = np.array([axes])

for i, cname in enumerate(CIRCUITS):
    pkl = pickle.loads((PL_DIR / f"{cname}.pkl").read_bytes())
    cw, ch = pkl["canvas_w"], pkl["canvas_h"]
    # Per-row common severity color scale (baseline max)
    vmax = float(pkl["baseline_severity"].max())
    for j, (key, label) in enumerate(VARIANTS):
        centers = pkl[f"{key}_centers"]
        sev = pkl[f"{key}_severity"]
        sizes = pkl["sizes"]
        stats = pkl["stats"]
        v_key = "baseline_v" if key == "baseline" else f"{key}_v"
        h_key = "baseline_h" if key == "baseline" else f"{key}_h"
        sub = f"viol={stats[v_key]:,}    HPWL={stats[h_key]:,.1f}"
        title = f"{label}\n{sub}" if i == 0 else sub
        draw(axes[i, j], centers, sizes, sev, cw, ch, title, vmax)
    axes[i, 0].set_ylabel(cname, fontsize=12, weight="bold")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
print(f"wrote {OUT}")
