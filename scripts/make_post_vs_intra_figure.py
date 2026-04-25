#!/usr/bin/env python3
"""Figure showing bigblue3 baseline vs POST-VSR vs INTRA-VSR placements.

Demonstrates the dramatic HPWL difference visually.
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/placements/bigblue3_post_vs_intra.pkl"
OUT = ROOT / "paper/figures/fig9_post_vs_intra.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

d = pickle.loads(IN.read_bytes())
cw, ch = d["canvas_w"], d["canvas_h"]
sizes = d["sizes"]
stats = d["stats"]

VARIANTS = [
    ("baseline", f"Baseline\nv={stats['base_v']:,}  h={stats['base_h']:,.0f}"),
    ("post", f"Post-processing VSR\nv={stats['post_v']:,}  h={stats['post_h']:,.0f}"),
    ("intra", f"Intra-sampling VSR\nv={stats['intra_v']:,}  h={stats['intra_h']:,.0f}"),
]


def draw(ax, centers, sev, canvas_w, canvas_h, title, vmax):
    ax.add_patch(patches.Rectangle((0, 0), canvas_w, canvas_h,
        linewidth=1, edgecolor="black", facecolor="none"))
    norm = mcolors.LogNorm(vmin=max(1e-3, sev[sev > 0].min() if (sev > 0).any() else 1e-3),
                            vmax=max(vmax, 1e-2))
    cmap = plt.get_cmap("YlOrRd")
    order = np.argsort(sev)
    for i in order:
        cx, cy = centers[i]; w, h = sizes[i]
        s = sev[i]
        if s <= 1e-6:
            fc = "#cfd7e3"; ec = "#627490"; alpha = 0.55
        else:
            fc = cmap(norm(s)); ec = "#5a0000"; alpha = 0.9
        ax.add_patch(patches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            linewidth=0.15, edgecolor=ec, facecolor=fc, alpha=alpha,
        ))
    ax.set_xlim(-0.02 * canvas_w, 1.02 * canvas_w)
    ax.set_ylim(-0.02 * canvas_h, 1.02 * canvas_h)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


vmax = max(d["baseline_severity"].max(), d["post_severity"].max(), d["intra_severity"].max())
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (key, label) in zip(axes, VARIANTS):
    draw(ax, d[f"{key}_centers"], d[f"{key}_severity"], cw, ch, label, vmax=float(vmax))

fig.suptitle("bigblue3 (1298 macros): POST vs INTRA comparison", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=180)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=180)
print(f"wrote {OUT}")
