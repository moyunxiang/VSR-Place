"""fig9 (NeurIPS version): bigblue3 baseline / VSR-post / VSR-intra-soft.

Replaces the old binary-RePaint+post-fix protocol viz with the actual
soft-mask intra-sampling protocol used in the main paper.
Source: results/placements/bigblue3_neurips.pkl
Output: paper/figures/fig9_post_vs_intra.{pdf,png}
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/placements/bigblue3_neurips.pkl"
OUT = ROOT / "paper/figures/fig9_post_vs_intra.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

d = pickle.loads(IN.read_bytes())
cw, ch = d["canvas_w"], d["canvas_h"]
sizes = d["sizes"]
stats = d["stats"]

# Caption-friendly variant titles
VARIANTS = [
    ("baseline", "Raw diffusion draft"),
    ("vsr_post", "VSR-post (force integrator)"),
    ("vsr_intra", "VSR-intra-soft (structured RePaint)"),
]


def draw(ax, centers, sev, canvas_w, canvas_h, title, vmax):
    ax.add_patch(patches.Rectangle((0, 0), canvas_w, canvas_h,
        linewidth=1, edgecolor="black", facecolor="none"))
    norm = mcolors.LogNorm(
        vmin=max(1e-3, sev[sev > 0].min() if (sev > 0).any() else 1e-3),
        vmax=max(vmax, 1e-2))
    cmap = plt.get_cmap("YlOrRd")
    order = np.argsort(sev)
    for i in order:
        cx, cy = centers[i]
        w, h = sizes[i]
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


vmax = max(d["baseline_severity"].max(),
           d["vsr_post_severity"].max(),
           d["vsr_intra_severity"].max())
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

for ax, (key, label) in zip(axes, VARIANTS):
    centers = d[f"{key}_centers"] if key != "baseline" else d["baseline_centers"]
    sev = d[f"{key}_severity"] if key != "baseline" else d["baseline_severity"]
    draw(ax, centers, sev, cw, ch, label, vmax)

# Subtitle with quantitative numbers from the matching protocol.
fig.suptitle(
    f"bigblue3 (1298 macros, seed=42).  "
    f"Baseline: $v$={stats['base_v']:,}, $h$={stats['base_h']:,.0f}.  "
    f"VSR-post: $v$={stats['vsr_post_v']:,} "
    f"({(stats['base_v']-stats['vsr_post_v'])/stats['base_v']*100:+.1f}\\%), "
    f"$h$={stats['vsr_post_h']:,.0f} "
    f"({(stats['vsr_post_h']-stats['base_h'])/abs(stats['base_h'])*100:+.1f}\\%).  "
    f"VSR-intra-soft: $v$={stats['vsr_intra_v']:,} "
    f"({(stats['base_v']-stats['vsr_intra_v'])/stats['base_v']*100:+.1f}\\%), "
    f"$h$={stats['vsr_intra_h']:,.0f} "
    f"({(stats['vsr_intra_h']-stats['base_h'])/abs(stats['base_h'])*100:+.1f}\\%).",
    fontsize=8, y=0.02,
)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT, bbox_inches="tight", dpi=150)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"wrote {OUT}")
print(f"  baseline:  v={stats['base_v']}  h={stats['base_h']:.1f}")
print(f"  vsr_post:  v={stats['vsr_post_v']}  h={stats['vsr_post_h']:.1f}")
print(f"  vsr_intra: v={stats['vsr_intra_v']}  h={stats['vsr_intra_h']:.1f}")
