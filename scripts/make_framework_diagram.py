#!/usr/bin/env python3
"""Figure 1 — Framework diagram for the paper.

Three-stage pipeline: Verifier → Selector → Repair Operator.
Input: diffusion-generated placement. Output: repaired placement (residual violations may remain).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

OUT = Path(__file__).resolve().parents[1] / "paper/figures/fig1_framework.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, label, facecolor, edgecolor="black", fontsize=10, subtext=None):
    rect = patches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(rect)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if subtext:
        ax.text(cx, cy + 0.06, label, ha="center", va="center", fontsize=fontsize, weight="bold")
        ax.text(cx, cy - 0.10, subtext, ha="center", va="center", fontsize=fontsize - 2, style="italic", color="#444")
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize, weight="bold")


def arrow(ax, x1, y1, x2, y2, label=None, color="black"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=color),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, label,
                ha="center", va="bottom", fontsize=8.5, color=color, style="italic")


fig, ax = plt.subplots(figsize=(11, 3.8))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")

# Stage boxes
box(ax, (0.1, 1.0), 1.8, 1.2, "Diffusion\nbackbone", "#E8F1FA",
    subtext="ChipDiffusion")
box(ax, (2.6, 1.0), 2.0, 1.2, "Verifier", "#FFE9C2",
    subtext="structured feedback\n(piecewise diff.)")
box(ax, (5.4, 1.0), 2.0, 1.2, "Selector", "#DDF2D9",
    subtext=r"offender mask $M$")
box(ax, (8.2, 1.0), 2.0, 1.2, "Repair operator", "#FFD4D4",
    subtext="post-process\nor intra-sample")
box(ax, (10.7, 1.0), 1.2, 1.2, "Repaired\nplacement", "#E8E8FF")

# Arrows
arrow(ax, 1.9, 1.6, 2.6, 1.6, r"$\hat{x}_0$")
arrow(ax, 4.6, 1.6, 5.4, 1.6, r"$(r_i, A_{ij}, b_i)$")
arrow(ax, 7.4, 1.6, 8.2, 1.6, r"$M \odot \hat{x}_0$")
arrow(ax, 10.2, 1.6, 10.7, 1.6, r"$x^\star$")

# Feedback arrow (top)
arrow(ax, 9.2, 2.2, 3.6, 2.7)
ax.plot([3.6, 3.6], [2.7, 2.2], "k-", lw=1.5)
ax.text(6.4, 2.78, "re-verify (optional closed loop)",
        ha="center", va="bottom", fontsize=9, style="italic", color="#333")

# Bottom labels
ax.text(1.0, 0.7, r"$x_0$ sample", ha="center", fontsize=8.5, style="italic", color="#333")
ax.text(3.6, 0.7, "severity scalar\noverlap graph\nboundary vector", ha="center", fontsize=8, color="#333")
ax.text(6.4, 0.7, "soft mask\n$M_i = r_i / \\max r$", ha="center", fontsize=8, color="#333")
ax.text(9.2, 0.7, "100-step force\nintegrator $\\lambda{=}2$\n(§\\,3.3)", ha="center", fontsize=8, color="#333")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
print(f"wrote {OUT}")
