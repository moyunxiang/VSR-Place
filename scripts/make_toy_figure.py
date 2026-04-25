#!/usr/bin/env python3
"""Toy 2D cross-domain demo figure."""
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/toy/toy_results.json"
OUT = ROOT / "paper/figures/fig_toy_2d.pdf"

METHODS = ["baseline", "classifier_guide", "repaint_binary", "vsr_intra", "vsr_post"]
LABEL = {
    "baseline": "raw sample",
    "classifier_guide": "classifier-\nguidance",
    "repaint_binary": "RePaint\n(binary)",
    "vsr_intra": "VSR-intra",
    "vsr_post": "VSR-post",
}
COLOR = {
    "baseline": "#999999",
    "classifier_guide": "#9467bd",
    "repaint_binary": "#7f7f7f",
    "vsr_intra": "#17becf",
    "vsr_post": "#1f77b4",
}


def main():
    data = json.load(open(IN))
    rows = data["rows"]
    cfg = data["config"]
    print(f"toy config: {cfg}")

    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metric_names = ["viol_count", "overlap_area", "wirelength"]
    metric_titles = ["Violations\n(# overlapping pairs + boundary)", "Total overlap area", "Wirelength"]

    for ax, mn, mt in zip(axes, metric_names, metric_titles):
        means = []
        stds = []
        for m in METHODS:
            vals = [r[mn] for r in by[m]]
            means.append(S.mean(vals))
            stds.append(S.pstdev(vals) if len(vals) > 1 else 0)
        bars = ax.bar(range(len(METHODS)), means, yerr=stds, capsize=4,
                      color=[COLOR[m] for m in METHODS], edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=10, ha="right", fontsize=9)
        ax.set_title(mt, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Toy 2D: {cfg['n_disks']} disks $\\times$ {cfg['n_test']} test prompts $\\times$ "
                 f"{cfg['n_seeds']} seeds = {cfg['n_test']*cfg['n_seeds']} runs/method "
                 f"(denoiser trained on {cfg['n_train']} legal layouts, {cfg['train_steps']} steps)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=150)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
