#!/usr/bin/env python3
"""fig4 redo — per-circuit, per-method comparison from main_neurips.json.

Generates two side-by-side panels:
- left: Δviolation% (negative = improvement) per circuit, per method
- right: ΔHPWL% per circuit, per method
"""
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/ispd2005/main_neurips.json"
OUT = ROOT / "paper/figures/fig4_percircuit.pdf"

METHODS = ["vsr_post", "vsr_intra", "cd_std", "cd_sched", "cg_strong", "repaint_bin"]
LABEL = {
    "vsr_post": "VSR-post",
    "vsr_intra": "VSR-intra",
    "cd_std": "CD-std",
    "cd_sched": "CD-sched",
    "cg_strong": "cg-strong",
    "repaint_bin": "RePaint-bin",
}
COLOR = {
    "vsr_post": "#1f77b4",
    "vsr_intra": "#17becf",
    "cd_std": "#ff7f0e",
    "cd_sched": "#d62728",
    "cg_strong": "#9467bd",
    "repaint_bin": "#7f7f7f",
}
CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]


def main():
    rows = json.load(open(IN))

    # group by (circuit, method) -> list of (dv%, dh%)
    by = defaultdict(list)
    for r in rows:
        if r.get("error") or r.get("baseline_v") is None:
            continue
        bv = r["baseline_v"]
        bh = r["baseline_h"]
        for m in METHODS:
            v = r.get(f"{m}_v")
            h = r.get(f"{m}_h")
            if v is None or h is None:
                continue
            dv = (v - bv) / max(bv, 1) * 100
            dh = (h - bh) / max(abs(bh), 1e-9) * 100
            by[(r["circuit"], m)].append((dv, dh))

    fig, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(13, 5))

    n_circ = len(CIRCUITS)
    n_meth = len(METHODS)
    bar_w = 0.13
    x = np.arange(n_circ)

    # Two panels
    for ax, axis in ((ax_v, 0), (ax_h, 1)):
        for j, m in enumerate(METHODS):
            means = []
            stds = []
            for c in CIRCUITS:
                d = by.get((c, m), [])
                if not d:
                    means.append(0); stds.append(0); continue
                vals = [t[axis] for t in d]
                means.append(S.mean(vals))
                stds.append(S.pstdev(vals) if len(vals) > 1 else 0)
            offset = (j - (n_meth - 1) / 2) * bar_w
            ax.bar(x + offset, means, bar_w, yerr=stds, capsize=2,
                   color=COLOR[m], label=LABEL[m] if axis == 0 else None)

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(CIRCUITS, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        if axis == 0:
            ax.set_ylabel(r"$\Delta$violations (%, negative = improvement)")
            ax.set_title("Violation reduction per circuit")
        else:
            ax.set_ylabel(r"$\Delta$HPWL (%, negative = improvement)")
            ax.set_title("Wirelength change per circuit")

    ax_v.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=150)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
