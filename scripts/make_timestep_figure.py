#!/usr/bin/env python3
"""Figure 8 (supplement): intra-sampling start_timestep sweep."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/ispd2005/intra_timestep_sweep.json"
OUT = ROOT / "paper/figures/fig8_timestep_sweep.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

d = json.load(open(IN))
by = defaultdict(list)
for r in d:
    by[(r["circuit"], r["start_timestep"])].append(r)

circuits = sorted({r["circuit"] for r in d})
ts = sorted({r["start_timestep"] for r in d})

fig, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(11, 4))
colors = {"adaptec3": "#2a9d8f", "bigblue3": "#e76f51"}

for c in circuits:
    xs, ys_v, ys_h, ys_v_std, ys_h_std = [], [], [], [], []
    for t in ts:
        rs = by.get((c, t), [])
        if not rs: continue
        dv = [(r["base_v"] - r["v"]) / max(r["base_v"], 1) * 100 for r in rs]
        dh = [(r["h"] - r["base_h"]) / max(r["base_h"], 1e-9) * 100 for r in rs]
        xs.append(t)
        ys_v.append(np.mean(dv)); ys_v_std.append(np.std(dv))
        ys_h.append(np.mean(dh)); ys_h_std.append(np.std(dh))
    ax_v.errorbar(xs, ys_v, yerr=ys_v_std, marker="o", capsize=3, label=c, color=colors.get(c))
    ax_h.errorbar(xs, ys_h, yerr=ys_h_std, marker="o", capsize=3, label=c, color=colors.get(c))

ax_v.set_xlabel(r"start timestep $t$")
ax_v.set_ylabel("Δviolations (%)")
ax_v.set_title("Intra-sampling: violations vs. $t$")
ax_v.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax_v.grid(alpha=0.3); ax_v.legend()

ax_h.set_xlabel(r"start timestep $t$")
ax_h.set_ylabel("ΔHPWL (%)")
ax_h.set_title("Intra-sampling: HPWL vs. $t$")
ax_h.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax_h.grid(alpha=0.3); ax_h.legend()

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=150)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"wrote {OUT}")
