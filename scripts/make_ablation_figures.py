#!/usr/bin/env python3
"""Ablation figures: num_steps, step_size, selective on/off."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/ispd2005/ablations.json"
OUTDIR = ROOT / "paper/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]


def load():
    rows = json.load(open(IN))
    by = defaultdict(list)
    for r in rows:
        by[(r["circuit"], r["ablation"], r["value"])].append(r)
    return by


def metric(rows, kind):
    if not rows: return np.nan, np.nan
    if kind == "viol":
        vs = [(r["base_v"] - r["v"]) / max(r["base_v"], 1) * 100 for r in rows]
    else:
        vs = [(r["h"] - r["base_h"]) / max(r["base_h"], 1e-9) * 100 for r in rows]
    return np.mean(vs), np.std(vs)


def fig_num_steps(by):
    values = [25, 50, 100, 200, 500]
    fig, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(11, 4))
    for c in CIRCUITS:
        vs, vse, hs, hse = [], [], [], []
        for v in values:
            r = by.get((c, "num_steps", v), [])
            m, s = metric(r, "viol"); vs.append(-m); vse.append(s)
            m, s = metric(r, "hpwl"); hs.append(m); hse.append(s)
        ax_v.errorbar(values, vs, yerr=vse, marker="o", label=c, capsize=3)
        ax_h.errorbar(values, hs, yerr=hse, marker="o", label=c, capsize=3)
    for ax, title, ylabel in [(ax_v, "Violations vs. num_steps", "Δviolations (%)"),
                               (ax_h, "HPWL vs. num_steps", "ΔHPWL (%)")]:
        ax.set_xlabel("num_steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
    ax_v.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = OUTDIR / "fig5_num_steps.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


def fig_step_size(by):
    values = [0.1, 0.3, 0.5, 1.0]
    fig, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(11, 4))
    for c in CIRCUITS:
        vs, vse, hs, hse = [], [], [], []
        for v in values:
            r = by.get((c, "step_size", v), [])
            m, s = metric(r, "viol"); vs.append(-m); vse.append(s)
            m, s = metric(r, "hpwl"); hs.append(m); hse.append(s)
        ax_v.errorbar(values, vs, yerr=vse, marker="o", label=c, capsize=3)
        ax_h.errorbar(values, hs, yerr=hse, marker="o", label=c, capsize=3)
    for ax, title, ylabel in [(ax_v, "Violations vs. step_size", "Δviolations (%)"),
                               (ax_h, "HPWL vs. step_size", "ΔHPWL (%)")]:
        ax.set_xlabel("step_size (η)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.grid(alpha=0.3)
    ax_v.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = OUTDIR / "fig6_step_size.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


def table_selective(by):
    # Compare selective on vs off per-circuit
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Circuit & $\Delta$viol on & $\Delta$viol off & $\Delta$HPWL on & $\Delta$HPWL off \\",
        r"\midrule",
    ]
    for c in CIRCUITS:
        on = by.get((c, "selective", "on"), [])
        off = by.get((c, "selective", "off"), [])
        if not on or not off: continue
        v_on, _ = metric(on, "viol"); h_on, _ = metric(on, "hpwl")
        v_off, _ = metric(off, "viol"); h_off, _ = metric(off, "hpwl")
        lines.append(f"{c} & ${-v_on:+.1f}$ & ${-v_off:+.1f}$ & ${h_on:+.1f}$ & ${h_off:+.1f}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out = OUTDIR / "table_selective.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    if not IN.exists():
        raise SystemExit(f"Missing {IN}")
    by = load()
    fig_num_steps(by)
    fig_step_size(by)
    table_selective(by)
