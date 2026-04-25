#!/usr/bin/env python3
"""Paper figures from Pareto sweep data.

Reads results/ispd2005/pareto_3seed_6w.json and produces:
- fig3_pareto.pdf: violation% vs HPWL% scatter per circuit, with curve per seed
- fig4_percircuit.pdf: per-circuit bar chart at w=2.0 (headline setting)
- table1_main.tex: LaTeX table of main results at w=2.0
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/ispd2005/pareto_3seed_6w.json"
OUTDIR = ROOT / "paper/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]
WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def load():
    rows = json.load(open(IN))
    by = defaultdict(list)  # (circuit, w) -> list of rows across seeds
    for r in rows:
        by[(r["circuit"], r["hpwl_weight"])].append(r)
    return by


def fig_pareto(by):
    # Colormap per weight — interpretable ordering
    cmap = plt.get_cmap("viridis")
    wcolors = {w: cmap(i / (len(WEIGHTS) - 1)) for i, w in enumerate(WEIGHTS)}

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=False)
    for ax, c in zip(axes.flat, CIRCUITS):
        # Scatter: per-seed points
        for w in WEIGHTS:
            rows = by.get((c, w), [])
            if not rows: continue
            hs = [r["hpwl_delta_pct"] for r in rows]
            # Negate viol_reduction_pct so y-axis is "Δviolations%" (lower is better)
            vs = [-r["viol_reduction_pct"] for r in rows]
            ax.scatter(hs, vs, s=40, alpha=0.55, color=wcolors[w])

        # Mean-line with error bars, annotated with lambda
        xs, ys, xerr, yerr, wlabels = [], [], [], [], []
        for w in WEIGHTS:
            rows = by.get((c, w), [])
            if not rows: continue
            h = [r["hpwl_delta_pct"] for r in rows]
            v = [-r["viol_reduction_pct"] for r in rows]
            xs.append(np.mean(h)); ys.append(np.mean(v))
            xerr.append(np.std(h)); yerr.append(np.std(v))
            wlabels.append(w)
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, color="black",
                    linewidth=1.2, capsize=2, marker="o", markersize=5, alpha=0.85)
        for x, y, wl in zip(xs, ys, wlabels):
            ax.annotate(f"λ={wl}", (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

        ax.set_title(c, fontsize=11, weight="bold")
        ax.set_xlabel("ΔHPWL (%)")
        ax.set_ylabel("Δviolations (%)")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.grid(alpha=0.3)

        # Shade Pareto-dominant region (both x<0 and y<0)
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.fill_between([min(xlim[0], 0), 0], min(ylim[0], 0), 0,
                         color="#c8e6c9", alpha=0.3, label="_nolegend_")

    fig.tight_layout()
    out = OUTDIR / "fig3_pareto.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


def fig_percircuit(by, w_head=2.0):
    fig, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(len(CIRCUITS))
    vs, vs_std, hs, hs_std = [], [], [], []
    for c in CIRCUITS:
        rows = by.get((c, w_head), [])
        if not rows:
            vs.append(0); vs_std.append(0); hs.append(0); hs_std.append(0); continue
        v = [r["viol_reduction_pct"] for r in rows]
        h = [r["hpwl_delta_pct"] for r in rows]
        vs.append(np.mean(v)); vs_std.append(np.std(v))
        hs.append(np.mean(h)); hs_std.append(np.std(h))

    ax_v.bar(xs, vs, yerr=vs_std, color="#2a9d8f", capsize=4)
    ax_v.set_xticks(xs); ax_v.set_xticklabels(CIRCUITS, rotation=20)
    ax_v.set_ylabel("Violation reduction (%)")
    ax_v.set_title(f"Violations @ w={w_head}")
    ax_v.axhline(0, color="k", linewidth=0.5)
    ax_v.grid(axis="y", alpha=0.3)

    ax_h.bar(xs, hs, yerr=hs_std, color="#e76f51", capsize=4)
    ax_h.set_xticks(xs); ax_h.set_xticklabels(CIRCUITS, rotation=20)
    ax_h.set_ylabel("HPWL change (%)")
    ax_h.set_title(f"HPWL @ w={w_head}")
    ax_h.axhline(0, color="k", linewidth=0.5)
    ax_h.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUTDIR / "fig4_percircuit.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


def table_main(by, w_head=2.0):
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Circuit & \# macros & baseline viol. & repaired viol. & $\Delta$viol.\% & $\Delta$HPWL\% \\",
        r"\midrule",
    ]
    all_v, all_h = [], []
    for c in CIRCUITS:
        rows = by.get((c, w_head), [])
        if not rows:
            continue
        n = rows[0]["n_macros"]
        bv = np.mean([r["baseline_violations"] for r in rows])
        rv = np.mean([r["repaired_violations"] for r in rows])
        dv = np.mean([r["viol_reduction_pct"] for r in rows])
        dh = np.mean([r["hpwl_delta_pct"] for r in rows])
        all_v.append(dv); all_h.append(dh)
        lines.append(
            f"{c} & {n} & {bv:,.0f} & {rv:,.0f} & ${-dv:+.1f}$ & ${dh:+.1f}$ \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        f"Mean & --- & --- & --- & ${-np.mean(all_v):+.1f}$ & ${np.mean(all_h):+.1f}$ \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out = OUTDIR / "table1_main.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def summary(by):
    print("\n=== Pareto summary ===")
    for c in CIRCUITS:
        print(f"\n{c}:")
        for w in WEIGHTS:
            rows = by.get((c, w), [])
            if not rows:
                continue
            v = np.mean([r["viol_reduction_pct"] for r in rows])
            h = np.mean([r["hpwl_delta_pct"] for r in rows])
            print(f"  w={w:<4}  Δviol={-v:+6.1f}%  ΔHPWL={h:+7.1f}%  ({len(rows)} seeds)")


if __name__ == "__main__":
    if not IN.exists():
        raise SystemExit(f"Missing {IN}")
    by = load()
    summary(by)
    fig_pareto(by)
    fig_percircuit(by)
    table_main(by)
