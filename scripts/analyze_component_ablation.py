"""Analyze component ablation results from results/vsr_extra/component_ablation.json.

Produces:
  - results/vsr_extra/component_ablation.csv
  - results/vsr_extra/component_ablation.md
  - paper/figures/table_component_ablation.tex

Pure local.
"""
from __future__ import annotations

import csv
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "results" / "vsr_extra" / "component_ablation.json"
OUT_DIR = REPO / "results" / "vsr_extra"
PAPER_FIG = REPO / "paper" / "figures"

ORDER = ["full", "severity_weighted", "no_overlap", "no_boundary",
         "no_attract", "no_repulsive", "random_select", "uniform_select"]
LABEL = {
    "full":              r"\textbf{full} (control)",
    "severity_weighted": r"severity soft-mask",
    "no_overlap":        r"$-$ overlap signal",
    "no_boundary":       r"$-$ boundary signal",
    "no_attract":        r"$-$ attractive force",
    "no_repulsive":      r"$-$ repulsive force",
    "random_select":     r"random selector",
    "uniform_select":    r"uniform (all macros)",
}


def main():
    rows = json.load(open(IN))
    agg = defaultdict(lambda: {"dv": [], "dh": [], "t": []})
    n_pairs = 0
    for r in rows:
        if r.get("baseline_v") is None:
            continue
        n_pairs += 1
        bv = r["baseline_v"]; bh = r["baseline_h"]
        for v, res in r["variants"].items():
            if "error" in res:
                continue
            dv = (res["v"] - bv) / max(bv, 1) * 100
            dh = (res["h"] - bh) / max(abs(bh), 1e-9) * 100
            agg[v]["dv"].append(dv)
            agg[v]["dh"].append(dh)
            agg[v]["t"].append(res["time"])

    csv_rows = []
    for v in ORDER:
        a = agg[v]
        if not a["dv"]:
            continue
        strict = sum(1 for i in range(len(a["dv"])) if a["dv"][i] < 0 and a["dh"][i] < 0)
        csv_rows.append({
            "variant": v,
            "n": len(a["dv"]),
            "dv_mean": S.mean(a["dv"]),
            "dv_median": S.median(a["dv"]),
            "dv_std": S.pstdev(a["dv"]) if len(a["dv"]) > 1 else 0,
            "dh_mean": S.mean(a["dh"]),
            "dh_median": S.median(a["dh"]),
            "dh_std": S.pstdev(a["dh"]) if len(a["dh"]) > 1 else 0,
            "strict_pareto": f"{strict}/{len(a['dv'])}",
            "time_mean": S.mean(a["t"]),
        })

    # CSV
    with open(OUT_DIR / "component_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    # Markdown
    with open(OUT_DIR / "component_ablation.md", "w") as f:
        f.write("| variant | n | Δv mean ± std | Δh mean ± std | strict-Pareto | t (s) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in csv_rows:
            f.write(f"| {r['variant']} | {r['n']} | "
                    f"{r['dv_mean']:+.1f} ± {r['dv_std']:.1f} | "
                    f"{r['dh_mean']:+.1f} ± {r['dh_std']:.1f} | "
                    f"{r['strict_pareto']} | {r['time_mean']:.2f} |\n")

    # LaTeX table for the paper
    with open(PAPER_FIG / "table_component_ablation.tex", "w") as f:
        f.write(r"\begin{tabular}{lrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"variant & $\Delta v\,\%$ & $\Delta h\,\%$ & strict-Pareto & time (s) \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in csv_rows:
            label = LABEL.get(r["variant"], r["variant"])
            f.write(f"{label} & "
                    f"${r['dv_mean']:+.1f}{{\\scriptscriptstyle\\pm{r['dv_std']:.1f}}}$ & "
                    f"${r['dh_mean']:+.1f}{{\\scriptscriptstyle\\pm{r['dh_std']:.1f}}}$ & "
                    f"{r['strict_pareto']} & "
                    f"{r['time_mean']:.2f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")

    # Print headline
    print(f"\n=== Component ablation summary (n={n_pairs} pairs × 8 variants) ===")
    for r in csv_rows:
        print(f"  {r['variant']:<20} Δv={r['dv_mean']:+.1f}±{r['dv_std']:.1f}%  "
              f"Δh={r['dh_mean']:+.1f}±{r['dh_std']:.1f}%  Pareto={r['strict_pareto']}")
    print(f"\nWrote:")
    print(f"  {OUT_DIR / 'component_ablation.csv'}")
    print(f"  {OUT_DIR / 'component_ablation.md'}")
    print(f"  {PAPER_FIG / 'table_component_ablation.tex'}")


if __name__ == "__main__":
    main()
