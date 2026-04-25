"""Aggregate baseline_sweeps.json — for each (circuit, family), pick the
best hyperparameter (lowest v) and report Δv%, Δh% vs baseline.

Then compare against VSR-post (λ=2, seed=42) from full_metrics.json so the
paper can claim that even *tuned* baselines do not beat VSR-post.

Outputs:
  - results/vsr_extra/baseline_sweeps_summary.{csv,md}
  - paper/figures/table_baseline_sweeps.tex
"""
import csv
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SWEEPS = REPO / "results" / "vsr_extra" / "baseline_sweeps.json"
FULL = REPO / "results" / "vsr_extra" / "full_metrics.json"
OUT_DIR = REPO / "results" / "vsr_extra"
PAPER_FIG = REPO / "paper" / "figures"

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]


def best_per_circuit(rows, family, key):
    """Return dict circuit -> row with min `key`."""
    out = {}
    for c in CIRCUITS:
        cands = [r for r in rows if r["family"] == family and r["circuit"] == c
                 and r.get("v") is not None]
        if not cands:
            continue
        out[c] = min(cands, key=lambda r: r[key])
    return out


def vsr_post_seed42(full_rows):
    """For each circuit at seed=42, return the vsr_post (λ=2) row."""
    out = {}
    for r in full_rows:
        if r.get("seed") == 42 and r.get("vsr_post_v") is not None:
            out[r["circuit"]] = r
    return out


def main():
    sweeps = json.load(open(SWEEPS))
    full = json.load(open(FULL))
    post42 = vsr_post_seed42(full)

    # Best hp per (family, circuit) by violations
    cg_best = best_per_circuit(sweeps, "cg", "v")
    re_best = best_per_circuit(sweeps, "repaint", "v")

    md_lines = ["| circuit | base v | cg best (hp,Δv,Δh) | repaint best (hp,Δv,Δh) | VSR-post (Δv,Δh) |",
                "|---|---|---|---|---|"]
    csv_rows = []
    tex_rows = []
    for c in CIRCUITS:
        bv = next((r["baseline_v"] for r in sweeps if r["circuit"] == c), None)
        bh = next((r["baseline_h"] for r in sweeps if r["circuit"] == c), None)
        if bv is None: continue
        cg = cg_best.get(c)
        rp = re_best.get(c)
        cg_dv = (cg["v"] - bv) / max(bv, 1) * 100 if cg else None
        cg_dh = (cg["h"] - bh) / max(abs(bh), 1e-9) * 100 if cg else None
        rp_dv = (rp["v"] - bv) / max(bv, 1) * 100 if rp else None
        rp_dh = (rp["h"] - bh) / max(abs(bh), 1e-9) * 100 if rp else None

        # VSR-post numbers (from same baseline / seed=42)
        p = post42.get(c)
        if p is not None:
            p_dv = (p["vsr_post_v"] - p["baseline_v"]) / max(p["baseline_v"], 1) * 100
            p_dh = (p["vsr_post_h"] - p["baseline_h"]) / max(abs(p["baseline_h"]), 1e-9) * 100
        else:
            p_dv = p_dh = None

        md_lines.append(
            f"| {c} | {bv} | "
            f"{cg['hp']:.1f}, {cg_dv:+.1f}%, {cg_dh:+.1f}% | "
            f"{rp['hp']:.1f}, {rp_dv:+.1f}%, {rp_dh:+.1f}% | "
            f"{p_dv:+.1f}%, {p_dh:+.1f}% |"
        )
        csv_rows.append({
            "circuit": c, "baseline_v": bv, "baseline_h": bh,
            "cg_best_hp": cg['hp'], "cg_best_dv": cg_dv, "cg_best_dh": cg_dh,
            "repaint_best_hp": rp['hp'], "repaint_best_dv": rp_dv,
            "repaint_best_dh": rp_dh,
            "vsr_post_dv": p_dv, "vsr_post_dh": p_dh,
        })
        tex_rows.append((c, bv, cg['hp'], cg_dv, cg_dh,
                        rp['hp'], rp_dv, rp_dh, p_dv, p_dh))

    # ----- Markdown -----
    with open(OUT_DIR / "baseline_sweeps_summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
        f.write("\n## Cross-circuit medians\n\n")
        med_cg_dv = S.median([r["cg_best_dv"] for r in csv_rows])
        med_cg_dh = S.median([r["cg_best_dh"] for r in csv_rows])
        med_rp_dv = S.median([r["repaint_best_dv"] for r in csv_rows])
        med_rp_dh = S.median([r["repaint_best_dh"] for r in csv_rows])
        med_p_dv  = S.median([r["vsr_post_dv"] for r in csv_rows])
        med_p_dh  = S.median([r["vsr_post_dh"] for r in csv_rows])
        f.write(f"- cg-tuned median: Δv={med_cg_dv:+.1f}%  Δh={med_cg_dh:+.1f}%\n")
        f.write(f"- RePaint-tuned median: Δv={med_rp_dv:+.1f}%  Δh={med_rp_dh:+.1f}%\n")
        f.write(f"- VSR-post (λ=2) median: Δv={med_p_dv:+.1f}%  Δh={med_p_dh:+.1f}%\n")
    print(f"Wrote {OUT_DIR/'baseline_sweeps_summary.md'}")

    # ----- CSV -----
    with open(OUT_DIR / "baseline_sweeps_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows: w.writerow(r)
    print(f"Wrote {OUT_DIR/'baseline_sweeps_summary.csv'}")

    # ----- LaTeX -----
    with open(PAPER_FIG / "table_baseline_sweeps.tex", "w") as f:
        f.write(r"\begin{tabular}{lr|rrr|rrr|rr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& base & \multicolumn{3}{c|}{cg-tuned} "
                r"& \multicolumn{3}{c|}{RePaint-tuned} "
                r"& \multicolumn{2}{c}{VSR-post ($\lambda{=}2$)} \\" + "\n")
        f.write(r"Circuit & $v$ & $w^\star$ & $\Delta v$\% & $\Delta h$\% "
                r"& $t^\star$ & $\Delta v$\% & $\Delta h$\% "
                r"& $\Delta v$\% & $\Delta h$\% \\" + "\n")
        f.write(r"\midrule" + "\n")
        for (c, bv, cgw, cgdv, cgdh, rpt, rpdv, rpdh, pdv, pdh) in tex_rows:
            f.write(f"{c} & {bv} & "
                    f"${cgw:.1f}$ & ${cgdv:+.1f}$ & ${cgdh:+.1f}$ & "
                    f"${rpt:.1f}$ & ${rpdv:+.1f}$ & ${rpdh:+.1f}$ & "
                    f"${pdv:+.1f}$ & ${pdh:+.1f}$ \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(f"\\textbf{{median}} & -- & -- & "
                f"${med_cg_dv:+.1f}$ & ${med_cg_dh:+.1f}$ & -- & "
                f"${med_rp_dv:+.1f}$ & ${med_rp_dh:+.1f}$ & "
                f"${med_p_dv:+.1f}$ & ${med_p_dh:+.1f}$ \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {PAPER_FIG/'table_baseline_sweeps.tex'}")

    # ----- Headlines -----
    print("\n=== Headlines ===")
    print(f"  cg-tuned median:        Δv={med_cg_dv:+.1f}%  Δh={med_cg_dh:+.1f}%")
    print(f"  RePaint-tuned median:   Δv={med_rp_dv:+.1f}%  Δh={med_rp_dh:+.1f}%")
    print(f"  VSR-post (λ=2) median:  Δv={med_p_dv:+.1f}%  Δh={med_p_dh:+.1f}%")
    print()
    print("Per-circuit verdict (does VSR-post Δv beat best-tuned baseline Δv?):")
    for r in csv_rows:
        better_cg = r["vsr_post_dv"] < r["cg_best_dv"]
        better_rp = r["vsr_post_dv"] < r["repaint_best_dv"]
        print(f"  {r['circuit']}: cg={'Y' if better_cg else 'N'}  repaint={'Y' if better_rp else 'N'}")


if __name__ == "__main__":
    main()
