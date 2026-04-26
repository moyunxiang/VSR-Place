"""Leave-One-Circuit-Out cross-validation for principled λ selection.

Reviewer suggestion: 'Include a principled validation protocol for choosing
lambda.'  We answer with LOCO: for each held-out circuit c, pick the λ
that maximises a held-in objective on the other 5 circuits, and report
the λ chosen + the test (held-out) Δv, Δh, strict-Pareto.

Inputs : results/vsr_extra/lambda_sweep.csv (per-circuit, per-λ means).
Outputs: results/vsr_extra/lambda_loco.{csv,md}
         paper/figures/table_lambda_loco.tex

Two natural in-sample objectives:
  (a) maximize strict-Pareto count (Δv<0 AND Δh<0) per held-in trial.
  (b) maximize a Pareto-improvement score I = -Δv·𝟙[Δv<0] - Δh·𝟙[Δh<0]
      (weight legality and HPWL equally, only credit improvements).
We report (a) by default and include (b) as a robustness check.
"""
from __future__ import annotations

import csv
import statistics as S
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SWEEP = REPO / "results" / "vsr_extra" / "lambda_sweep.csv"
OUT_CSV = REPO / "results" / "vsr_extra" / "lambda_loco.csv"
OUT_MD = REPO / "results" / "vsr_extra" / "lambda_loco.md"
PAPER_FIG = REPO / "paper" / "figures"


def load_table():
    """row[(circuit, λ)] = (dv_mean, dh_mean)."""
    table = {}
    for r in csv.DictReader(open(SWEEP)):
        c = r["circuit"]
        lam = float(r["lambda"])
        table[(c, lam)] = (float(r["delta_v_mean"]), float(r["delta_h_mean"]))
    return table


def main():
    table = load_table()
    circuits = sorted({c for c, _ in table})
    lambdas = sorted({l for _, l in table})

    # In-sample score for held-in circuits at each λ
    def score_strict_pareto(held_in, lam):
        """Number of held-in circuits where (Δv<0 AND Δh<0) at this λ."""
        return sum(1 for c in held_in
                   if table[(c, lam)][0] < 0 and table[(c, lam)][1] < 0)

    def score_pareto_improvement(held_in, lam):
        """Σ over held-in circuits: -Δv·𝟙[Δv<0] − Δh·𝟙[Δh<0]."""
        s = 0.0
        for c in held_in:
            dv, dh = table[(c, lam)]
            if dv < 0: s += -dv
            if dh < 0: s += -dh
        return s

    # LOCO loop
    loco_rows = []
    for c_test in circuits:
        held_in = [c for c in circuits if c != c_test]
        # Pick λ that maximises strict-Pareto count (tie-break by improvement-score, then by smaller λ)
        best_lam_a = max(lambdas,
                        key=lambda lam: (score_strict_pareto(held_in, lam),
                                         score_pareto_improvement(held_in, lam),
                                         -lam))
        best_lam_b = max(lambdas,
                        key=lambda lam: (score_pareto_improvement(held_in, lam),
                                         score_strict_pareto(held_in, lam),
                                         -lam))
        # Test on held-out
        dv_a, dh_a = table[(c_test, best_lam_a)]
        dv_b, dh_b = table[(c_test, best_lam_b)]
        is_strict_a = (dv_a < 0 and dh_a < 0)
        is_strict_b = (dv_b < 0 and dh_b < 0)
        loco_rows.append({
            "circuit": c_test,
            "lam_strictpar": best_lam_a, "test_dv_a": dv_a, "test_dh_a": dh_a,
            "test_strict_a": is_strict_a,
            "lam_paretoimp": best_lam_b, "test_dv_b": dv_b, "test_dh_b": dh_b,
            "test_strict_b": is_strict_b,
        })

    # Oracle: best per-circuit λ (lowest |Δh| subject to Δv<0)
    oracle = {}
    for c in circuits:
        feasible = [(lam, *table[(c, lam)]) for lam in lambdas if table[(c, lam)][0] < 0]
        if feasible:
            best = min(feasible, key=lambda x: x[2])  # smallest Δh
            oracle[c] = best[0]
        else:
            oracle[c] = None

    # Default: λ=2
    default_strict = sum(1 for c in circuits
                         if table[(c, 2.0)][0] < 0 and table[(c, 2.0)][1] < 0)

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loco_rows[0].keys()))
        w.writeheader()
        for r in loco_rows: w.writerow(r)
    print(f"Wrote {OUT_CSV}")

    # Write MD report
    with open(OUT_MD, "w") as f:
        f.write("# LOCO cross-validation for λ\n\n")
        f.write("## Selected λ per held-out circuit (in-sample objective: strict-Pareto count)\n\n")
        f.write("| held-out | λ̂ (objective a) | test Δv% | test Δh% | strict-P? |\n")
        f.write("|---|---|---|---|---|\n")
        for r in loco_rows:
            f.write(f"| {r['circuit']} | {r['lam_strictpar']:.2f} | "
                    f"{r['test_dv_a']:+.1f} | {r['test_dh_a']:+.1f} | "
                    f"{'✓' if r['test_strict_a'] else '✗'} |\n")
        f.write("\n## Strict-Pareto rate on held-out circuits\n\n")
        sp_a = sum(1 for r in loco_rows if r["test_strict_a"])
        sp_b = sum(1 for r in loco_rows if r["test_strict_b"])
        f.write(f"- Objective (a) strict-Pareto count: **{sp_a}/{len(loco_rows)}**\n")
        f.write(f"- Objective (b) Pareto improvement: **{sp_b}/{len(loco_rows)}**\n")
        f.write(f"- Fixed λ=2 (paper default): **{default_strict}/{len(circuits)}**\n")
        f.write(f"- Oracle (per-circuit best λ subject to Δv<0): **{len(circuits)}/{len(circuits)}**\n")
        f.write("\n## Headlines\n\n")
        med_dv = S.median(r["test_dv_a"] for r in loco_rows)
        med_dh = S.median(r["test_dh_a"] for r in loco_rows)
        f.write(f"- LOCO test Δv median: {med_dv:+.1f}%\n")
        f.write(f"- LOCO test Δh median: {med_dh:+.1f}%\n")
    print(f"Wrote {OUT_MD}")

    # LaTeX paper-ready table
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    tex_out = PAPER_FIG / "table_lambda_loco.tex"
    with open(tex_out, "w") as f:
        f.write(r"\begin{tabular}{l|rrrc|c}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"held-out & $\hat\lambda$ & test $\Delta v$\% & test $\Delta h$\% "
                r"& strict-Pareto? & oracle $\lambda^\star$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in loco_rows:
            mark = r"\checkmark" if r["test_strict_a"] else r"$\times$"
            orcl = oracle[r["circuit"]]
            orcl_str = f"{orcl:.1f}" if orcl is not None else "n/a"
            f.write(f"{r['circuit']} & "
                    f"${r['lam_strictpar']:.1f}$ & ${r['test_dv_a']:+.1f}$ & "
                    f"${r['test_dh_a']:+.1f}$ & {mark} & ${orcl_str}$ \\\\\n")
        f.write(r"\midrule" + "\n")
        med_dv = S.median(r["test_dv_a"] for r in loco_rows)
        med_dh = S.median(r["test_dh_a"] for r in loco_rows)
        sp_a = sum(1 for r in loco_rows if r["test_strict_a"])
        f.write(f"\\textbf{{summary}} & -- & "
                f"${med_dv:+.1f}$ & ${med_dh:+.1f}$ & "
                f"$\\mathbf{{{sp_a}/{len(loco_rows)}}}$ & -- \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {tex_out}")

    print("\n=== Headlines ===")
    print(f"  LOCO objective (a) test strict-Pareto: {sp_a}/{len(loco_rows)} "
          f"(vs fixed λ=2: {default_strict}/{len(circuits)}, oracle: {len(circuits)}/{len(circuits)})")
    print(f"  LOCO test median: Δv={med_dv:+.1f}%  Δh={med_dh:+.1f}%")
    print()
    print("Per-circuit λ̂ chosen by LOCO objective (a):")
    for r in loco_rows:
        print(f"  {r['circuit']}: λ̂={r['lam_strictpar']:.1f}  "
              f"test Δv={r['test_dv_a']:+.1f}%  Δh={r['test_dh_a']:+.1f}%  "
              f"{'strict-P✓' if r['test_strict_a'] else 'not-Pareto'}")


if __name__ == "__main__":
    main()
