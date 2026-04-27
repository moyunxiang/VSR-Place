"""Q3/S2: error analysis of residual violations.

Builds a breakdown table for raw / VSR-post(λ=2) / VSR-post(λ=8) showing:
  - violation count
  - total overlap area
  - max single-pair overlap
  - n violator macros
  - "spread": overlap area / max overlap (small = concentrated, large = scattered)

Inputs: results/vsr_extra/full_metrics.json   (λ=2, all 4 metrics)
        results/vsr_extra/round2_review.json   (λ=8, all 4 metrics)

Output: paper/figures/table_violation_breakdown.tex
"""
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FM = REPO / "results" / "vsr_extra" / "full_metrics.json"
R2 = REPO / "results" / "vsr_extra" / "round2_review.json"
OUT_TEX = REPO / "paper" / "figures" / "table_violation_breakdown.tex"

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]


def main():
    fm = [r for r in json.load(open(FM)) if r.get("baseline_v") is not None]
    r2 = [r for r in json.load(open(R2)) if r.get("vsr8_v") is not None]
    fm_by = {(r["circuit"], r["seed"]): r for r in fm}
    r2_by = {(r["circuit"], r["seed"]): r for r in r2}

    # For each method, compute cross-circuit medians of (count, oa, max, count_per_pair=count/max)
    methods = [
        ("baseline", fm_by, ("baseline_v", "baseline_overlap_area", "baseline_max_overlap")),
        ("VSR-post lambda=2", fm_by, ("vsr_post_v", "vsr_post_overlap_area", "vsr_post_max_overlap")),
        ("VSR-post lambda=8", r2_by, ("vsr8_v", "vsr8_oa", "vsr8_max")),
    ]

    rows_out = []
    for label, source, (kv, koa, kmx) in methods:
        cm_v, cm_oa, cm_mx = [], [], []
        for c in CIRCUITS:
            rs = [r for k, r in source.items() if k[0] == c
                  and r.get(kv) is not None and r.get(koa) is not None]
            if not rs: continue
            cm_v.append(S.mean(r[kv] for r in rs))
            cm_oa.append(S.mean(r[koa] for r in rs))
            cm_mx.append(S.mean(r[kmx] for r in rs))
        if cm_v:
            rows_out.append({
                "label": label,
                "med_v": S.median(cm_v),
                "med_oa": S.median(cm_oa),
                "med_mx": S.median(cm_mx),
                "med_oa_per_v": S.median([oa / max(v, 1) for v, oa in zip(cm_v, cm_oa)]),
            })

    # LaTeX table
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write(r"\begin{tabular}{l|rrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & violation count & overlap area & max single-pair & area / count \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows_out:
            f.write(f"{r['label']} & "
                    f"${r['med_v']:.0f}$ & "
                    f"${r['med_oa']:.1f}$ & "
                    f"${r['med_mx']:.2f}$ & "
                    f"${r['med_oa_per_v']:.3f}$ \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {OUT_TEX}")

    print("\n=== Violation breakdown (cross-circuit medians) ===")
    print(f"{'method':25s} {'count':>8s} {'oa':>10s} {'max':>8s} {'oa/count':>10s}")
    for r in rows_out:
        print(f"{r['label']:25s} {r['med_v']:8.0f} {r['med_oa']:10.1f} "
              f"{r['med_mx']:8.2f} {r['med_oa_per_v']:10.3f}")


if __name__ == "__main__":
    main()
