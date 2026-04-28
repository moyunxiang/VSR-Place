"""S6: concise core-story summary table.

Rows: raw, VSR(λ=8), FD-pure, raw→DP, FD-pure→DP, VSR(λ=8)→DP
Cols: median residual v_post, full legality (count v=0)
"""
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DP = REPO / "results" / "vsr_extra" / "dreamplace_pipeline.json"
R2 = REPO / "results" / "vsr_extra" / "round2_review.json"
OUT = REPO / "paper" / "figures" / "table_core_summary.tex"

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]


def median_per_circuit(rows, key, circuits=CIRCUITS):
    cm = []
    for c in circuits:
        rs = [r for r in rows if r["circuit"] == c and r.get(key) is not None]
        if rs:
            cm.append(S.mean(r[key] for r in rs))
    return S.median(cm) if cm else None


def main():
    dp = [r for r in json.load(open(DP)) if r.get("baseline_v") is not None]
    r2 = [r for r in json.load(open(R2)) if r.get("baseline_v") is not None]

    rows_data = [
        # (label, source, key, n_total)
        ("raw (no repair)",      r2, "baseline_v",      24),
        ("FD-pure",              r2, "fd_pure_v",       24),
        ("FD+spring",            r2, "fd_spring_v",     24),
        ("VSR-post ($\\lambda{=}8$)", r2, "vsr8_v",      24),
        ("raw $\\to$ cd-sched",  r2, "cd_raw_v",        24),
        ("VSR $\\to$ cd-sched",  r2, "pipe_vsr8_cd_v",  24),
        ("raw $\\to$ DREAMPlace",         dp, "raw_dp_v",        24),
        ("FD-pure $\\to$ DREAMPlace",     dp, "fdpure_dp_v",     24),
        ("FD+spring $\\to$ DREAMPlace",   dp, "fdspring_dp_v",   24),
        ("VSR $\\to$ DREAMPlace",         dp, "vsr8_dp_v",       24),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(r"\begin{tabular}{l|rc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Treatment & median $v_\text{post}$ & full legality $v{=}0$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for label, source, key, ntot in rows_data:
            v = median_per_circuit(source, key)
            n_legal = sum(1 for r in source if r.get(key) == 0)
            n_with_data = sum(1 for r in source if r.get(key) is not None)
            f.write(f"{label} & ${v:.0f}$ & ${n_legal}/{n_with_data}$ \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {OUT}")
    for label, source, key, ntot in rows_data:
        v = median_per_circuit(source, key)
        n_legal = sum(1 for r in source if r.get(key) == 0)
        n_with_data = sum(1 for r in source if r.get(key) is not None)
        print(f"  {label:35s}  median={v:>8.0f}  legal={n_legal}/{n_with_data}")


if __name__ == "__main__":
    main()
