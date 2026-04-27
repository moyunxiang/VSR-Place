"""Q4: build LaTeX table from iterative_vsr_legalizer.json showing
per-stage residual + finding that iteration oscillates (no plateau).
"""
import json
import statistics as S
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "results" / "vsr_extra" / "iterative_vsr_legalizer.json"
OUT_TEX = REPO / "paper" / "figures" / "table_iterative.tex"

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]
STAGES = [
    ("baseline", "baseline"),
    ("raw $\\to$ cd-sched", "raw_cd"),
    ("after VSR$_1$", "after_vsr_1"),
    ("after cd$_1$", "after_cd_1"),
    ("after VSR$_2$", "after_vsr_2"),
    ("after cd$_2$", "after_cd_2"),
    ("after VSR$_3$", "after_vsr_3"),
    ("after cd$_3$", "after_cd_3"),
]


def main():
    data = [r for r in json.load(open(IN)) if r.get("baseline_v") is not None]
    print(f"trials: {len(data)}")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write(r"\begin{tabular}{l|rrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Stage & median $v_\text{post}$ & median $\Delta v$\% & median overlap area \\" + "\n")
        f.write(r"\midrule" + "\n")
        bv_med = None
        for label, key in STAGES:
            cm_v, cm_oa = [], []
            for c in CIRCUITS:
                rs = [r for r in data if r["circuit"] == c and r.get(key)]
                if not rs: continue
                cm_v.append(S.mean(r[key]["v"] for r in rs))
                cm_oa.append(S.mean(r[key]["oa"] for r in rs))
            if not cm_v: continue
            v = S.median(cm_v); oa = S.median(cm_oa)
            if bv_med is None: bv_med = v
            dv = (v - bv_med) / max(bv_med, 1) * 100
            f.write(f"{label} & ${v:.0f}$ & ${dv:+.1f}$ & ${oa:.0f}$ \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
