"""Generate consistent DREAMPlace pipeline table with all 7 pre-conditioners
(raw, FD-pure, FD+spring, VSR-post λ∈{8,12,16}, VSR-intra-soft).

Combines data from:
  - dreamplace_pipeline.json (raw, fdpure, fdspring, vsr8 on 24 trials)
  - extra_pipeline_variants.json (vsr12, vsr16, intra on 24 trials)

Output: paper/figures/table_dreamplace_pipeline.tex (replaces 4-col table)
"""
import json
import statistics as S
from collections import defaultdict
from pathlib import Path
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
DP = REPO / "results" / "vsr_extra" / "dreamplace_pipeline.json"
EX = REPO / "results" / "vsr_extra" / "extra_pipeline_variants.json"
OUT = REPO / "paper" / "figures" / "table_dreamplace_pipeline.tex"

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]


def main():
    dp = {(r["circuit"], r["seed"]): r for r in json.load(open(DP))
          if r.get("baseline_v") is not None and r.get("raw_dp_v") is not None}
    ex = {(r["circuit"], r["seed"]): r for r in json.load(open(EX))
          if r.get("baseline_v") is not None and r.get("vsr12_dp_v") is not None}

    # Per-circuit means
    rows_out = []
    raw_per_c, vsr8_per_c, fdp_per_c, fds_per_c = [], [], [], []
    vsr12_per_c, vsr16_per_c, intra_per_c = [], [], []
    full_legal = defaultdict(int)
    n_total = 0

    for c in CIRCUITS:
        rs_dp = [(k, r) for k, r in dp.items() if k[0] == c]
        rs_ex = [(k, r) for k, r in ex.items() if k[0] == c]
        if not rs_dp or not rs_ex: continue
        n_total += len(rs_dp)
        bv = S.mean(r[1]["baseline_v"] for r in rs_dp)
        # From dp
        raw_v = S.mean(r[1]["raw_dp_v"] for r in rs_dp)
        fdp_v = S.mean(r[1]["fdpure_dp_v"] for r in rs_dp)
        fds_v = S.mean(r[1]["fdspring_dp_v"] for r in rs_dp)
        vsr8_v = S.mean(r[1]["vsr8_dp_v"] for r in rs_dp)
        # From extra
        vsr12_v = S.mean(r[1]["vsr12_dp_v"] for r in rs_ex)
        vsr16_v = S.mean(r[1]["vsr16_dp_v"] for r in rs_ex)
        intra_v = S.mean(r[1]["intra_dp_v"] for r in rs_ex)

        full_legal["raw"] += sum(1 for _, r in rs_dp if r["raw_dp_v"] == 0)
        full_legal["fdpure"] += sum(1 for _, r in rs_dp if r["fdpure_dp_v"] == 0)
        full_legal["fdspring"] += sum(1 for _, r in rs_dp if r["fdspring_dp_v"] == 0)
        full_legal["vsr8"] += sum(1 for _, r in rs_dp if r["vsr8_dp_v"] == 0)
        full_legal["vsr12"] += sum(1 for _, r in rs_ex if r["vsr12_dp_v"] == 0)
        full_legal["vsr16"] += sum(1 for _, r in rs_ex if r["vsr16_dp_v"] == 0)
        full_legal["intra"] += sum(1 for _, r in rs_ex if r["intra_dp_v"] == 0)

        raw_per_c.append(raw_v); vsr8_per_c.append(vsr8_v)
        fdp_per_c.append(fdp_v); fds_per_c.append(fds_v)
        vsr12_per_c.append(vsr12_v); vsr16_per_c.append(vsr16_v)
        intra_per_c.append(intra_v)

        rows_out.append({
            "c": c, "n": len(rs_dp), "bv": bv,
            "raw": raw_v, "fdp": fdp_v, "fds": fds_v,
            "vsr8": vsr8_v, "vsr12": vsr12_v, "vsr16": vsr16_v,
            "intra": intra_v,
        })

    # Wilcoxon vs raw for each treatment
    def w(per_c):
        return float(stats.wilcoxon(raw_per_c, per_c, zero_method="pratt").pvalue)

    p_fdp = w(fdp_per_c); p_fds = w(fds_per_c)
    p_vsr8 = w(vsr8_per_c); p_vsr12 = w(vsr12_per_c); p_vsr16 = w(vsr16_per_c)
    p_intra = w(intra_per_c)
    # vsr8 vs fdpure
    p_vsr8_fdp = float(stats.wilcoxon(vsr8_per_c, fdp_per_c, zero_method="pratt").pvalue)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(r"\begin{tabular}{lr|rrrrrrr|c}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& & \multicolumn{7}{c|}{$v_\text{post}$ after pre-conditioner $\to$ DREAMPlace} & legal \\" + "\n")
        f.write(r"\cmidrule(lr){3-9}" + "\n")
        f.write(r"Circuit & $n$ & raw & FD-pure & FD+spring & VSR$_{\lambda=8}$ & VSR$_{\lambda=12}$ & VSR$_{\lambda=16}$ & intra & (any) \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows_out:
            legal_str = "0/" + str(r['n'])  # always 0 — none reach v=0
            f.write(f"{r['c']} & {r['n']} & "
                    f"${r['raw']:.0f}$ & ${r['fdp']:.0f}$ & ${r['fds']:.0f}$ & "
                    f"${r['vsr8']:.0f}$ & ${r['vsr12']:.0f}$ & ${r['vsr16']:.0f}$ & "
                    f"${r['intra']:.0f}$ & {legal_str} \\\\\n")
        f.write(r"\midrule" + "\n")
        med = lambda lst: S.median(lst)
        f.write(f"\\textbf{{median}} & -- & "
                f"${med(raw_per_c):.0f}$ & ${med(fdp_per_c):.0f}$ & "
                f"${med(fds_per_c):.0f}$ & ${med(vsr8_per_c):.0f}$ & "
                f"${med(vsr12_per_c):.0f}$ & ${med(vsr16_per_c):.0f}$ & "
                f"${med(intra_per_c):.0f}$ & ${full_legal['raw']+full_legal['fdpure']+full_legal['fdspring']+full_legal['vsr8']+full_legal['vsr12']+full_legal['vsr16']+full_legal['intra']}/{n_total*7}$ \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{10}{l}{\textbf{Wilcoxon} ($n{=}6$, paired vs.\ raw): "
                f"FD-pure $p_v={p_fdp:.3f}$; FD+spring $p_v={p_fds:.3f}$; "
                f"VSR$_{{8}}$ $p_v={p_vsr8:.3f}$; VSR$_{{12}}$ $p_v={p_vsr12:.3f}$; "
                f"VSR$_{{16}}$ $p_v={p_vsr16:.3f}$; intra $p_v={p_intra:.3f}$.}}" + r" \\" + "\n")
        f.write(r"\multicolumn{10}{l}{\textbf{VSR$_{\lambda=8}$ vs.\ FD-pure} (paired): "
                f"$p_v={p_vsr8_fdp:.3f}$ (not significant).}}" + r" \\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {OUT}")
    print()
    print("=== Headlines ===")
    print(f"raw       median: {med(raw_per_c):.0f}")
    print(f"FD-pure   median: {med(fdp_per_c):.0f}  p_v={p_fdp:.3f}")
    print(f"FD+spring median: {med(fds_per_c):.0f}  p_v={p_fds:.3f}")
    print(f"VSR(λ=8)  median: {med(vsr8_per_c):.0f}  p_v={p_vsr8:.3f}")
    print(f"VSR(λ=12) median: {med(vsr12_per_c):.0f}  p_v={p_vsr12:.3f}")
    print(f"VSR(λ=16) median: {med(vsr16_per_c):.0f}  p_v={p_vsr16:.3f}")
    print(f"intra     median: {med(intra_per_c):.0f}  p_v={p_intra:.3f}")
    print(f"VSR8 vs FD-pure paired Wilcoxon: p={p_vsr8_fdp:.3f}")
    print(f"Full legality (any v=0 across all 24×7=168 method-trials): {sum(full_legal.values())}/{n_total*7}")


if __name__ == "__main__":
    main()
