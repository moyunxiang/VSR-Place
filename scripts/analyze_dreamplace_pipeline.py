"""Aggregate dreamplace_pipeline.json into paper table.

Per-trial: baseline_v / baseline_h / raw_dp_v / raw_dp_h / vsr8_dp_v / vsr8_dp_h.
Computes per-circuit residual + Δv pct + circuit-level paired Wilcoxon (n=6).
"""
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "results" / "vsr_extra" / "dreamplace_pipeline.json"
PAPER_FIG = REPO / "paper" / "figures"
SUMMARY = REPO / "results" / "vsr_extra" / "dreamplace_pipeline_summary.md"

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]


def pct(new, base):
    return (new - base) / max(abs(base), 1e-9) * 100


def main():
    rows = json.load(open(IN))
    rows = [r for r in rows if r.get("baseline_v") is not None
            and r.get("raw_dp_v") is not None and r.get("vsr8_dp_v") is not None]
    print(f"Loaded {len(rows)} valid trials")
    have_fd = all(r.get("fdpure_dp_v") is not None and r.get("fdspring_dp_v") is not None for r in rows)
    print(f"Have FD baselines: {have_fd}")

    g = defaultdict(list)
    for r in rows: g[r["circuit"]].append(r)

    rows_out = []
    cm_raw_v, cm_pipe_v = [], []
    cm_fdp_v, cm_fds_v = [], []
    full_legal_raw = full_legal_pipe = full_legal_fdp = full_legal_fds = 0
    n_total = 0
    for c in CIRCUITS:
        rs = g.get(c, [])
        if not rs: continue
        n_total += len(rs)
        full_legal_raw += sum(1 for r in rs if r["raw_dp_v"] == 0)
        full_legal_pipe += sum(1 for r in rs if r["vsr8_dp_v"] == 0)
        bv = S.mean(r["baseline_v"] for r in rs)
        rv = S.mean(r["raw_dp_v"] for r in rs)
        pv = S.mean(r["vsr8_dp_v"] for r in rs)
        cm_raw_v.append(rv); cm_pipe_v.append(pv)
        row = {
            "c": c, "n": len(rs),
            "raw_v": rv, "pipe_v": pv,
            "raw_dv": pct(rv, bv), "pipe_dv": pct(pv, bv),
            "raw_legal": sum(1 for r in rs if r["raw_dp_v"] == 0),
            "pipe_legal": sum(1 for r in rs if r["vsr8_dp_v"] == 0),
            "raw_t": S.mean(r.get("raw_dp_time", 0) for r in rs),
            "pipe_t": S.mean(r.get("vsr8_dp_time", 0) for r in rs),
        }
        if have_fd:
            fdpv = S.mean(r["fdpure_dp_v"] for r in rs)
            fdsv = S.mean(r["fdspring_dp_v"] for r in rs)
            cm_fdp_v.append(fdpv); cm_fds_v.append(fdsv)
            full_legal_fdp += sum(1 for r in rs if r["fdpure_dp_v"] == 0)
            full_legal_fds += sum(1 for r in rs if r["fdspring_dp_v"] == 0)
            row.update({
                "fdp_v": fdpv, "fds_v": fdsv,
                "fdp_dv": pct(fdpv, bv), "fds_dv": pct(fdsv, bv),
                "fdp_legal": sum(1 for r in rs if r["fdpure_dp_v"] == 0),
                "fds_legal": sum(1 for r in rs if r["fdspring_dp_v"] == 0),
            })
        rows_out.append(row)

    # Circuit-level paired Wilcoxon (n=6)
    w_v = stats.wilcoxon(cm_raw_v, cm_pipe_v, zero_method="pratt")
    p_v = float(w_v.pvalue)
    if have_fd:
        w_fdp = stats.wilcoxon(cm_raw_v, cm_fdp_v, zero_method="pratt")
        w_fds = stats.wilcoxon(cm_raw_v, cm_fds_v, zero_method="pratt")
        w_vsr_fdp = stats.wilcoxon(cm_pipe_v, cm_fdp_v, zero_method="pratt")
        w_vsr_fds = stats.wilcoxon(cm_pipe_v, cm_fds_v, zero_method="pratt")
        p_fdp = float(w_fdp.pvalue); p_fds = float(w_fds.pvalue)
        p_vsr_fdp = float(w_vsr_fdp.pvalue); p_vsr_fds = float(w_vsr_fds.pvalue)
    else:
        p_fdp = p_fds = p_vsr_fdp = p_vsr_fds = None

    # LaTeX table
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIG / "table_dreamplace_pipeline.tex"
    with open(out, "w") as f:
        if have_fd:
            f.write(r"\begin{tabular}{lr|rr|rr|rr|rr}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"& & \multicolumn{2}{c|}{raw} & \multicolumn{2}{c|}{FD-pure} "
                    r"& \multicolumn{2}{c|}{FD+spring} & \multicolumn{2}{c}{VSR ($\lambda{=}8$)} \\" + "\n")
            f.write(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}" + "\n")
            f.write(r"Circuit & $n$ & $v_\text{post}$ & $\Delta v$\% & "
                    r"$v_\text{post}$ & $\Delta v$\% & $v_\text{post}$ & $\Delta v$\% & "
                    r"$v_\text{post}$ & $\Delta v$\% \\" + "\n")
            f.write(r"\midrule" + "\n")
            for r in rows_out:
                f.write(f"{r['c']} & {r['n']} & "
                        f"${r['raw_v']:.0f}$ & ${r['raw_dv']:+.1f}$ & "
                        f"${r['fdp_v']:.0f}$ & ${r['fdp_dv']:+.1f}$ & "
                        f"${r['fds_v']:.0f}$ & ${r['fds_dv']:+.1f}$ & "
                        f"${r['pipe_v']:.0f}$ & ${r['pipe_dv']:+.1f}$ \\\\\n")
            f.write(r"\midrule" + "\n")
            med_raw_v = S.median(r["raw_v"] for r in rows_out)
            med_pipe_v = S.median(r["pipe_v"] for r in rows_out)
            med_raw_dv = S.median(r["raw_dv"] for r in rows_out)
            med_pipe_dv = S.median(r["pipe_dv"] for r in rows_out)
            med_fdp_v = S.median(r["fdp_v"] for r in rows_out)
            med_fdp_dv = S.median(r["fdp_dv"] for r in rows_out)
            med_fds_v = S.median(r["fds_v"] for r in rows_out)
            med_fds_dv = S.median(r["fds_dv"] for r in rows_out)
            f.write(f"\\textbf{{median}} & -- & "
                    f"${med_raw_v:.0f}$ & ${med_raw_dv:+.1f}$ & "
                    f"${med_fdp_v:.0f}$ & ${med_fdp_dv:+.1f}$ & "
                    f"${med_fds_v:.0f}$ & ${med_fds_dv:+.1f}$ & "
                    f"${med_pipe_v:.0f}$ & ${med_pipe_dv:+.1f}$ \\\\\n")
            f.write(f"\\multicolumn{{10}}{{l}}{{Circuit-level paired Wilcoxon "
                    f"($n{{=}}6$): raw vs.\\ VSR $p_v={p_v:.3f}$; "
                    f"raw vs.\\ FD-pure $p_v={p_fdp:.3f}$; "
                    f"raw vs.\\ FD+spring $p_v={p_fds:.3f}$; "
                    f"VSR vs.\\ FD-pure $p_v={p_vsr_fdp:.3f}$.}} \\\\\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
        else:
            f.write(r"\begin{tabular}{lr|rrc|rrc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"& & \multicolumn{3}{c|}{raw $\to$ DREAMPlace} & "
                    r"\multicolumn{3}{c}{raw $\to$ VSR($\lambda{=}8$) $\to$ DREAMPlace} \\" + "\n")
            f.write(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}" + "\n")
            f.write(r"Circuit & $n$ & $v_\text{post}$ & $\Delta v$\% & legal & "
                    r"$v_\text{post}$ & $\Delta v$\% & legal \\" + "\n")
            f.write(r"\midrule" + "\n")
            for r in rows_out:
                f.write(f"{r['c']} & {r['n']} & "
                        f"${r['raw_v']:.0f}$ & ${r['raw_dv']:+.1f}$ & "
                        f"{r['raw_legal']}/{r['n']} & "
                        f"${r['pipe_v']:.0f}$ & ${r['pipe_dv']:+.1f}$ & "
                        f"{r['pipe_legal']}/{r['n']} \\\\\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            med_raw_v = S.median(r["raw_v"] for r in rows_out)
            med_pipe_v = S.median(r["pipe_v"] for r in rows_out)
            med_raw_dv = S.median(r["raw_dv"] for r in rows_out)
            med_pipe_dv = S.median(r["pipe_dv"] for r in rows_out)
    print(f"Wrote {out}")

    # Per-circuit win count
    wins_vsr_vs_raw = sum(1 for r in rows_out if r["pipe_v"] < r["raw_v"])
    print(f"\nPer-circuit win count (vsr8+DP < raw+DP on residual_v): {wins_vsr_vs_raw}/{len(rows_out)}")

    # Per-trial win count
    trial_wins_vsr_vs_raw = sum(1 for r in rows if r["vsr8_dp_v"] < r["raw_dp_v"])
    print(f"Per-trial win count (vsr vs raw): {trial_wins_vsr_vs_raw}/{len(rows)}")

    if have_fd:
        wins_fdp_vs_raw = sum(1 for r in rows_out if r["fdp_v"] < r["raw_v"])
        wins_vsr_vs_fdp = sum(1 for r in rows_out if r["pipe_v"] < r["fdp_v"])
        trial_wins_fdp_vs_raw = sum(1 for r in rows if r["fdpure_dp_v"] < r["raw_dp_v"])
        trial_wins_vsr_vs_fdp = sum(1 for r in rows if r["vsr8_dp_v"] < r["fdpure_dp_v"])
        print(f"Per-circuit FD-pure < raw: {wins_fdp_vs_raw}/{len(rows_out)}")
        print(f"Per-circuit VSR < FD-pure: {wins_vsr_vs_fdp}/{len(rows_out)}")
        print(f"Per-trial    FD-pure < raw: {trial_wins_fdp_vs_raw}/{len(rows)}")
        print(f"Per-trial    VSR < FD-pure: {trial_wins_vsr_vs_fdp}/{len(rows)}")

    print()
    print("=== Headlines (DREAMPlace pipeline, %d trials) ===" % len(rows))
    print(f"  raw         → DP median:  Δv={med_raw_dv:+.1f}%  v_post={med_raw_v:.0f}")
    if have_fd:
        print(f"  FD-pure     → DP median:  Δv={med_fdp_dv:+.1f}%  v_post={med_fdp_v:.0f}")
        print(f"  FD+spring   → DP median:  Δv={med_fds_dv:+.1f}%  v_post={med_fds_v:.0f}")
    print(f"  VSR(λ=8)    → DP median:  Δv={med_pipe_dv:+.1f}%  v_post={med_pipe_v:.0f}")
    print(f"  full-legality (v=0): raw {full_legal_raw}/{n_total}  vsr {full_legal_pipe}/{n_total}", end="")
    if have_fd:
        print(f"  fdp {full_legal_fdp}/{n_total}  fds {full_legal_fds}/{n_total}")
    else:
        print()
    print(f"  Circuit Wilcoxon n=6:  raw vs vsr p={p_v:.3f}", end="")
    if have_fd:
        print(f";  raw vs fdp p={p_fdp:.3f};  vsr vs fdp p={p_vsr_fdp:.3f}")
    else:
        print()

    with open(SUMMARY, "w") as f:
        f.write("# DREAMPlace pipeline summary\n\n")
        f.write(f"Trials: {len(rows)}\n\n")
        for r in rows_out:
            f.write(f"- {r['c']} (n={r['n']}): raw+DP v={r['raw_v']:.0f} (Δv={r['raw_dv']:+.1f}%), "
                    f"vsr+DP v={r['pipe_v']:.0f} (Δv={r['pipe_dv']:+.1f}%)\n")
        f.write(f"\n## Cross-circuit medians\n\n")
        f.write(f"- raw+DP: Δv={med_raw_dv:+.1f}% v_post={med_raw_v:.0f}\n")
        f.write(f"- vsr+DP: Δv={med_pipe_dv:+.1f}% v_post={med_pipe_v:.0f}\n")
        f.write(f"- Per-trial wins (vsr<raw on v): {trial_wins_vsr_vs_raw}/{len(rows)}\n")
        f.write(f"- Circuit-level Wilcoxon n=6: raw vs vsr p={p_v:.3f}\n")
        if have_fd:
            f.write(f"- Per-trial wins (fdpure<raw): {trial_wins_fdp_vs_raw}/{len(rows)}\n")
            f.write(f"- Per-trial wins (vsr<fdpure): {trial_wins_vsr_vs_fdp}/{len(rows)}\n")
            f.write(f"- Wilcoxon raw vs fdpure: p={p_fdp:.3f}\n")
            f.write(f"- Wilcoxon vsr vs fdpure: p={p_vsr_fdp:.3f}\n")


if __name__ == "__main__":
    main()
