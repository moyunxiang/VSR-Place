"""Regenerate paper tables to address reviewer's three additional asks:

  (1) Present main result as a lambda-family: Table 1 shows VSR-post at
      lambda=2 AND lambda=8.
  (2) Soften "decisive downstream" claim by reporting absolute residual
      violations + full-legality count post-pipeline (not just deltas).
  (3) Circuit-level paired Wilcoxon for the downstream pipeline test.

Inputs:
  - results/ispd2005/main_neurips.json    (lambda=2, 24 trials)
  - results/vsr_extra/round2_review.json  (lambda=8 + pipeline, 24 trials)

Outputs:
  - paper/figures/table_main_neurips.tex  (replaced: lambda-family layout)
  - paper/figures/table_downstream_pipeline.tex (replaced: + residual v
        + full-legality + circuit-level Wilcoxon p-values)
  - results/vsr_extra/round2_pipeline_stats.md
"""
import csv
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / "results" / "ispd2005" / "main_neurips.json"
R2 = REPO / "results" / "vsr_extra" / "round2_review.json"
CDSTD = REPO / "results" / "vsr_extra" / "cdstd_pipeline.json"
PAPER_FIG = REPO / "paper" / "figures"

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]
N_MACROS = {"adaptec1": 543, "adaptec2": 566, "adaptec3": 723,
            "adaptec4": 1329, "bigblue1": 560, "bigblue3": 1298}


def pct(new, base):
    return (new - base) / max(abs(base), 1e-9) * 100


def fmt_pm(mean, std):
    return rf"${mean:+.1f}{{\scriptscriptstyle\pm{std:.1f}}}$"


def main():
    main_rows = json.load(open(MAIN))
    r2_rows = json.load(open(R2))
    r2_by = {(r["circuit"], r["seed"]): r for r in r2_rows}

    # ---------------------------------------------------------------- Table 1
    # Build per-circuit (circuit, lambda) means/stds for VSR-post + others.
    grouped = defaultdict(list)
    for r in main_rows:
        grouped[r["circuit"]].append(r)
    grouped_r2 = defaultdict(list)
    for r in r2_rows:
        grouped_r2[r["circuit"]].append(r)

    rows_out = []
    for c in CIRCUITS:
        ms = grouped[c]
        rs = grouped_r2[c]
        # baseline_v / baseline_h from main set (paired)
        bvs = [r["baseline_v"] for r in ms]
        bhs = [r["baseline_h"] for r in ms]
        # VSR-post lambda=2 (from main_neurips)
        v2_dvs = [pct(r["vsr_post_v"], r["baseline_v"]) for r in ms if r.get("vsr_post_v") is not None]
        v2_dhs = [pct(r["vsr_post_h"], r["baseline_h"]) for r in ms if r.get("vsr_post_h") is not None]
        # VSR-post lambda=8 (from round2_review)
        v8_dvs = [pct(r["vsr8_v"], r["baseline_v"]) for r in rs if r.get("vsr8_v") is not None]
        v8_dhs = [pct(r["vsr8_h"], r["baseline_h"]) for r in rs if r.get("vsr8_h") is not None]
        # VSR-intra (from main)
        vi_dvs = [pct(r["vsr_intra_v"], r["baseline_v"]) for r in ms if r.get("vsr_intra_v") is not None]
        vi_dhs = [pct(r["vsr_intra_h"], r["baseline_h"]) for r in ms if r.get("vsr_intra_h") is not None]
        # CD-std
        cds_dvs = [pct(r["cd_std_v"], r["baseline_v"]) for r in ms if r.get("cd_std_v") is not None]
        cds_dhs = [pct(r["cd_std_h"], r["baseline_h"]) for r in ms if r.get("cd_std_h") is not None]
        # CD-sched
        cdc_dvs = [pct(r["cd_sched_v"], r["baseline_v"]) for r in ms if r.get("cd_sched_v") is not None]
        cdc_dhs = [pct(r["cd_sched_h"], r["baseline_h"]) for r in ms if r.get("cd_sched_h") is not None]
        rows_out.append({
            "c": c, "n": N_MACROS[c],
            "v2_dv": (S.mean(v2_dvs), S.stdev(v2_dvs) if len(v2_dvs) > 1 else 0),
            "v2_dh": (S.mean(v2_dhs), S.stdev(v2_dhs) if len(v2_dhs) > 1 else 0),
            "v8_dv": (S.mean(v8_dvs), S.stdev(v8_dvs) if len(v8_dvs) > 1 else 0),
            "v8_dh": (S.mean(v8_dhs), S.stdev(v8_dhs) if len(v8_dhs) > 1 else 0),
            "vi_dv": (S.mean(vi_dvs), S.stdev(vi_dvs) if len(vi_dvs) > 1 else 0),
            "vi_dh": (S.mean(vi_dhs), S.stdev(vi_dhs) if len(vi_dhs) > 1 else 0),
            "cds_dv": (S.mean(cds_dvs), S.stdev(cds_dvs) if len(cds_dvs) > 1 else 0),
            "cds_dh": (S.mean(cds_dhs), S.stdev(cds_dhs) if len(cds_dhs) > 1 else 0),
            "cdc_dv": (S.mean(cdc_dvs), S.stdev(cdc_dvs) if len(cdc_dvs) > 1 else 0),
            "cdc_dh": (S.mean(cdc_dhs), S.stdev(cdc_dhs) if len(cdc_dhs) > 1 else 0),
        })

    # Write table_main_neurips.tex with VSR-post (lambda=2) and (lambda=8) sub-columns
    out = PAPER_FIG / "table_main_neurips.tex"
    with open(out, "w") as f:
        f.write(r"\begin{tabular}{lrrrrrrrrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r" & & \multicolumn{4}{c}{VSR-post ($\lambda$-family)} & "
                r"\multicolumn{2}{c}{VSR-intra} & \multicolumn{2}{c}{CD-std} & "
                r"\multicolumn{2}{c}{CD-sched} \\" + "\n")
        f.write(r"\cmidrule(lr){3-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12}" + "\n")
        f.write(r" & & \multicolumn{2}{c}{$\lambda=2$} & \multicolumn{2}{c}{$\lambda=8$} & & & & & & \\" + "\n")
        f.write(r"Circuit & $N$ & $\Delta v$ & $\Delta h$ & $\Delta v$ & $\Delta h$ & "
                r"$\Delta v$ & $\Delta h$ & $\Delta v$ & $\Delta h$ & "
                r"$\Delta v$ & $\Delta h$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows_out:
            f.write(f"{r['c']} & {r['n']} & "
                    f"{fmt_pm(*r['v2_dv'])} & {fmt_pm(*r['v2_dh'])} & "
                    f"{fmt_pm(*r['v8_dv'])} & {fmt_pm(*r['v8_dh'])} & "
                    f"{fmt_pm(*r['vi_dv'])} & {fmt_pm(*r['vi_dh'])} & "
                    f"{fmt_pm(*r['cds_dv'])} & {fmt_pm(*r['cds_dh'])} & "
                    f"{fmt_pm(*r['cdc_dv'])} & {fmt_pm(*r['cdc_dh'])} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {out}")

    # ---------------------------------------------------------------- Pipeline
    # Build per-circuit and overall stats: mean residual_v, full-legal count,
    # full-design HPWL, plus circuit-level paired Wilcoxon p-values for
    # cd_raw_v vs pipe_vsr8_cd_v and cd_raw_full_h vs pipe_vsr8_cd_full_h.

    # circuit-level (mean over seeds within circuit) paired vectors
    circuit_means_raw_v, circuit_means_pipe_v = [], []
    circuit_means_raw_hf, circuit_means_pipe_hf = [], []
    full_legal_raw = 0
    full_legal_pipe = 0
    n_total = 0
    rows_pipe = []
    for c in CIRCUITS:
        rs = [r for r in r2_rows if r["circuit"] == c and r.get("cd_raw_v") is not None
              and r.get("pipe_vsr8_cd_v") is not None]
        if not rs:
            continue
        n_total += len(rs)
        full_legal_raw += sum(1 for r in rs if r["cd_raw_v"] == 0)
        full_legal_pipe += sum(1 for r in rs if r["pipe_vsr8_cd_v"] == 0)
        # Per-circuit means
        m_raw_v = S.mean(r["cd_raw_v"] for r in rs)
        m_pipe_v = S.mean(r["pipe_vsr8_cd_v"] for r in rs)
        m_raw_hf = S.mean(r["cd_raw_full_h"] for r in rs)
        m_pipe_hf = S.mean(r["pipe_vsr8_cd_full_h"] for r in rs)
        circuit_means_raw_v.append(m_raw_v)
        circuit_means_pipe_v.append(m_pipe_v)
        circuit_means_raw_hf.append(m_raw_hf)
        circuit_means_pipe_hf.append(m_pipe_hf)
        # Δ relative to baseline_v (raw draft)
        m_bv = S.mean(r["baseline_v"] for r in rs)
        m_bhf = S.mean(r["baseline_full_h"] for r in rs)
        rows_pipe.append({
            "c": c, "n": len(rs),
            "raw_v_abs": m_raw_v, "raw_v_dpct": pct(m_raw_v, m_bv),
            "pipe_v_abs": m_pipe_v, "pipe_v_dpct": pct(m_pipe_v, m_bv),
            "raw_hf_dpct": pct(m_raw_hf, m_bhf),
            "pipe_hf_dpct": pct(m_pipe_hf, m_bhf),
            "raw_legal": sum(1 for r in rs if r["cd_raw_v"] == 0),
            "pipe_legal": sum(1 for r in rs if r["pipe_vsr8_cd_v"] == 0),
        })

    # Wilcoxon (circuit-level n=6, paired)
    w_v = stats.wilcoxon(circuit_means_raw_v, circuit_means_pipe_v, zero_method="pratt")
    w_hf = stats.wilcoxon(circuit_means_raw_hf, circuit_means_pipe_hf, zero_method="pratt")
    p_v = float(w_v.pvalue)
    p_hf = float(w_hf.pvalue)

    # Write replacement table_downstream_pipeline.tex
    out = PAPER_FIG / "table_downstream_pipeline.tex"
    with open(out, "w") as f:
        f.write(r"\begin{tabular}{lr|rrc|rrc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& & \multicolumn{3}{c|}{raw $\to$ cd-sched} & "
                r"\multicolumn{3}{c}{raw $\to$ VSR($\lambda{=}8$) $\to$ cd-sched} \\" + "\n")
        f.write(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}" + "\n")
        f.write(r"Circuit & $n$ & $v_\text{post}$ & $\Delta h_f$\% & legal & "
                r"$v_\text{post}$ & $\Delta h_f$\% & legal \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows_pipe:
            f.write(f"{r['c']} & {r['n']} & "
                    f"${r['raw_v_abs']:.0f}$ & ${r['raw_hf_dpct']:+.1f}$ & "
                    f"{r['raw_legal']}/{r['n']} & "
                    f"${r['pipe_v_abs']:.0f}$ & ${r['pipe_hf_dpct']:+.1f}$ & "
                    f"{r['pipe_legal']}/{r['n']} \\\\\n")
        f.write(r"\midrule" + "\n")
        # Circuit-level medians + Wilcoxon row
        med_raw_v = S.median(r["raw_v_abs"] for r in rows_pipe)
        med_pipe_v = S.median(r["pipe_v_abs"] for r in rows_pipe)
        med_raw_hf = S.median(r["raw_hf_dpct"] for r in rows_pipe)
        med_pipe_hf = S.median(r["pipe_hf_dpct"] for r in rows_pipe)
        f.write(f"\\textbf{{median}} & -- & ${med_raw_v:.0f}$ & ${med_raw_hf:+.1f}$ & "
                f"{full_legal_raw}/{n_total} & "
                f"${med_pipe_v:.0f}$ & ${med_pipe_hf:+.1f}$ & "
                f"{full_legal_pipe}/{n_total} \\\\\n")
        # Wilcoxon row (circuit-level n=6, paired)
        f.write(f"\\multicolumn{{8}}{{l}}{{Circuit-level paired Wilcoxon "
                f"($n{{=}}6$): $v_\\text{{post}}$ $p={p_v:.3f}$; "
                f"$\\Delta h_f$ $p={p_hf:.3f}$.}} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {out}")

    # MD summary
    with open(REPO / "results" / "vsr_extra" / "round2_pipeline_stats.md", "w") as f:
        f.write("# Round-2 downstream pipeline statistics\n\n")
        f.write("## Per-circuit residual + full legality\n\n")
        for r in rows_pipe:
            f.write(f"- {r['c']}: raw $v_\\text{{post}}={r['raw_v_abs']:.0f}$, "
                    f"pipe $v_\\text{{post}}={r['pipe_v_abs']:.0f}$, "
                    f"legal raw={r['raw_legal']}/{r['n']}, "
                    f"pipe={r['pipe_legal']}/{r['n']}\n")
        f.write("\n## Cross-circuit medians\n\n")
        f.write(f"- raw → cd-sched residual_v median: {med_raw_v:.0f}\n")
        f.write(f"- vsr8 → cd-sched residual_v median: {med_pipe_v:.0f}\n")
        f.write(f"- raw → cd-sched full Δh median: {med_raw_hf:+.1f}%\n")
        f.write(f"- vsr8 → cd-sched full Δh median: {med_pipe_hf:+.1f}%\n")
        f.write(f"- full-legality (v=0): raw {full_legal_raw}/{n_total}, "
                f"pipe {full_legal_pipe}/{n_total}\n")
        f.write("\n## Circuit-level paired Wilcoxon (n=6)\n\n")
        f.write(f"- residual $v_\\text{{post}}$: $p = {p_v:.3f}$\n")
        f.write(f"- full-design $\\Delta h_f$: $p = {p_hf:.3f}$\n")

    print()
    print("=== Headlines (cd-sched pipeline) ===")
    print(f"Pipeline residual v_post median:  raw={med_raw_v:.0f}  vsr8+cd={med_pipe_v:.0f}")
    print(f"Pipeline Δh_full median:          raw={med_raw_hf:+.1f}%  vsr8+cd={med_pipe_hf:+.1f}%")
    print(f"Full-legality (v=0): raw {full_legal_raw}/{n_total}  pipe {full_legal_pipe}/{n_total}")
    print(f"Circuit-level Wilcoxon (n=6): v_post p={p_v:.3f}  Δh_f p={p_hf:.3f}")

    # ----------------------------------------------- cd-std sensitivity table
    if CDSTD.exists():
        cdstd_rows = json.load(open(CDSTD))
        cdstd_rows = [r for r in cdstd_rows if r.get("cdstd_raw_v") is not None
                      and r.get("pipe_vsr8_cdstd_v") is not None]
        cm_raw_v, cm_pipe_v = [], []
        cm_raw_hf, cm_pipe_hf = [], []
        full_legal_raw_s = 0; full_legal_pipe_s = 0; n_total_s = 0
        rows_s = []
        for c in CIRCUITS:
            rs = [r for r in cdstd_rows if r["circuit"] == c]
            if not rs: continue
            n_total_s += len(rs)
            full_legal_raw_s += sum(1 for r in rs if r["cdstd_raw_v"] == 0)
            full_legal_pipe_s += sum(1 for r in rs if r["pipe_vsr8_cdstd_v"] == 0)
            m_raw_v = S.mean(r["cdstd_raw_v"] for r in rs)
            m_pipe_v = S.mean(r["pipe_vsr8_cdstd_v"] for r in rs)
            m_raw_hf = S.mean(r["cdstd_raw_full_h"] for r in rs)
            m_pipe_hf = S.mean(r["pipe_vsr8_cdstd_full_h"] for r in rs)
            cm_raw_v.append(m_raw_v); cm_pipe_v.append(m_pipe_v)
            cm_raw_hf.append(m_raw_hf); cm_pipe_hf.append(m_pipe_hf)
            m_bv = S.mean(r["baseline_v"] for r in rs)
            m_bhf = S.mean(r["baseline_full_h"] for r in rs)
            rows_s.append({
                "c": c, "n": len(rs),
                "raw_v": m_raw_v, "pipe_v": m_pipe_v,
                "raw_hf": pct(m_raw_hf, m_bhf),
                "pipe_hf": pct(m_pipe_hf, m_bhf),
                "raw_legal": sum(1 for r in rs if r["cdstd_raw_v"] == 0),
                "pipe_legal": sum(1 for r in rs if r["pipe_vsr8_cdstd_v"] == 0),
            })
        if rows_s:
            w_v_s = stats.wilcoxon(cm_raw_v, cm_pipe_v, zero_method="pratt")
            w_hf_s = stats.wilcoxon(cm_raw_hf, cm_pipe_hf, zero_method="pratt")
            p_v_s = float(w_v_s.pvalue); p_hf_s = float(w_hf_s.pvalue)
            out = PAPER_FIG / "table_cdstd_pipeline.tex"
            with open(out, "w") as f:
                f.write(r"\begin{tabular}{lr|rrc|rrc}" + "\n")
                f.write(r"\toprule" + "\n")
                f.write(r"& & \multicolumn{3}{c|}{raw $\to$ cd-std} & "
                        r"\multicolumn{3}{c}{raw $\to$ VSR($\lambda{=}8$) $\to$ cd-std} \\" + "\n")
                f.write(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}" + "\n")
                f.write(r"Circuit & $n$ & $v_\text{post}$ & $\Delta h_f$\% & legal & "
                        r"$v_\text{post}$ & $\Delta h_f$\% & legal \\" + "\n")
                f.write(r"\midrule" + "\n")
                for r in rows_s:
                    f.write(f"{r['c']} & {r['n']} & "
                            f"${r['raw_v']:.0f}$ & ${r['raw_hf']:+.1f}$ & "
                            f"{r['raw_legal']}/{r['n']} & "
                            f"${r['pipe_v']:.0f}$ & ${r['pipe_hf']:+.1f}$ & "
                            f"{r['pipe_legal']}/{r['n']} \\\\\n")
                f.write(r"\midrule" + "\n")
                med_raw_v_s = S.median(r["raw_v"] for r in rows_s)
                med_pipe_v_s = S.median(r["pipe_v"] for r in rows_s)
                med_raw_hf_s = S.median(r["raw_hf"] for r in rows_s)
                med_pipe_hf_s = S.median(r["pipe_hf"] for r in rows_s)
                f.write(f"\\textbf{{median}} & -- & ${med_raw_v_s:.0f}$ & "
                        f"${med_raw_hf_s:+.1f}$ & {full_legal_raw_s}/{n_total_s} & "
                        f"${med_pipe_v_s:.0f}$ & ${med_pipe_hf_s:+.1f}$ & "
                        f"{full_legal_pipe_s}/{n_total_s} \\\\\n")
                f.write(f"\\multicolumn{{8}}{{l}}{{Circuit-level paired Wilcoxon "
                        f"($n{{=}}6$): $v_\\text{{post}}$ $p={p_v_s:.3f}$; "
                        f"$\\Delta h_f$ $p={p_hf_s:.3f}$.}} \\\\\n")
                f.write(r"\bottomrule" + "\n")
                f.write(r"\end{tabular}" + "\n")
            print(f"\nWrote {out}")
            print(f"=== Headlines (cd-std pipeline) ===")
            print(f"residual_v median: raw={med_raw_v_s:.0f}  vsr8+cdstd={med_pipe_v_s:.0f}")
            print(f"Δh_full median:    raw={med_raw_hf_s:+.1f}%  vsr8+cdstd={med_pipe_hf_s:+.1f}%")
            print(f"Wilcoxon n=6: p_v={p_v_s:.3f}  p_hf={p_hf_s:.3f}")
            print(f"Full-legality: raw {full_legal_raw_s}/{n_total_s}  pipe {full_legal_pipe_s}/{n_total_s}")


if __name__ == "__main__":
    main()
