"""Aggregate round2_review.json into paper-ready tables.

Tables produced:
  - paper/figures/table_downstream_pipeline.tex
        decisive experiment: raw + cd_sched vs vsr8 + cd_sched
        per circuit + cross-circuit medians; metrics: v, h_macro, h_full, t
  - paper/figures/table_lambda8_main.tex
        Δv, Δh, residual v at lambda=8 on 4 seeds (R2 / Q3 / Q6)
  - paper/figures/table_classical_legalizer_24.tex
        FD-pure / FD+spring on the same 6 circuits × 4 seeds (R4)
  - paper/figures/table_baseline_sweeps_24.tex
        cg w=2,8 + RePaint t=0.3,0.5 on 4 seeds (R3, partial sweep)
  - results/vsr_extra/round2_review_summary.md

Inputs : results/vsr_extra/round2_review.json
"""
from __future__ import annotations

import csv
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "results" / "vsr_extra" / "round2_review.json"
PAPER_FIG = REPO / "paper" / "figures"
SUMMARY_MD = REPO / "results" / "vsr_extra" / "round2_review_summary.md"

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]


def pct(new, base):
    if base is None or new is None: return None
    base_abs = max(abs(base), 1e-9)
    return (new - base) / base_abs * 100


def main():
    rows = json.load(open(IN))
    rows = [r for r in rows if r.get("baseline_v") is not None]
    print(f"Loaded {len(rows)} valid trials")

    g = defaultdict(list)
    for r in rows: g[r["circuit"]].append(r)

    # --- 1) Downstream pipeline table (decisive) -----------------------------
    pipe_rows = []
    for c in CIRCUITS:
        rs = g.get(c, [])
        if not rs: continue
        n = len(rs)
        bv = S.mean(r["baseline_v"] for r in rs)
        bh = S.mean(r["baseline_h"] for r in rs)
        bhf = S.mean(r["baseline_full_h"] for r in rs if r.get("baseline_full_h") is not None)
        # raw + cd
        cdr_v = S.mean(r["cd_raw_v"] for r in rs if r.get("cd_raw_v") is not None)
        cdr_h = S.mean(r["cd_raw_h"] for r in rs if r.get("cd_raw_h") is not None)
        cdr_hf = S.mean(r["cd_raw_full_h"] for r in rs if r.get("cd_raw_full_h") is not None)
        cdr_t = S.mean(r["cd_raw_time"] for r in rs if r.get("cd_raw_time") is not None)
        # vsr8 + cd
        pp_v = S.mean(r["pipe_vsr8_cd_v"] for r in rs if r.get("pipe_vsr8_cd_v") is not None)
        pp_h = S.mean(r["pipe_vsr8_cd_h"] for r in rs if r.get("pipe_vsr8_cd_h") is not None)
        pp_hf = S.mean(r["pipe_vsr8_cd_full_h"] for r in rs if r.get("pipe_vsr8_cd_full_h") is not None)
        pp_t = S.mean(r["pipe_vsr8_cd_time"] for r in rs if r.get("pipe_vsr8_cd_time") is not None)
        pipe_rows.append((c, n, bv, bh, bhf, cdr_v, cdr_h, cdr_hf, cdr_t, pp_v, pp_h, pp_hf, pp_t))

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    with open(PAPER_FIG / "table_downstream_pipeline.tex", "w") as f:
        f.write(r"\begin{tabular}{lr|rrrr|rrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& seeds & \multicolumn{4}{c|}{raw $\to$ cd-sched} & "
                r"\multicolumn{4}{c}{raw $\to$ VSR($\lambda{=}8$) $\to$ cd-sched} \\" + "\n")
        f.write(r"Circuit & $n$ & $\Delta v$\% & $\Delta h_m$\% & $\Delta h_f$\% & $t$\,s "
                r"& $\Delta v$\% & $\Delta h_m$\% & $\Delta h_f$\% & $t$\,s \\" + "\n")
        f.write(r"\midrule" + "\n")
        all_dv_r, all_dh_r, all_dhf_r, all_t_r = [], [], [], []
        all_dv_p, all_dh_p, all_dhf_p, all_t_p = [], [], [], []
        for (c, n, bv, bh, bhf, cdr_v, cdr_h, cdr_hf, cdr_t, pp_v, pp_h, pp_hf, pp_t) in pipe_rows:
            dv_r = pct(cdr_v, bv); dh_r = pct(cdr_h, bh); dhf_r = pct(cdr_hf, bhf)
            dv_p = pct(pp_v, bv); dh_p = pct(pp_h, bh); dhf_p = pct(pp_hf, bhf)
            all_dv_r.append(dv_r); all_dh_r.append(dh_r); all_dhf_r.append(dhf_r); all_t_r.append(cdr_t)
            all_dv_p.append(dv_p); all_dh_p.append(dh_p); all_dhf_p.append(dhf_p); all_t_p.append(pp_t)
            f.write(f"{c} & {n} & ${dv_r:+.1f}$ & ${dh_r:+.1f}$ & ${dhf_r:+.1f}$ & {cdr_t:.0f} & "
                    f"${dv_p:+.1f}$ & ${dh_p:+.1f}$ & ${dhf_p:+.1f}$ & {pp_t:.0f} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(f"\\textbf{{median}} & -- & "
                f"${S.median(all_dv_r):+.1f}$ & ${S.median(all_dh_r):+.1f}$ & "
                f"${S.median(all_dhf_r):+.1f}$ & {S.median(all_t_r):.0f} & "
                f"${S.median(all_dv_p):+.1f}$ & ${S.median(all_dh_p):+.1f}$ & "
                f"${S.median(all_dhf_p):+.1f}$ & {S.median(all_t_p):.0f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {PAPER_FIG / 'table_downstream_pipeline.tex'}")

    # --- 2) λ=8 main table on 4 seeds ----------------------------------------
    with open(PAPER_FIG / "table_lambda8_main.tex", "w") as f:
        f.write(r"\begin{tabular}{lr|rrr|r}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& seeds & \multicolumn{3}{c|}{VSR-post ($\lambda{=}8$)} & residual \\" + "\n")
        f.write(r"Circuit & $n$ & $\Delta v$\% & $\Delta h_m$\% & $\Delta h_f$\% & $v_\text{post}$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        all_dv, all_dhm, all_dhf, all_resv = [], [], [], []
        for c in CIRCUITS:
            rs = [r for r in g.get(c, []) if r.get("vsr8_v") is not None]
            if not rs: continue
            bv = S.mean(r["baseline_v"] for r in rs)
            bh = S.mean(r["baseline_h"] for r in rs)
            bhf = S.mean(r["baseline_full_h"] for r in rs if r.get("baseline_full_h") is not None)
            v8 = S.mean(r["vsr8_v"] for r in rs)
            h8 = S.mean(r["vsr8_h"] for r in rs)
            h8f = S.mean(r["vsr8_full_h"] for r in rs if r.get("vsr8_full_h") is not None)
            dv = pct(v8, bv); dhm = pct(h8, bh); dhf = pct(h8f, bhf)
            all_dv.append(dv); all_dhm.append(dhm); all_dhf.append(dhf); all_resv.append(v8)
            f.write(f"{c} & {len(rs)} & ${dv:+.1f}$ & ${dhm:+.1f}$ & ${dhf:+.1f}$ & ${v8:.0f}$ \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(f"\\textbf{{median}} & -- & ${S.median(all_dv):+.1f}$ & "
                f"${S.median(all_dhm):+.1f}$ & ${S.median(all_dhf):+.1f}$ & "
                f"${S.median(all_resv):.0f}$ \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {PAPER_FIG / 'table_lambda8_main.tex'}")

    # --- 3) Classical FD on 24 trials ----------------------------------------
    with open(PAPER_FIG / "table_classical_legalizer_24.tex", "w") as f:
        f.write(r"\begin{tabular}{lr|rr|rr|rr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& seeds & \multicolumn{2}{c|}{FD-pure} & "
                r"\multicolumn{2}{c|}{FD+spring} & "
                r"\multicolumn{2}{c}{VSR-post ($\lambda{=}8$)} \\" + "\n")
        f.write(r"Circuit & $n$ & $\Delta v$\% & $\Delta h$\% & "
                r"$\Delta v$\% & $\Delta h$\% & $\Delta v$\% & $\Delta h$\% \\" + "\n")
        f.write(r"\midrule" + "\n")
        a_dv_fd, a_dh_fd = [], []
        a_dv_fs, a_dh_fs = [], []
        a_dv_v, a_dh_v = [], []
        for c in CIRCUITS:
            rs = [r for r in g.get(c, []) if r.get("fd_pure_v") is not None]
            if not rs: continue
            bv = S.mean(r["baseline_v"] for r in rs)
            bh = S.mean(r["baseline_h"] for r in rs)
            dv_fd = pct(S.mean(r["fd_pure_v"] for r in rs), bv)
            dh_fd = pct(S.mean(r["fd_pure_h"] for r in rs), bh)
            dv_fs = pct(S.mean(r["fd_spring_v"] for r in rs if r.get("fd_spring_v") is not None), bv)
            dh_fs = pct(S.mean(r["fd_spring_h"] for r in rs if r.get("fd_spring_h") is not None), bh)
            dv_v  = pct(S.mean(r["vsr8_v"] for r in rs), bv)
            dh_v  = pct(S.mean(r["vsr8_h"] for r in rs), bh)
            a_dv_fd.append(dv_fd); a_dh_fd.append(dh_fd)
            a_dv_fs.append(dv_fs); a_dh_fs.append(dh_fs)
            a_dv_v.append(dv_v);   a_dh_v.append(dh_v)
            f.write(f"{c} & {len(rs)} & ${dv_fd:+.1f}$ & ${dh_fd:+.1f}$ & "
                    f"${dv_fs:+.1f}$ & ${dh_fs:+.1f}$ & "
                    f"${dv_v:+.1f}$ & ${dh_v:+.1f}$ \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(f"\\textbf{{median}} & -- & "
                f"${S.median(a_dv_fd):+.1f}$ & ${S.median(a_dh_fd):+.1f}$ & "
                f"${S.median(a_dv_fs):+.1f}$ & ${S.median(a_dh_fs):+.1f}$ & "
                f"${S.median(a_dv_v):+.1f}$ & ${S.median(a_dh_v):+.1f}$ \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {PAPER_FIG / 'table_classical_legalizer_24.tex'}")

    # --- 4) Baseline sweeps on 4 seeds (cg, repaint partial) ----------------
    cg_keys = [("cg2.0_v", "cg2.0_h"), ("cg8.0_v", "cg8.0_h")]
    rp_keys = [("rp0.3_v", "rp0.3_h"), ("rp0.5_v", "rp0.5_h")]
    with open(PAPER_FIG / "table_baseline_sweeps_24.tex", "w") as f:
        f.write(r"\begin{tabular}{lr|rr|rr|rr|rr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& seeds & \multicolumn{2}{c|}{cg $w{=}2$} "
                r"& \multicolumn{2}{c|}{cg $w{=}8$} "
                r"& \multicolumn{2}{c|}{RePaint $t{=}0.3$} "
                r"& \multicolumn{2}{c}{RePaint $t{=}0.5$} \\" + "\n")
        f.write(r"Circuit & $n$ & $\Delta v$\% & $\Delta h$\% & "
                r"$\Delta v$\% & $\Delta h$\% & $\Delta v$\% & $\Delta h$\% & "
                r"$\Delta v$\% & $\Delta h$\% \\" + "\n")
        f.write(r"\midrule" + "\n")
        for c in CIRCUITS:
            rs = g.get(c, [])
            if not rs: continue
            bv = S.mean(r["baseline_v"] for r in rs)
            bh = S.mean(r["baseline_h"] for r in rs)
            cells = []
            for vk, hk in cg_keys + rp_keys:
                vs = [r[vk] for r in rs if r.get(vk) is not None]
                hs = [r[hk] for r in rs if r.get(hk) is not None]
                if vs and hs:
                    dv = pct(S.mean(vs), bv); dh = pct(S.mean(hs), bh)
                    cells.append(f"${dv:+.1f}$ & ${dh:+.1f}$")
                else:
                    cells.append("-- & --")
            f.write(f"{c} & {len(rs)} & " + " & ".join(cells) + r" \\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {PAPER_FIG / 'table_baseline_sweeps_24.tex'}")

    # --- Summary MD -----------------------------------------------------------
    with open(SUMMARY_MD, "w") as f:
        f.write("# Round-2 review GPU experiment summary\n\n")
        f.write(f"Trials: {len(rows)}\n\n")
        f.write("## Decisive downstream pipeline\n\n")
        if all_dv_r and all_dv_p:
            f.write(f"- raw → cd-sched          median: Δv={S.median(all_dv_r):+.1f}%  "
                    f"Δh_m={S.median(all_dh_r):+.1f}%  Δh_f={S.median(all_dhf_r):+.1f}%  "
                    f"t={S.median(all_t_r):.0f}s\n")
            f.write(f"- raw → VSR(λ=8) → cd-sched median: Δv={S.median(all_dv_p):+.1f}%  "
                    f"Δh_m={S.median(all_dh_p):+.1f}%  Δh_f={S.median(all_dhf_p):+.1f}%  "
                    f"t={S.median(all_t_p):.0f}s\n")
        f.write("\n## λ=8 main on 4 seeds\n\n")
        if all_dv:
            f.write(f"- median Δv={S.median(all_dv):+.1f}%  Δh_m={S.median(all_dhm):+.1f}%  "
                    f"Δh_f={S.median(all_dhf):+.1f}%  residual_v median={S.median(all_resv):.0f}\n")
        f.write("\n## Classical FD on 4 seeds\n\n")
        if a_dv_fd:
            f.write(f"- FD-pure        median: Δv={S.median(a_dv_fd):+.1f}%  Δh={S.median(a_dh_fd):+.1f}%\n")
            f.write(f"- FD+spring      median: Δv={S.median(a_dv_fs):+.1f}%  Δh={S.median(a_dh_fs):+.1f}%\n")
            f.write(f"- VSR-post(λ=8)  median: Δv={S.median(a_dv_v):+.1f}%  Δh={S.median(a_dh_v):+.1f}%\n")
    print(f"Wrote {SUMMARY_MD}")
    print()
    print("=== Headlines ===")
    if all_dv_r and all_dv_p:
        print(f"  raw+cd      median Δv={S.median(all_dv_r):+.1f}%  Δh_m={S.median(all_dh_r):+.1f}%  Δh_f={S.median(all_dhf_r):+.1f}%")
        print(f"  vsr8+cd     median Δv={S.median(all_dv_p):+.1f}%  Δh_m={S.median(all_dh_p):+.1f}%  Δh_f={S.median(all_dhf_p):+.1f}%")
    if all_dv:
        print(f"  vsr8 (mac)  median Δv={S.median(all_dv):+.1f}%  Δh_m={S.median(all_dhm):+.1f}%  Δh_f={S.median(all_dhf):+.1f}%  resv={S.median(all_resv):.0f}")


if __name__ == "__main__":
    main()
