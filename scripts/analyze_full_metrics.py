"""Aggregate the full_metrics.json (overlap area, max overlap, full-design HPWL).

Outputs:
  - results/vsr_extra/full_metrics_summary.{csv,md}
  - paper/figures/table_full_metrics.tex (paper-ready)
"""
import csv
import json
import statistics as S
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "results" / "vsr_extra" / "full_metrics.json"
OUT_DIR = REPO / "results" / "vsr_extra"
PAPER_FIG = REPO / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]


def main():
    rows = json.load(open(IN))
    g = defaultdict(list)
    for r in rows:
        if r.get("baseline_v") is None:
            continue
        g[r["circuit"]].append(r)

    csv_rows = []
    md_rows = []
    for c in CIRCUITS:
        rs = g.get(c, [])
        if not rs:
            continue
        bv = S.mean(r["baseline_v"] for r in rs)
        bh_full = S.mean(r["baseline_full_hpwl"] for r in rs)
        b_oa = S.mean(r["baseline_overlap_area"] for r in rs)

        # vsr_post
        pv = S.mean(r["vsr_post_v"] for r in rs)
        ph_full = S.mean(r["vsr_post_full_hpwl"] for r in rs)
        p_oa = S.mean(r["vsr_post_overlap_area"] for r in rs)

        # vsr_intra
        iv = S.mean(r["vsr_intra_v"] for r in rs)
        ih_full = S.mean(r["vsr_intra_full_hpwl"] for r in rs)
        i_oa = S.mean(r["vsr_intra_overlap_area"] for r in rs)

        csv_rows.append({
            "circuit": c, "n_seeds": len(rs),
            "base_v": bv, "base_oa": b_oa, "base_full_h": bh_full,
            "post_v": pv, "post_oa": p_oa, "post_full_h": ph_full,
            "intra_v": iv, "intra_oa": i_oa, "intra_full_h": ih_full,
            "post_dv_pct": (pv - bv) / max(bv, 1) * 100,
            "post_doa_pct": (p_oa - b_oa) / max(b_oa, 1e-6) * 100,
            "post_dh_full_pct": (ph_full - bh_full) / max(bh_full, 1e-6) * 100,
            "intra_dv_pct": (iv - bv) / max(bv, 1) * 100,
            "intra_doa_pct": (i_oa - b_oa) / max(b_oa, 1e-6) * 100,
            "intra_dh_full_pct": (ih_full - bh_full) / max(bh_full, 1e-6) * 100,
        })

    # CSV
    with open(OUT_DIR / "full_metrics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows: w.writerow(r)
    print(f"Wrote {OUT_DIR/'full_metrics_summary.csv'}")

    # Markdown
    with open(OUT_DIR / "full_metrics_summary.md", "w") as f:
        f.write("| circuit | Δviol% | Δoverlap_area% | Δfull_HPWL% (post) | Δfull_HPWL% (intra) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in csv_rows:
            f.write(f"| {r['circuit']} | {r['post_dv_pct']:+.1f} | {r['post_doa_pct']:+.1f} | {r['post_dh_full_pct']:+.1f} | {r['intra_dh_full_pct']:+.1f} |\n")

    # LaTeX
    with open(PAPER_FIG / "table_full_metrics.tex", "w") as f:
        f.write(r"\begin{tabular}{lrrrr|rr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& \multicolumn{4}{c|}{VSR-post (\,$\lambda{=}2$)} & \multicolumn{2}{c}{VSR-intra-soft} \\" + "\n")
        f.write(r"\cmidrule(lr){2-5} \cmidrule(lr){6-7}" + "\n")
        f.write(r"Circuit & $\Delta v$\% & $\Delta\text{oa}$\% & $\Delta h_{\text{macro}}$\% & $\Delta h_{\text{full}}$\% & $\Delta v$\% & $\Delta h_{\text{full}}$\% \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in csv_rows:
            # macros-only HPWL delta (we have base_v but not bh in csv, recompute)
            rs = g[r["circuit"]]
            bh_macro = S.mean(rr["baseline_h"] for rr in rs)
            ph_macro = S.mean(rr["vsr_post_h"] for rr in rs)
            dh_macro = (ph_macro - bh_macro) / max(abs(bh_macro), 1e-9) * 100
            f.write(f"{r['circuit']} & "
                    f"${r['post_dv_pct']:+.1f}$ & "
                    f"${r['post_doa_pct']:+.1f}$ & "
                    f"${dh_macro:+.1f}$ & "
                    f"${r['post_dh_full_pct']:+.1f}$ & "
                    f"${r['intra_dv_pct']:+.1f}$ & "
                    f"${r['intra_dh_full_pct']:+.1f}$ \\\\\n")
        # cross-circuit medians
        f.write(r"\midrule" + "\n")
        med_dv_p = S.median(r["post_dv_pct"] for r in csv_rows)
        med_doa_p = S.median(r["post_doa_pct"] for r in csv_rows)
        med_dh_full_p = S.median(r["post_dh_full_pct"] for r in csv_rows)
        med_dv_i = S.median(r["intra_dv_pct"] for r in csv_rows)
        med_dh_full_i = S.median(r["intra_dh_full_pct"] for r in csv_rows)
        # macros-only median over circuits
        macros_hs = [(S.mean(rr["vsr_post_h"] for rr in g[c]) -
                      S.mean(rr["baseline_h"] for rr in g[c])) /
                     max(abs(S.mean(rr["baseline_h"] for rr in g[c])), 1e-9) * 100
                     for c in CIRCUITS if g[c]]
        med_dh_macro = S.median(macros_hs)
        f.write(f"\\textbf{{median}} & ${med_dv_p:+.1f}$ & ${med_doa_p:+.1f}$ & "
                f"${med_dh_macro:+.1f}$ & ${med_dh_full_p:+.1f}$ & "
                f"${med_dv_i:+.1f}$ & ${med_dh_full_i:+.1f}$ \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {PAPER_FIG/'table_full_metrics.tex'}")

    # Print key headlines
    print("\n=== Key headlines ===")
    print(f"Post-VSR median:")
    print(f"  Δv (count):           {med_dv_p:+.1f}%")
    print(f"  Δoverlap_area:        {med_doa_p:+.1f}%   (the metric reviewer asked for)")
    print(f"  Δh (macros-only):     {med_dh_macro:+.1f}%   (paper Table 1)")
    print(f"  Δh (full-design):     {med_dh_full_p:+.1f}%   (NEW, addresses W5/E4)")
    print()
    print(f"Intra-VSR median:")
    print(f"  Δv:           {med_dv_i:+.1f}%")
    print(f"  Δh (full):    {med_dh_full_i:+.1f}%")


if __name__ == "__main__":
    main()
