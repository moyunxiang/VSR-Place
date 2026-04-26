"""Independent classical force-directed legalizer baseline (no VSR mask,
no severity, no HPWL-aware attraction unless explicitly enabled).

Two variants:
  - FD-pure:    only repulsion + boundary, all macros move.
  - FD+spring:  repulsion + boundary + net-spring attraction (NTUplace3-style),
                all macros move.

This addresses reviewer W2/Q5: 'Standard placement/legalization baselines
such as DREAMPlace, RePlAce, NTUPlace-style legalization, ... are not
actually benchmarked.' We benchmark a classical force-directed legalizer
with a comparable budget (T=100 steps, step_size=0.3) on the SAME cached
drafts so the comparison is apples-to-apples.

Input: data/ispd_placements_full.pkl (cached ChipDiffusion guided drafts).
Output: results/vsr_extra/classical_legalizer.json + summary table.
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from vsr_place.metrics.hpwl import compute_hpwl_from_edges  # noqa: E402
from vsr_place.renoising.local_repair import local_repair_loop  # noqa: E402
from vsr_place.verifier.verifier import Verifier  # noqa: E402

NAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
         "bigblue1", "bigblue2", "bigblue3", "bigblue4"]

DRAFTS_PKL = REPO / "data" / "ispd_placements_full.pkl"
OUT = REPO / "results" / "vsr_extra" / "classical_legalizer.json"
PAPER_FIG = REPO / "paper" / "figures"

NUM_STEPS = 100
STEP_SIZE = 0.3
SPRING_LAMBDA = 1.0  # NTUplace3-style net-spring weight


def evaluate(centers, sizes, edge_index, edge_attr, ver):
    fb = ver(centers, sizes)
    v = int(fb.global_stats["total_violations"])
    h = float(compute_hpwl_from_edges(centers, edge_index, edge_attr))
    return v, h


def main():
    drafts = pickle.load(open(DRAFTS_PKL, "rb"))
    print(f"Loaded {len(drafts)} cached drafts")

    rows = []
    for i, item in enumerate(drafts):
        ci = item["circuit_idx"]
        name = NAMES[ci]
        seed = item["seed"]
        centers0 = item["centers_bad"].cpu()
        sizes = item["sizes"].cpu()
        edge_index = item["edge_index"].cpu()
        edge_attr = item.get("edge_attr")
        edge_attr = edge_attr.cpu() if edge_attr is not None else None
        cw, ch = float(item["canvas_w"]), float(item["canvas_h"])

        ver = Verifier(canvas_width=cw, canvas_height=ch)
        b_v, b_h = evaluate(centers0, sizes, edge_index, edge_attr, ver)
        print(f"\n[{i+1}/{len(drafts)}] {name} seed={seed}  N={centers0.shape[0]}  "
              f"baseline v={b_v} h={b_h:.0f}", flush=True)

        # FD-pure: only repulsion + boundary, no mask, no spring
        t0 = time.time()
        c_fd = local_repair_loop(centers0, sizes, cw, ch,
                                 num_steps=NUM_STEPS, step_size=STEP_SIZE,
                                 only_mask=None,
                                 edge_index=None, hpwl_weight=0.0)
        t_fd = time.time() - t0
        fd_v, fd_h = evaluate(c_fd, sizes, edge_index, edge_attr, ver)

        # FD+spring: + net-spring attraction (classical analytical legalizer flavor)
        t0 = time.time()
        c_fs = local_repair_loop(centers0, sizes, cw, ch,
                                 num_steps=NUM_STEPS, step_size=STEP_SIZE,
                                 only_mask=None,
                                 edge_index=edge_index,
                                 hpwl_weight=SPRING_LAMBDA)
        t_fs = time.time() - t0
        fs_v, fs_h = evaluate(c_fs, sizes, edge_index, edge_attr, ver)

        # VSR-post (lambda=2): mask + repulsion + spring
        sev = ver(centers0, sizes).severity_vector
        t0 = time.time()
        c_v = local_repair_loop(centers0, sizes, cw, ch,
                                num_steps=NUM_STEPS, step_size=STEP_SIZE,
                                only_mask=(sev > 0),
                                edge_index=edge_index, hpwl_weight=2.0)
        t_v = time.time() - t0
        v_v, v_h = evaluate(c_v, sizes, edge_index, edge_attr, ver)

        row = {
            "circuit": name, "seed": seed, "n_macros": int(centers0.shape[0]),
            "baseline_v": b_v, "baseline_h": b_h,
            "fd_pure_v": fd_v, "fd_pure_h": fd_h, "fd_pure_t": t_fd,
            "fd_spring_v": fs_v, "fd_spring_h": fs_h, "fd_spring_t": t_fs,
            "vsr_post_v": v_v, "vsr_post_h": v_h, "vsr_post_t": t_v,
        }
        rows.append(row)
        print(f"  FD-pure:    v={fd_v:6d} ({(fd_v-b_v)/max(b_v,1)*100:+.1f}%)  "
              f"h={fd_h:8.1f} ({(fd_h-b_h)/max(abs(b_h),1e-9)*100:+.1f}%)  t={t_fd:.1f}s")
        print(f"  FD+spring:  v={fs_v:6d} ({(fs_v-b_v)/max(b_v,1)*100:+.1f}%)  "
              f"h={fs_h:8.1f} ({(fs_h-b_h)/max(abs(b_h),1e-9)*100:+.1f}%)  t={t_fs:.1f}s")
        print(f"  VSR-post:   v={v_v:6d} ({(v_v-b_v)/max(b_v,1)*100:+.1f}%)  "
              f"h={v_h:8.1f} ({(v_h-b_h)/max(abs(b_h),1e-9)*100:+.1f}%)  t={t_v:.1f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")

    # Aggregate per-circuit (mean over seeds)
    grouped = defaultdict(list)
    for r in rows: grouped[r["circuit"]].append(r)

    # CSV summary
    csv_out = REPO / "results" / "vsr_extra" / "classical_legalizer_summary.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["circuit", "n_seeds",
                    "fd_pure_dv_pct", "fd_pure_dh_pct",
                    "fd_spring_dv_pct", "fd_spring_dh_pct",
                    "vsr_dv_pct", "vsr_dh_pct"])
        for c, rs in grouped.items():
            n = len(rs)
            def avg_pct(field, base):
                return sum((r[field]-r[base])/max(abs(r[base]), 1e-9)*100 for r in rs)/n
            row = [c, n,
                   avg_pct("fd_pure_v","baseline_v"), avg_pct("fd_pure_h","baseline_h"),
                   avg_pct("fd_spring_v","baseline_v"), avg_pct("fd_spring_h","baseline_h"),
                   avg_pct("vsr_post_v","baseline_v"), avg_pct("vsr_post_h","baseline_h")]
            w.writerow(row)
    print(f"Wrote {csv_out}")

    # LaTeX paper-ready table
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    tex_out = PAPER_FIG / "table_classical_legalizer.tex"
    with open(tex_out, "w") as f:
        f.write(r"\begin{tabular}{lr|rr|rr|rr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"& seeds & \multicolumn{2}{c|}{FD-pure} & "
                r"\multicolumn{2}{c|}{FD+spring} & \multicolumn{2}{c}{VSR-post} \\" + "\n")
        f.write(r"Circuit & $n$ & $\Delta v$\% & $\Delta h$\% & "
                r"$\Delta v$\% & $\Delta h$\% & $\Delta v$\% & $\Delta h$\% \\" + "\n")
        f.write(r"\midrule" + "\n")
        all_dv_fd, all_dh_fd, all_dv_fs, all_dh_fs, all_dv_v, all_dh_v = [],[],[],[],[],[]
        for c, rs in grouped.items():
            n = len(rs)
            def avg_pct(field, base):
                return sum((r[field]-r[base])/max(abs(r[base]), 1e-9)*100 for r in rs)/n
            dv_fd = avg_pct("fd_pure_v","baseline_v"); dh_fd = avg_pct("fd_pure_h","baseline_h")
            dv_fs = avg_pct("fd_spring_v","baseline_v"); dh_fs = avg_pct("fd_spring_h","baseline_h")
            dv_v  = avg_pct("vsr_post_v","baseline_v"); dh_v  = avg_pct("vsr_post_h","baseline_h")
            all_dv_fd.append(dv_fd); all_dh_fd.append(dh_fd)
            all_dv_fs.append(dv_fs); all_dh_fs.append(dh_fs)
            all_dv_v.append(dv_v);   all_dh_v.append(dh_v)
            f.write(f"{c} & {n} & "
                    f"${dv_fd:+.1f}$ & ${dh_fd:+.1f}$ & "
                    f"${dv_fs:+.1f}$ & ${dh_fs:+.1f}$ & "
                    f"${dv_v:+.1f}$ & ${dh_v:+.1f}$ \\\\\n")
        f.write(r"\midrule" + "\n")
        import statistics as S
        f.write(f"\\textbf{{median}} & -- & "
                f"${S.median(all_dv_fd):+.1f}$ & ${S.median(all_dh_fd):+.1f}$ & "
                f"${S.median(all_dv_fs):+.1f}$ & ${S.median(all_dh_fs):+.1f}$ & "
                f"${S.median(all_dv_v):+.1f}$ & ${S.median(all_dh_v):+.1f}$ \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Wrote {tex_out}")

    print("\n=== Headlines ===")
    print(f"FD-pure   median:  Δv={S.median(all_dv_fd):+.1f}%  Δh={S.median(all_dh_fd):+.1f}%")
    print(f"FD+spring median:  Δv={S.median(all_dv_fs):+.1f}%  Δh={S.median(all_dh_fs):+.1f}%")
    print(f"VSR-post  median:  Δv={S.median(all_dv_v):+.1f}%  Δh={S.median(all_dh_v):+.1f}%")


if __name__ == "__main__":
    main()
