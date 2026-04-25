"""Robust statistics for the NeurIPS-rigor unified main JSON.

Reads results/ispd2005/main_neurips.json (one row per (circuit, seed) with
7 methods side-by-side) and produces:
  - per-circuit mean +/- std for each method
  - cross-circuit median (robust to small-edge bigblue1 outlier)
  - 95% bootstrap CI on the mean
  - paired Wilcoxon: VSR-post vs every baseline; VSR-intra vs every baseline
  - strict-Pareto count per method

Output:
  results/ispd2005/neurips_stats.json
  paper/figures/table_main_neurips.tex
  paper/figures/table_wilcoxon.tex
"""
from __future__ import annotations

import json
import math
import statistics as S
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "ispd2005"
PAPER_FIG = REPO / "paper" / "figures"

# All methods in main_neurips.json
METHODS = [
    "baseline",
    "cg_strong",
    "vsr_post",
    "repaint_bin",
    "vsr_intra",
    "cd_std",
    "cd_sched",
]
# Methods that perform actual repair / sampling (excludes raw baseline)
REPAIR_METHODS = [m for m in METHODS if m != "baseline"]


def bootstrap_ci(values, n_resample=10_000, alpha=0.05, seed=0):
    import random
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = []
    for _ in range(n_resample):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_resample * alpha / 2)]
    hi = means[int(n_resample * (1 - alpha / 2))]
    return (S.mean(values), lo, hi)


def wilcoxon(x, y):
    if len(x) != len(y):
        raise ValueError("paired samples required")
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n == 0:
        return (0.0, 1.0)
    abs_diffs = sorted([(abs(d), 1 if d > 0 else -1) for d in diffs])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    W_pos = sum(r for r, (_, sgn) in zip(ranks, abs_diffs) if sgn > 0)
    W_neg = sum(r for r, (_, sgn) in zip(ranks, abs_diffs) if sgn < 0)
    W = min(W_pos, W_neg)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return (W, 1.0)
    z = (W - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (W, p)


def pct_v(row, method):
    base_v = row.get("baseline_v")
    v = row.get(f"{method}_v")
    if base_v is None or base_v == 0 or v is None:
        return None
    return (v - base_v) / base_v * 100


def pct_h(row, method):
    base_h = row.get("baseline_h")
    h = row.get(f"{method}_h")
    if base_h is None or h is None or abs(base_h) < 1e-9:
        return None
    return (h - base_h) / abs(base_h) * 100


def pair_was_strict(row, method):
    """Was (Δv<0 AND Δh<0) for this method on this row?"""
    dv = pct_v(row, method)
    dh = pct_h(row, method)
    return dv is not None and dh is not None and dv < 0 and dh < 0


def time_for(row, method):
    return row.get(f"{method}_time")


def per_circuit_summary(rows):
    g = defaultdict(list)
    for r in rows:
        if r.get("error") or r.get("baseline_v") is None:
            continue
        g[r["circuit"]].append(r)

    summary = {}
    for c in sorted(g.keys()):
        rs = g[c]
        circ = {"n_macros": rs[0]["n_macros"], "n_seeds": len(rs)}
        for m in METHODS:
            dv = [pct_v(r, m) for r in rs if pct_v(r, m) is not None]
            dh = [pct_h(r, m) for r in rs if pct_h(r, m) is not None]
            t = [time_for(r, m) for r in rs if time_for(r, m) is not None]
            if not dv:
                circ[m] = {"n": 0}
                continue
            mean_dv, lo_dv, hi_dv = bootstrap_ci(dv)
            mean_dh, lo_dh, hi_dh = bootstrap_ci(dh)
            circ[m] = {
                "n": len(dv),
                "dv_pct_mean": mean_dv,
                "dv_pct_std": S.pstdev(dv) if len(dv) > 1 else 0.0,
                "dv_pct_ci95": [lo_dv, hi_dv],
                "dh_pct_mean": mean_dh,
                "dh_pct_std": S.pstdev(dh) if len(dh) > 1 else 0.0,
                "dh_pct_ci95": [lo_dh, hi_dh],
                "time_mean": S.mean(t) if t else None,
            }
        summary[c] = circ
    return summary


def cross_circuit_medians(summary):
    out = {}
    for m in METHODS:
        dv_means = [v[m]["dv_pct_mean"] for v in summary.values() if v.get(m, {}).get("n", 0) > 0]
        dh_means = [v[m]["dh_pct_mean"] for v in summary.values() if v.get(m, {}).get("n", 0) > 0]
        out[m] = {
            "n_circuits": len(dv_means),
            "dv_median": S.median(dv_means) if dv_means else None,
            "dv_mean": S.mean(dv_means) if dv_means else None,
            "dh_median": S.median(dh_means) if dh_means else None,
            "dh_mean": S.mean(dh_means) if dh_means else None,
        }
    return out


def strict_pareto(rows):
    out = {}
    valid = [r for r in rows if not r.get("error") and r.get("baseline_v") is not None]
    for m in METHODS:
        n = sum(1 for r in valid if pct_v(r, m) is not None)
        strict = sum(1 for r in valid if pair_was_strict(r, m))
        out[m] = {"strict": strict, "n": n}
    return out


def stat_tests(rows):
    """Paired Wilcoxon between each pair of methods."""
    valid = [r for r in rows if not r.get("error") and r.get("baseline_v") is not None]
    out = {}
    pairs = [
        ("vsr_post", "cd_std"),
        ("vsr_post", "cd_sched"),
        ("vsr_post", "cg_strong"),
        ("vsr_post", "repaint_bin"),
        ("vsr_post", "vsr_intra"),
        ("vsr_intra", "cd_std"),
        ("vsr_intra", "cg_strong"),
        ("vsr_intra", "repaint_bin"),
    ]
    for a, b in pairs:
        dv_a = [pct_v(r, a) for r in valid]
        dv_b = [pct_v(r, b) for r in valid]
        dh_a = [pct_h(r, a) for r in valid]
        dh_b = [pct_h(r, b) for r in valid]
        # Filter to rows where BOTH methods have data
        keep = [i for i in range(len(valid)) if dv_a[i] is not None and dv_b[i] is not None]
        if not keep:
            continue
        a_dv = [dv_a[i] for i in keep]
        b_dv = [dv_b[i] for i in keep]
        a_dh = [dh_a[i] for i in keep]
        b_dh = [dh_b[i] for i in keep]
        Wv, pv = wilcoxon(a_dv, b_dv)
        Wh, ph = wilcoxon(a_dh, b_dh)
        out[f"{a}_vs_{b}"] = {
            "n": len(keep),
            "dv": {"W": Wv, "p": pv},
            "dh": {"W": Wh, "p": ph},
        }
    return out


# ---- LaTeX table emitters ----------------------------------------------

def fmt_pct(stat, mean_key, std_key):
    if stat.get("n", 0) == 0:
        return r"\textemdash"
    return f"${stat[mean_key]:+.1f}{{\\scriptscriptstyle\\pm{stat[std_key]:.1f}}}$"


_LBL = {
    "vsr_post": "VSR-post",
    "vsr_intra": "VSR-intra",
    "cd_std": "CD-std",
    "cd_sched": "CD-sched",
    "cg_strong": "cg-strong",
    "repaint_bin": "RePaint-bin",
}


def _emit_table(summary, path, show_methods, fontsize=None):
    n = len(show_methods)
    head_cols = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{_LBL[m]}}}" for m in show_methods
    )
    cmid = " ".join(
        rf"\cmidrule(lr){{{i*2+3}-{i*2+4}}}" for i in range(n)
    )
    sub = " & ".join(r"$\Delta v$ & $\Delta h$" for _ in show_methods)
    lines = [
        r"\begin{tabular}{lr" + "rr" * n + r"}",
        r"\toprule",
        r" & & " + head_cols + r" \\",
        cmid,
        r"Circuit & $N$ & " + sub + r" \\",
        r"\midrule",
    ]
    for c, v in sorted(summary.items()):
        cells = [c, str(v["n_macros"])]
        for m in show_methods:
            cells.append(fmt_pct(v.get(m, {}), "dv_pct_mean", "dv_pct_std"))
            cells.append(fmt_pct(v.get(m, {}), "dh_pct_mean", "dh_pct_std"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def emit_main_table(summary, path):
    """Slim main-paper version: 4 most important methods, fits 1-column."""
    _emit_table(summary, path, ["vsr_post", "vsr_intra", "cd_std", "cd_sched"])


def emit_full_table(summary, path):
    """Full supplement version: all 6 methods."""
    _emit_table(summary, path,
                ["vsr_post", "vsr_intra", "cd_std", "cd_sched", "cg_strong", "repaint_bin"])


def emit_wilcoxon_table(stats, path):
    """Format Wilcoxon paired-test table nicely.

    - Primary VSR-post comparisons first, then VSR-intra ones, then any others.
    - p-values: use $<10^{-3}$ when p<0.001 (instead of 0.0000); else 4 decimals.
    - Stars rendered as LaTeX superscript with a thin space, not glued to the
      number: e.g.\ ``$<\!10^{-3}$\,$^{\star\star\star}$''.
    """
    lines = [
        r"\begin{tabular}{lrll}",
        r"\toprule",
        r"Comparison & $n$ & $p_{\Delta v}$ & $p_{\Delta h}$ \\",
        r"\midrule",
    ]

    def fmt(p):
        if p < 1e-3:
            num = r"$<\!10^{-3}$"
            stars = r"\,$^{\star\star\star}$"
        elif p < 1e-2:
            num = f"${p:.4f}$"
            stars = r"\,$^{\star\star}$"
        elif p < 5e-2:
            num = f"${p:.4f}$"
            stars = r"\,$^{\star}$"
        else:
            num = f"${p:.4f}$"
            stars = r""
        return num + stars

    # Row ordering: put vsr_post comparisons first, then vsr_intra
    def sort_key(k):
        head = k.split("_vs_")[0]
        order = {"vsr_post": 0, "vsr_intra": 1}.get(head, 2)
        return (order, k)

    for k in sorted(stats.keys(), key=sort_key):
        v = stats[k]
        # Pretty-print row label
        a, b = k.split("_vs_")
        nice = a.replace("_", "-") + r" vs.\ " + b.replace("_", "-")
        lines.append(f"{nice} & {v['n']} & {fmt(v['dv']['p'])} & {fmt(v['dh']['p'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    in_path = RESULTS / "main_neurips.json"
    if not in_path.exists():
        print(f"FATAL: {in_path} not found. Run run_main_neurips.py first.")
        return 1
    rows = json.load(open(in_path))
    print(f"Loaded {len(rows)} rows from {in_path}")

    summary = per_circuit_summary(rows)
    medians = cross_circuit_medians(summary)
    strict = strict_pareto(rows)
    stats = stat_tests(rows)

    out = {
        "n_rows": len(rows),
        "per_circuit": summary,
        "cross_circuit_medians": medians,
        "strict_pareto": strict,
        "wilcoxon": stats,
    }
    out_path = RESULTS / "neurips_stats.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {out_path}")

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    emit_main_table(summary, PAPER_FIG / "table_main_neurips.tex")
    emit_full_table(summary, PAPER_FIG / "table_main_full.tex")
    emit_wilcoxon_table(stats, PAPER_FIG / "table_wilcoxon.tex")
    print(f"Wrote {PAPER_FIG/'table_main_neurips.tex'} (slim)")
    print(f"Wrote {PAPER_FIG/'table_main_full.tex'} (6-method)")
    print(f"Wrote {PAPER_FIG/'table_wilcoxon.tex'}")

    # ---- print headline ----
    print("\n=== Cross-circuit medians ===")
    for m, s in medians.items():
        if s.get("dv_median") is not None:
            print(f"  {m:<14} median Δv={s['dv_median']:+.1f}%  median Δh={s['dh_median']:+.1f}%  n={s['n_circuits']} circuits")
    print("\n=== Strict Pareto improvement (Δv<0 AND Δh<0) ===")
    for m, s in strict.items():
        print(f"  {m:<14} {s['strict']}/{s['n']} (circuit, seed) pairs strictly improved")
    print("\n=== Wilcoxon paired tests ===")
    for k, v in stats.items():
        sv = "***" if v["dv"]["p"] < 0.001 else ("**" if v["dv"]["p"] < 0.01 else ("*" if v["dv"]["p"] < 0.05 else ""))
        sh = "***" if v["dh"]["p"] < 0.001 else ("**" if v["dh"]["p"] < 0.01 else ("*" if v["dh"]["p"] < 0.05 else ""))
        print(f"  {k:<32}  n={v['n']}  Δv: p={v['dv']['p']:.4g}{sv}  Δh: p={v['dh']['p']:.4g}{sh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
