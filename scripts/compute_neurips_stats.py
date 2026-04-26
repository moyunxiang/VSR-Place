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


def stat_tests_circuit_level(rows):
    """Paired Wilcoxon at the CIRCUIT level (n=6).

    Conservative version of the seed-level test: average across seeds within
    each circuit first, then run paired Wilcoxon over the 6 paired circuit
    means.  This is the right test if circuit-to-circuit variance dominates
    seed-level noise (which it does on ISPD2005 due to 4-orders-of-magnitude
    HPWL scale variation across circuits).
    """
    from collections import defaultdict as DD
    g = DD(list)
    valid = [r for r in rows if not r.get("error") and r.get("baseline_v") is not None]
    for r in valid:
        g[r["circuit"]].append(r)

    def circ_mean(c, key):
        vs = [r[key] for r in g[c] if r.get(key) is not None]
        return sum(vs) / len(vs) if vs else None

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
    circuits = sorted(g.keys())
    for a, b in pairs:
        a_dv, b_dv, a_dh, b_dh = [], [], [], []
        for c in circuits:
            base_v = circ_mean(c, "baseline_v")
            base_h = circ_mean(c, "baseline_h")
            va = circ_mean(c, f"{a}_v"); ha = circ_mean(c, f"{a}_h")
            vb = circ_mean(c, f"{b}_v"); hb = circ_mean(c, f"{b}_h")
            if None in (va, vb, ha, hb): continue
            a_dv.append((va - base_v) / max(base_v, 1) * 100)
            b_dv.append((vb - base_v) / max(base_v, 1) * 100)
            a_dh.append((ha - base_h) / max(abs(base_h), 1e-9) * 100)
            b_dh.append((hb - base_h) / max(abs(base_h), 1e-9) * 100)
        if not a_dv: continue
        Wv, pv = wilcoxon(a_dv, b_dv)
        Wh, ph = wilcoxon(a_dh, b_dh)
        out[f"{a}_vs_{b}"] = {
            "n_circuits": len(a_dv),
            "dv": {"W": Wv, "p": pv},
            "dh": {"W": Wh, "p": ph},
        }
    return out


def fully_legal_stats(rows):
    """Count trials achieving v=0 (fully legal) per method."""
    valid = [r for r in rows if not r.get("error") and r.get("baseline_v") is not None]
    methods = ["baseline", "vsr_post", "vsr_intra", "cd_std", "cd_sched",
               "cg_strong", "repaint_bin"]
    out = {}
    for m in methods:
        vs = [r.get(f"{m}_v") for r in valid if r.get(f"{m}_v") is not None]
        if not vs: continue
        out[m] = {
            "n": len(vs),
            "fully_legal": sum(1 for v in vs if v == 0),
            "v_le_5":      sum(1 for v in vs if v <= 5),
            "v_le_100":    sum(1 for v in vs if v <= 100),
            "min_v":       min(vs),
            "median_v":    sorted(vs)[len(vs) // 2],
        }
    return out


def stat_tests(rows):
    """Paired Wilcoxon between each pair of methods (seed-level, n=24)."""
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


def adjust_pvalues(p_list, method="bh"):
    """Adjust a list of p-values for multiple comparisons.

    method == 'bonferroni': p_adj_i = min(1, m * p_i)  (m = number of tests)
    method == 'bh': Benjamini-Hochberg FDR control.
    Returns list aligned with input.
    """
    m = len(p_list)
    if m == 0: return []
    if method == "bonferroni":
        return [min(1.0, p * m) for p in p_list]
    # BH: sort, scale, then take running-min from the largest end
    indexed = sorted(enumerate(p_list), key=lambda x: x[1])
    adj = [None] * m
    cur_min = 1.0
    # Walk from largest to smallest p
    for rank in range(m - 1, -1, -1):
        idx, p = indexed[rank]
        bh = p * m / (rank + 1)
        cur_min = min(cur_min, bh)
        adj[idx] = cur_min
    return adj


def emit_wilcoxon_table(stats, path, with_correction=True):
    """Format Wilcoxon paired-test table nicely.

    Accepts both seed-level (key 'n') and circuit-level (key 'n_circuits') formats.
    If with_correction=True, adds Bonferroni and BH-adjusted p-value columns.
    """
    # Detect which n field is present
    sample_v = next(iter(stats.values()))
    n_key = "n" if "n" in sample_v else "n_circuits"

    # Row ordering: vsr_post comparisons first, then vsr_intra
    def sort_key(k):
        head = k.split("_vs_")[0]
        order = {"vsr_post": 0, "vsr_intra": 1}.get(head, 2)
        return (order, k)

    keys = sorted(stats.keys(), key=sort_key)
    raw_pv = [stats[k]["dv"]["p"] for k in keys]
    raw_ph = [stats[k]["dh"]["p"] for k in keys]
    bh_pv = adjust_pvalues(raw_pv, "bh") if with_correction else None
    bh_ph = adjust_pvalues(raw_ph, "bh") if with_correction else None
    bf_pv = adjust_pvalues(raw_pv, "bonferroni") if with_correction else None
    bf_ph = adjust_pvalues(raw_ph, "bonferroni") if with_correction else None

    def fmt(p, sig=True):
        if sig and p < 1e-3:
            return r"$<\!10^{-3}$\,$^{\star\star\star}$"
        if sig and p < 1e-2:
            return f"${p:.3f}$" + r"\,$^{\star\star}$"
        if sig and p < 5e-2:
            return f"${p:.3f}$" + r"\,$^{\star}$"
        return f"${p:.3f}$"

    if with_correction:
        lines = [
            r"\begin{tabular}{lr|ll|ll|ll}",
            r"\toprule",
            r" & & \multicolumn{2}{c|}{raw} & \multicolumn{2}{c|}{BH-adj.} & \multicolumn{2}{c}{Bonferroni} \\",
            r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
            r"Comparison & $n$ & $p_{\Delta v}$ & $p_{\Delta h}$ & $p_{\Delta v}$ & $p_{\Delta h}$ & $p_{\Delta v}$ & $p_{\Delta h}$ \\",
            r"\midrule",
        ]
        for i, k in enumerate(keys):
            v = stats[k]
            a, b = k.split("_vs_")
            nice = a.replace("_", "-") + r" vs.\ " + b.replace("_", "-")
            n_val = v.get(n_key, v.get("n", v.get("n_circuits", 0)))
            lines.append(
                f"{nice} & {n_val} & "
                f"{fmt(raw_pv[i])} & {fmt(raw_ph[i])} & "
                f"{fmt(bh_pv[i])} & {fmt(bh_ph[i])} & "
                f"{fmt(bf_pv[i], sig=False)} & {fmt(bf_ph[i], sig=False)} \\\\"
            )
    else:
        lines = [
            r"\begin{tabular}{lrll}",
            r"\toprule",
            r"Comparison & $n$ & $p_{\Delta v}$ & $p_{\Delta h}$ \\",
            r"\midrule",
        ]
        for i, k in enumerate(keys):
            v = stats[k]
            a, b = k.split("_vs_")
            nice = a.replace("_", "-") + r" vs.\ " + b.replace("_", "-")
            n_val = v.get(n_key, v.get("n", v.get("n_circuits", 0)))
            lines.append(f"{nice} & {n_val} & {fmt(raw_pv[i])} & {fmt(raw_ph[i])} \\\\")

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

    stats_circ = stat_tests_circuit_level(rows)
    legality = fully_legal_stats(rows)
    out = {
        "n_rows": len(rows),
        "per_circuit": summary,
        "cross_circuit_medians": medians,
        "strict_pareto": strict,
        "wilcoxon_seed_level": stats,        # n=24 seed-level (paired bootstrap)
        "wilcoxon_circuit_level": stats_circ, # n=6 circuit-level (conservative)
        "legality": legality,
    }
    out_path = RESULTS / "neurips_stats.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {out_path}")

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    emit_main_table(summary, PAPER_FIG / "table_main_neurips.tex")
    emit_full_table(summary, PAPER_FIG / "table_main_full.tex")
    emit_wilcoxon_table(stats, PAPER_FIG / "table_wilcoxon.tex")
    emit_wilcoxon_table(stat_tests_circuit_level(rows),
                        PAPER_FIG / "table_wilcoxon_circuit.tex")
    print(f"Wrote {PAPER_FIG/'table_main_neurips.tex'} (slim)")
    print(f"Wrote {PAPER_FIG/'table_main_full.tex'} (6-method)")
    print(f"Wrote {PAPER_FIG/'table_wilcoxon.tex'} (seed-level n=24)")
    print(f"Wrote {PAPER_FIG/'table_wilcoxon_circuit.tex'} (circuit-level n=6)")

    # ---- print headline ----
    print("\n=== Cross-circuit medians ===")
    for m, s in medians.items():
        if s.get("dv_median") is not None:
            print(f"  {m:<14} median Δv={s['dv_median']:+.1f}%  median Δh={s['dh_median']:+.1f}%  n={s['n_circuits']} circuits")
    print("\n=== Strict Pareto improvement (Δv<0 AND Δh<0) ===")
    for m, s in strict.items():
        print(f"  {m:<14} {s['strict']}/{s['n']} (circuit, seed) pairs strictly improved")
    print("\n=== Wilcoxon (seed-level, n=24, treats seeds as paired bootstrap) ===")
    for k, v in stats.items():
        sv = "***" if v["dv"]["p"] < 0.001 else ("**" if v["dv"]["p"] < 0.01 else ("*" if v["dv"]["p"] < 0.05 else ""))
        sh = "***" if v["dh"]["p"] < 0.001 else ("**" if v["dh"]["p"] < 0.01 else ("*" if v["dh"]["p"] < 0.05 else ""))
        print(f"  {k:<32}  n={v['n']}  Δv: p={v['dv']['p']:.4g}{sv}  Δh: p={v['dh']['p']:.4g}{sh}")

    print("\n=== Wilcoxon (circuit-level, n=6, conservative) ===")
    for k, v in stats_circ.items():
        sv = "***" if v["dv"]["p"] < 0.001 else ("**" if v["dv"]["p"] < 0.01 else ("*" if v["dv"]["p"] < 0.05 else ""))
        sh = "***" if v["dh"]["p"] < 0.001 else ("**" if v["dh"]["p"] < 0.01 else ("*" if v["dh"]["p"] < 0.05 else ""))
        print(f"  {k:<32}  n={v['n_circuits']}  Δv: p={v['dv']['p']:.4g}{sv}  Δh: p={v['dh']['p']:.4g}{sh}")

    print("\n=== Legality (fraction of trials achieving v=0) ===")
    for m, lg in legality.items():
        print(f"  {m:<14} fully_legal={lg['fully_legal']}/{lg['n']}  v<=5: {lg['v_le_5']}  median_v={lg['median_v']}  min_v={lg['min_v']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
