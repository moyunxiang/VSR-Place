"""Recompute paper headline numbers with NeurIPS-main statistical rigor.

Reads existing per-(circuit, seed) JSON results and produces:
  - per-circuit mean +/- std
  - cross-circuit median (robust to small-edge outliers)
  - 95% bootstrap CI on mean
  - paired Wilcoxon signed-rank test between method pairs
  - strict-Pareto count

Output:
  results/ispd2005/robust_stats.json
  paper/figures/table1_robust.tex     (per-circuit table)
  paper/figures/table_stat_tests.tex  (stat-test table)

Pure local, no GPU, no torch.
"""
from __future__ import annotations

import json
import math
import statistics as S
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "ispd2005"
PAPER_FIG = REPO / "paper" / "figures"


# ---------- bootstrap + Wilcoxon (no scipy dependency) ----------

def bootstrap_ci(values: list[float], n_resample: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the mean."""
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


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float, float]:
    """Two-sided paired Wilcoxon signed-rank test (no scipy).

    Returns (W, approx p-value via normal approximation).
    For n < 6 the normal approximation is unreliable; we still return it.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n == 0:
        return (0.0, 1.0)
    abs_diffs = sorted([(abs(d), 1 if d > 0 else -1) for d in diffs])
    # Compute average ranks for ties on |d|.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2  # ranks are 1-based, so (i+1 + j+1)/2
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    W_pos = sum(r for r, (_, sgn) in zip(ranks, abs_diffs) if sgn > 0)
    W_neg = sum(r for r, (_, sgn) in zip(ranks, abs_diffs) if sgn < 0)
    W = min(W_pos, W_neg)
    # Normal approximation
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return (W, 1.0)
    z = (W - mu) / sigma
    # Two-sided p
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (W, p)


# ---------- main result aggregation ----------

def load_pareto():
    """Load Pareto sweep at lambda=2 (3 seeds, 6 circuits)."""
    rows = json.load(open(RESULTS_DIR / "pareto_3seed_6w.json"))
    return [r for r in rows if r["hpwl_weight"] == 2.0]


def load_cd_compare():
    rows = json.load(open(RESULTS_DIR / "cd_compare_3seed.json"))
    return [r for r in rows if "vsr_v" in r and r["vsr_v"] is not None]


def load_cd_scheduled():
    rows = json.load(open(RESULTS_DIR / "cd_scheduled.json"))
    return rows


def load_extra():
    rows = json.load(open(RESULTS_DIR / "extra_experiments.json"))
    return [r for r in rows if "intra_v" in r and r["intra_v"] is not None]


def per_circuit_summary():
    """Build per-(circuit) aggregated record across seeds."""
    cd = load_cd_compare()
    sched = load_cd_scheduled()
    extra = load_extra()

    # group by circuit
    cd_g = defaultdict(list)
    for r in cd:
        cd_g[r["circuit"]].append(r)
    sched_g = defaultdict(list)
    for r in sched:
        sched_g[r["circuit"]].append(r)
    extra_g = defaultdict(list)
    for r in extra:
        extra_g[r["circuit"]].append(r)

    summary = {}
    for c in sorted(cd_g.keys()):
        rs = cd_g[c]
        # baseline (mean) for reference
        bv = S.mean(r["baseline_v"] for r in rs)
        bh = S.mean(r["baseline_h"] for r in rs)

        # VSR-post (lambda=2) per-seed deltas (percent)
        vsr_dv = [(r["vsr_v"] - r["baseline_v"]) / r["baseline_v"] * 100 for r in rs]
        vsr_dh = [(r["vsr_h"] - r["baseline_h"]) / max(r["baseline_h"], 1e-9) * 100 for r in rs]
        vsr_t = [r["vsr_time"] for r in rs]

        # CD-std
        cd_dv = [(r["cd_v"] - r["baseline_v"]) / r["baseline_v"] * 100 for r in rs]
        cd_dh = [(r["cd_h"] - r["baseline_h"]) / max(r["baseline_h"], 1e-9) * 100 for r in rs]
        cd_t = [r["cd_time"] for r in rs]

        # CD-sched
        sr = sched_g.get(c, [])
        sched_dv = [(r["cd_sched_v"] - r["baseline_v"]) / r["baseline_v"] * 100 for r in sr]
        sched_dh = [(r["cd_sched_h"] - r["baseline_h"]) / max(r["baseline_h"], 1e-9) * 100 for r in sr]
        sched_t = [r["cd_sched_time"] for r in sr]

        # Intra-sampling (extra_experiments.json, 3 seeds)
        er = extra_g.get(c, [])
        intra_dv = [(r["intra_v"] - r["baseline_v"]) / r["baseline_v"] * 100 for r in er]
        intra_dh = [(r["intra_h"] - r["baseline_h"]) / max(r["baseline_h"], 1e-9) * 100 for r in er]
        intra_t = [r["intra_time"] for r in er]

        summary[c] = {
            "n_macros": rs[0]["n_macros"],
            "n_seeds_cd": len(rs),
            "n_seeds_sched": len(sr),
            "n_seeds_intra": len(er),
            "baseline": {"v": bv, "h": bh},
            "vsr_post": _summarize(vsr_dv, vsr_dh, vsr_t),
            "cd_std": _summarize(cd_dv, cd_dh, cd_t),
            "cd_sched": _summarize(sched_dv, sched_dh, sched_t),
            "vsr_intra": _summarize(intra_dv, intra_dh, intra_t),
        }
    return summary


def _summarize(dv: list[float], dh: list[float], t: list[float]):
    if not dv:
        return {"n": 0}
    mean_dv, lo_dv, hi_dv = bootstrap_ci(dv)
    mean_dh, lo_dh, hi_dh = bootstrap_ci(dh)
    return {
        "n": len(dv),
        "dv_pct_mean": mean_dv,
        "dv_pct_std": S.pstdev(dv) if len(dv) > 1 else 0.0,
        "dv_pct_ci95": [lo_dv, hi_dv],
        "dh_pct_mean": mean_dh,
        "dh_pct_std": S.pstdev(dh) if len(dh) > 1 else 0.0,
        "dh_pct_ci95": [lo_dh, hi_dh],
        "time_mean": S.mean(t),
        "time_std": S.pstdev(t) if len(t) > 1 else 0.0,
    }


def cross_circuit_medians(summary: dict):
    """Robust headline numbers using median-across-circuits."""
    methods = ["vsr_post", "cd_std", "cd_sched", "vsr_intra"]
    out = {}
    for m in methods:
        dv_means = [v[m]["dv_pct_mean"] for v in summary.values() if v[m].get("n", 0) > 0]
        dh_means = [v[m]["dh_pct_mean"] for v in summary.values() if v[m].get("n", 0) > 0]
        out[m] = {
            "n_circuits": len(dv_means),
            "dv_median": S.median(dv_means) if dv_means else None,
            "dv_mean": S.mean(dv_means) if dv_means else None,
            "dh_median": S.median(dh_means) if dh_means else None,
            "dh_mean": S.mean(dh_means) if dh_means else None,
        }
    return out


def strict_pareto_counts():
    """Per-method count of (circuit, seed) pairs that are strictly Pareto-improved."""
    cd = load_cd_compare()
    sched = load_cd_scheduled()
    extra = load_extra()

    out = {}
    n = len(cd)
    out["vsr_post"] = {
        "strict": sum(1 for r in cd if r["vsr_v"] < r["baseline_v"] and r["vsr_h"] < r["baseline_h"]),
        "n": n,
    }
    out["cd_std"] = {
        "strict": sum(1 for r in cd if r["cd_v"] is not None and r["cd_v"] < r["baseline_v"] and r["cd_h"] < r["baseline_h"]),
        "n": n,
    }
    n_s = len(sched)
    out["cd_sched"] = {
        "strict": sum(1 for r in sched if r["cd_sched_v"] < r["baseline_v"] and r["cd_sched_h"] < r["baseline_h"]),
        "n": n_s,
    }
    n_e = len(extra)
    out["vsr_intra"] = {
        "strict": sum(1 for r in extra if r["intra_v"] < r["baseline_v"] and r["intra_h"] < r["baseline_h"]),
        "n": n_e,
    }
    return out


def stat_tests():
    """Paired Wilcoxon: VSR-post vs each baseline, on (Δviolation%, Δhpwl%) per (circuit, seed)."""
    cd = load_cd_compare()
    sched = load_cd_scheduled()
    extra = load_extra()

    # Index by (circuit, seed)
    cd_idx = {(r["circuit"], r["seed"]): r for r in cd}
    sched_idx = {(r["circuit"], r["seed"]): r for r in sched}
    extra_idx = {(r["circuit"], r["seed"]): r for r in extra}

    def pct_v(r, key_v):
        return (r[key_v] - r["baseline_v"]) / r["baseline_v"] * 100

    def pct_h(r, key_h):
        return (r[key_h] - r["baseline_h"]) / max(r["baseline_h"], 1e-9) * 100

    out = {}
    # VSR-post vs CD-std on (circuit, seed) pairs that exist in both
    common_cd = sorted(set(cd_idx.keys()) & set(cd_idx.keys()))  # cd table has both
    vsr_dv = [pct_v(cd_idx[k], "vsr_v") for k in common_cd]
    cdstd_dv = [pct_v(cd_idx[k], "cd_v") for k in common_cd if cd_idx[k]["cd_v"] is not None]
    vsr_dv_aligned = [pct_v(cd_idx[k], "vsr_v") for k in common_cd if cd_idx[k]["cd_v"] is not None]
    if vsr_dv_aligned and cdstd_dv:
        W, p = wilcoxon_signed_rank(vsr_dv_aligned, cdstd_dv)
        out["vsr_post_vs_cd_std_dv"] = {"W": W, "p": p, "n": len(vsr_dv_aligned)}
        vsr_dh = [pct_h(cd_idx[k], "vsr_h") for k in common_cd if cd_idx[k]["cd_v"] is not None]
        cdstd_dh = [pct_h(cd_idx[k], "cd_h") for k in common_cd if cd_idx[k]["cd_v"] is not None]
        W, p = wilcoxon_signed_rank(vsr_dh, cdstd_dh)
        out["vsr_post_vs_cd_std_dh"] = {"W": W, "p": p, "n": len(vsr_dh)}

    # VSR-post vs CD-sched
    common_sched = sorted(set(cd_idx.keys()) & set(sched_idx.keys()))
    if common_sched:
        a_dv = [pct_v(cd_idx[k], "vsr_v") for k in common_sched]
        b_dv = [pct_v(sched_idx[k], "cd_sched_v") for k in common_sched]
        W, p = wilcoxon_signed_rank(a_dv, b_dv)
        out["vsr_post_vs_cd_sched_dv"] = {"W": W, "p": p, "n": len(a_dv)}
        a_dh = [pct_h(cd_idx[k], "vsr_h") for k in common_sched]
        b_dh = [pct_h(sched_idx[k], "cd_sched_h") for k in common_sched]
        W, p = wilcoxon_signed_rank(a_dh, b_dh)
        out["vsr_post_vs_cd_sched_dh"] = {"W": W, "p": p, "n": len(a_dh)}

    # VSR-intra vs VSR-post
    common_intra = sorted(set(cd_idx.keys()) & set(extra_idx.keys()))
    if common_intra:
        a_dv = [pct_v(cd_idx[k], "vsr_v") for k in common_intra]
        b_dv = [pct_v(extra_idx[k], "intra_v") for k in common_intra]
        W, p = wilcoxon_signed_rank(a_dv, b_dv)
        out["vsr_post_vs_vsr_intra_dv"] = {"W": W, "p": p, "n": len(a_dv)}
        a_dh = [pct_h(cd_idx[k], "vsr_h") for k in common_intra]
        b_dh = [pct_h(extra_idx[k], "intra_h") for k in common_intra]
        W, p = wilcoxon_signed_rank(a_dh, b_dh)
        out["vsr_post_vs_vsr_intra_dh"] = {"W": W, "p": p, "n": len(a_dh)}
    return out


# ---------- LaTeX emitters ----------

def emit_per_circuit_table(summary: dict, path: Path):
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{VSR-post ($\lambda=2$)} & \multicolumn{2}{c}{VSR-intra} & \multicolumn{2}{c}{CD-standard} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"Circuit & $N$ & $\Delta v\%$ & $\Delta h\%$ & $\Delta v\%$ & $\Delta h\%$ & $\Delta v\%$ & $\Delta h\%$ \\",
        r"\midrule",
    ]
    for c, v in sorted(summary.items()):
        n = v["n_macros"]
        vp = v["vsr_post"]; vi = v["vsr_intra"]; cd = v["cd_std"]

        def fmt(stat, k_mean, k_std):
            if stat.get("n", 0) == 0:
                return r"\textemdash"
            return f"${stat[k_mean]:+.1f}{{\\scriptscriptstyle\\pm{stat[k_std]:.1f}}}$"

        lines.append(
            f"{c} & {n} & {fmt(vp, 'dv_pct_mean', 'dv_pct_std')} & {fmt(vp, 'dh_pct_mean', 'dh_pct_std')}"
            f" & {fmt(vi, 'dv_pct_mean', 'dv_pct_std')} & {fmt(vi, 'dh_pct_mean', 'dh_pct_std')}"
            f" & {fmt(cd, 'dv_pct_mean', 'dv_pct_std')} & {fmt(cd, 'dh_pct_mean', 'dh_pct_std')} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")


def emit_stat_test_table(stat: dict, path: Path):
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Comparison & $W$ & $p$ \\",
        r"\midrule",
    ]
    label_map = {
        "vsr_post_vs_cd_std_dv": r"VSR-post vs CD-std ($\Delta v$)",
        "vsr_post_vs_cd_std_dh": r"VSR-post vs CD-std ($\Delta h$)",
        "vsr_post_vs_cd_sched_dv": r"VSR-post vs CD-sched ($\Delta v$)",
        "vsr_post_vs_cd_sched_dh": r"VSR-post vs CD-sched ($\Delta h$)",
        "vsr_post_vs_vsr_intra_dv": r"VSR-post vs VSR-intra ($\Delta v$)",
        "vsr_post_vs_vsr_intra_dh": r"VSR-post vs VSR-intra ($\Delta h$)",
    }
    for k, v in stat.items():
        label = label_map.get(k, k)
        lines.append(f"{label} & {v['W']:.1f} & {v['p']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    summary = per_circuit_summary()
    medians = cross_circuit_medians(summary)
    strict = strict_pareto_counts()
    stats = stat_tests()

    out_path = RESULTS_DIR / "robust_stats.json"
    out = {
        "per_circuit": summary,
        "cross_circuit_medians": medians,
        "strict_pareto": strict,
        "wilcoxon": stats,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {out_path}")

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    emit_per_circuit_table(summary, PAPER_FIG / "table1_robust.tex")
    emit_stat_test_table(stats, PAPER_FIG / "table_stat_tests.tex")
    print(f"Wrote {PAPER_FIG/'table1_robust.tex'}, {PAPER_FIG/'table_stat_tests.tex'}")

    # Print headlines
    print("\n=== Cross-circuit headline numbers ===")
    for m, s in medians.items():
        if s.get("dv_median") is not None:
            print(f"  {m:<10} median Δv={s['dv_median']:+.1f}%  median Δh={s['dh_median']:+.1f}%  (mean Δv={s['dv_mean']:+.1f} mean Δh={s['dh_mean']:+.1f}, n={s['n_circuits']} circuits)")

    print("\n=== Strict Pareto improvement ===")
    for m, s in strict.items():
        print(f"  {m:<10} {s['strict']}/{s['n']} (circuit, seed) pairs strictly improved")

    print("\n=== Wilcoxon paired tests ===")
    for k, v in stats.items():
        sig = " ***" if v["p"] < 0.001 else (" **" if v["p"] < 0.01 else (" *" if v["p"] < 0.05 else ""))
        print(f"  {k:<32}  W={v['W']:.1f}  p={v['p']:.4g}  n={v['n']}{sig}")


if __name__ == "__main__":
    main()
