#!/usr/bin/env python3
"""Analyze critical_eval.json output and print paper-ready tables."""
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "results/ispd2005/critical_eval.json"
if len(sys.argv) > 1:
    path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Missing {path}")

d = json.load(open(path))
by = defaultdict(list)
for r in d:
    if "error" in r: continue
    by[r["circuit"]].append(r)

def agg(rs, k):
    vals = [r[k] for r in rs if r.get(k) is not None]
    if not vals: return float("nan"), 0
    return np.mean(vals), np.std(vals)

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]

print("="*110)
print("HPWL (rescaled, units as reported by CD utils.hpwl_fast)")
print("="*110)
print(f"{'Circuit':<10} {'baseline':<15} {'CD-opt':<15} {'VSR':<15} {'INTRA':<15} {'INTRA-only':<15}")
for c in CIRCUITS:
    rs = by.get(c, [])
    if not rs: continue
    bh, _ = agg(rs, "base_hpwl_rescaled")
    ch, _ = agg(rs, "cd_hpwl_rescaled")
    vh, _ = agg(rs, "vsr_hpwl_rescaled")
    ih, _ = agg(rs, "intra_hpwl_rescaled")
    i0h, _ = agg(rs, "intra_only_hpwl_rescaled")
    print(f"{c:<10} {bh:<15.3e} {ch:<15.3e} {vh:<15.3e} {ih:<15.3e} {i0h:<15.3e}")

print()
print("="*110)
print("Legality score (CD check_legality_new, 0-1, higher is better)")
print("="*110)
print(f"{'Circuit':<10} {'baseline':<12} {'CD-opt':<12} {'VSR':<12} {'INTRA':<12} {'INTRA-only':<12}")
for c in CIRCUITS:
    rs = by.get(c, [])
    if not rs: continue
    bl, _ = agg(rs, "base_legality")
    cl, _ = agg(rs, "cd_legality")
    vl, _ = agg(rs, "vsr_legality")
    il, _ = agg(rs, "intra_legality")
    i0l, _ = agg(rs, "intra_only_legality")
    print(f"{c:<10} {bl:<12.4f} {cl:<12.4f} {vl:<12.4f} {il:<12.4f} {i0l:<12.4f}")

print()
print("="*110)
print("Violations (our verifier, count)")
print("="*110)
print(f"{'Circuit':<10} {'baseline':<12} {'CD-opt':<12} {'VSR':<12} {'INTRA':<12}")
for c in CIRCUITS:
    rs = by.get(c, [])
    if not rs: continue
    bv, _ = agg(rs, "base_v"); cv, _ = agg(rs, "cd_v")
    vv, _ = agg(rs, "vsr_v"); iv, _ = agg(rs, "intra_v")
    print(f"{c:<10} {bv:<12.0f} {cv:<12.0f} {vv:<12.0f} {iv:<12.0f}")

print()
print("="*110)
print("Time (seconds)")
print("="*110)
print(f"{'Circuit':<10} {'sample':<10} {'CD-opt':<10} {'VSR':<10} {'INTRA':<10}")
for c in CIRCUITS:
    rs = by.get(c, [])
    if not rs: continue
    ts, _ = agg(rs, "t_sample"); tc, _ = agg(rs, "cd_time")
    tv, _ = agg(rs, "vsr_time"); ti, _ = agg(rs, "intra_time")
    print(f"{c:<10} {ts:<10.1f} {tc:<10.1f} {tv:<10.1f} {ti:<10.1f}")

# Mean summary
print()
print("="*80)
print("MEAN across 6 circuits (3 seeds each)")
print("="*80)
method_keys = {
    "baseline": ("base_hpwl_rescaled", "base_legality", None),
    "CD-opt":   ("cd_hpwl_rescaled",   "cd_legality",   "cd_time"),
    "VSR":      ("vsr_hpwl_rescaled",  "vsr_legality",  "vsr_time"),
    "INTRA":    ("intra_hpwl_rescaled","intra_legality","intra_time"),
    "INTRA-0":  ("intra_only_hpwl_rescaled","intra_only_legality","intra_only_time"),
}
all_rs = [r for rs in by.values() for r in rs]
print(f"{'Method':<12} {'HPWL':<15} {'Legality':<12} {'Time':<10}")
for name, (hk, lk, tk) in method_keys.items():
    h, _ = agg(all_rs, hk)
    l, _ = agg(all_rs, lk)
    t = agg(all_rs, tk)[0] if tk else 0.0
    print(f"{name:<12} {h:<15.3e} {l:<12.4f} {t:<10.1f}")
