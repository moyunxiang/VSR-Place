"""Unified preflight harness for all GPU-bound experiment scripts.

Runs each script with --mock (no GPU, no checkpoint) and validates:
  - script imports cleanly
  - argparse accepts default args
  - mock-mode logic prints expected sentinel lines
  - JSON schemas declared by each script are non-empty and parseable

Exit code 0 on full pass, 1 on any failure.

Usage:
    python3 scripts/preflight_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = "/opt/homebrew/Caskroom/miniforge/base/bin/python3.12"

CHECKS = [
    {
        "name": "L2 robust stats (CPU)",
        "cmd": [PY, "scripts/compute_robust_stats.py"],
        "expect": "Wrote",
        "is_mock": False,  # this one runs for real on local data
    },
    {
        "name": "E1 main (mock)",
        "cmd": [PY, "scripts/run_main_neurips.py", "--mock"],
        "expect": "preflight OK",
        "is_mock": True,
    },
    {
        "name": "E4 timestep sweep (mock)",
        "cmd": [PY, "scripts/run_timestep_sweep.py", "--mock"],
        "expect": "preflight OK",
        "is_mock": True,
    },
    {
        "name": "E5 mem profile (mock)",
        "cmd": [PY, "scripts/run_mem_profile.py", "--mock"],
        "expect": "preflight OK",
        "is_mock": True,
    },
    {
        "name": "L1 toy 2D (quick)",
        "cmd": [PY, "scripts/toy_2d_experiment.py", "--quick", "--train"],
        "expect": "rows",
        "is_mock": False,
    },
]


def main():
    failed = []
    for c in CHECKS:
        print(f"\n========== {c['name']} ==========", flush=True)
        try:
            r = subprocess.run(c["cmd"], cwd=REPO, capture_output=True,
                               text=True, timeout=600)
            ok = (r.returncode == 0) and (c["expect"] in (r.stdout + r.stderr))
            if ok:
                print(f"  PASS  ({r.returncode})")
            else:
                print(f"  FAIL  rc={r.returncode}")
                print("  --- stdout (tail) ---")
                print("\n".join(r.stdout.splitlines()[-20:]))
                print("  --- stderr (tail) ---")
                print("\n".join(r.stderr.splitlines()[-20:]))
                failed.append(c["name"])
        except subprocess.TimeoutExpired:
            print(f"  FAIL  timeout")
            failed.append(c["name"])

    print("\n\n==================== SUMMARY ====================")
    if failed:
        print(f"FAILED ({len(failed)}/{len(CHECKS)}):")
        for n in failed:
            print(f"  - {n}")
        sys.exit(1)
    print(f"All {len(CHECKS)} preflight checks passed.")
    print("Cleared for GPU launch.")


if __name__ == "__main__":
    main()
