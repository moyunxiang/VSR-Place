#!/usr/bin/env python3
"""Aggregate experiment results into tables.

Reads results from results/runs/ and produces CSV and LaTeX tables.
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def collect_results(runs_dir: Path) -> pd.DataFrame:
    """Collect all experiment results into a DataFrame.

    Args:
        runs_dir: Path to results/runs/ directory.

    Returns:
        DataFrame with one row per experiment run.
    """
    rows = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.yaml"

        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        row = {
            "run_id": run_dir.name,
            **metrics,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std across seeds for each method.

    Args:
        df: Raw results DataFrame.

    Returns:
        Summary DataFrame with mean ± std.
    """
    if "method" not in df.columns:
        return df

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "seed"]

    summary = df.groupby("method")[numeric_cols].agg(["mean", "std"])
    return summary


def to_latex(df: pd.DataFrame, caption: str = "Results") -> str:
    """Convert DataFrame to LaTeX table string."""
    return df.to_latex(
        caption=caption,
        float_format="%.3f",
        bold_rows=True,
    )


def main():
    runs_dir = PROJECT_ROOT / "results" / "runs"
    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"No results directory found at {runs_dir}")
        print("Run experiments first with scripts/run_vsr.py")
        return

    df = collect_results(runs_dir)

    if df.empty:
        print("No results found.")
        return

    # Save raw results
    df.to_csv(tables_dir / "raw_results.csv", index=False)
    print(f"Raw results: {tables_dir / 'raw_results.csv'}")

    # Summary if method column exists
    if "method" in df.columns:
        summary = summarize_by_method(df)
        summary.to_csv(tables_dir / "summary.csv")
        print(f"Summary: {tables_dir / 'summary.csv'}")

        latex = to_latex(summary)
        with open(tables_dir / "summary.tex", "w") as f:
            f.write(latex)
        print(f"LaTeX: {tables_dir / 'summary.tex'}")

    print(f"\nTotal runs: {len(df)}")
    print(df.describe())


if __name__ == "__main__":
    main()
