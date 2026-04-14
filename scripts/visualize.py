#!/usr/bin/env python3
"""Placement visualization utilities.

Can be used standalone or imported by other scripts.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def plot_placement(
    centers: np.ndarray,
    sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    severity: np.ndarray | None = None,
    title: str = "Macro Placement",
    save_path: str | None = None,
    show_labels: bool = True,
):
    """Visualize a macro placement on the chip canvas.

    Args:
        centers: (N, 2) array of macro center coordinates.
        sizes: (N, 2) array of macro (width, height).
        canvas_width: Width of the chip canvas.
        canvas_height: Height of the chip canvas.
        severity: Optional (N,) array of violation severity per macro.
            Used to color macros (green=legal, red=violation).
        title: Plot title.
        save_path: If provided, save figure to this path.
        show_labels: Whether to show macro index labels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw canvas
    canvas = patches.Rectangle(
        (0, 0), canvas_width, canvas_height,
        linewidth=2, edgecolor="black", facecolor="lightyellow",
    )
    ax.add_patch(canvas)

    # Color map based on severity
    if severity is not None:
        max_sev = severity.max() if severity.max() > 0 else 1.0
        norm_sev = severity / max_sev
    else:
        norm_sev = np.zeros(len(centers))

    for i in range(len(centers)):
        x, y = centers[i]
        w, h = sizes[i]

        # Bottom-left corner
        bl_x = x - w / 2
        bl_y = y - h / 2

        # Color: green (legal) to red (violation)
        if norm_sev[i] > 0:
            color = plt.cm.RdYlGn(1.0 - norm_sev[i])
        else:
            color = "lightgreen"

        alpha = 0.6
        rect = patches.Rectangle(
            (bl_x, bl_y), w, h,
            linewidth=1.5, edgecolor="darkblue", facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)

        if show_labels:
            ax.text(x, y, str(i), ha="center", va="center", fontsize=8, fontweight="bold")

    # Set axis
    margin = max(canvas_width, canvas_height) * 0.05
    ax.set_xlim(-margin, canvas_width + margin)
    ax.set_ylim(-margin, canvas_height + margin)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_violation_trajectory(
    violation_counts: list[int],
    title: str = "Violation Count vs. Repair Iteration",
    save_path: str | None = None,
):
    """Plot violation count over repair iterations.

    Args:
        violation_counts: List of total violation counts per iteration.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    iterations = list(range(len(violation_counts)))
    ax.plot(iterations, violation_counts, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xlabel("Repair Iteration")
    ax.set_ylabel("Total Violations")
    ax.set_title(title)
    ax.set_xticks(iterations)
    ax.grid(True, alpha=0.3)

    # Annotate values
    for i, v in enumerate(violation_counts):
        ax.annotate(str(v), (i, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def demo_visualization():
    """Generate a demo visualization with synthetic data."""
    np.random.seed(42)

    n = 8
    canvas_w, canvas_h = 10.0, 10.0
    centers = np.random.uniform(1, 9, size=(n, 2))
    sizes = np.random.uniform(0.8, 2.5, size=(n, 2))

    # Add some violations
    centers[0] = [0.5, 5.0]  # boundary violation
    centers[1] = [centers[2][0] + 0.3, centers[2][1]]  # overlap

    severity = np.array([0.5, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])

    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_placement(
        centers, sizes, canvas_w, canvas_h,
        severity=severity,
        title="Demo: Macro Placement with Violations",
        save_path=str(output_dir / "demo_placement.png"),
    )

    plot_violation_trajectory(
        [12, 7, 3, 1, 0],
        title="Demo: VSR Repair Convergence",
        save_path=str(output_dir / "demo_convergence.png"),
    )


if __name__ == "__main__":
    demo_visualization()
