"""Main VSR-Place closed-loop controller.

Orchestrates the generate -> verify -> select -> renoise -> redenoise cycle.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor

from vsr_place.loop.budget import RepairBudget
from vsr_place.renoising.renoise import selective_renoise
from vsr_place.renoising.selector import MacroSelector
from vsr_place.renoising.strength import StrengthSchedule
from vsr_place.verifier.feedback import ViolationFeedback
from vsr_place.verifier.verifier import Verifier


class DiffusionBackend(Protocol):
    """Protocol defining what the VSR loop needs from a diffusion backbone."""

    def sample(self, cond: Any, **kwargs: Any) -> Tensor:
        """Generate a placement from noise."""
        ...

    def denoise_from(
        self,
        x_start: Tensor,
        cond: Any,
        start_timestep: float,
        num_steps: int,
        **kwargs: Any,
    ) -> Tensor:
        """Resume denoising from an intermediate noisy state."""
        ...

    def decode_placement(self, x_normalized: Tensor, cond: Any) -> tuple[Tensor, Tensor]:
        """Convert normalized coords to absolute (centers, sizes)."""
        ...

    def encode_placement(self, centers: Tensor, cond: Any) -> Tensor:
        """Convert absolute coords back to normalized space."""
        ...


@dataclass
class VSRResult:
    """Result of a VSR-Place run.

    Attributes:
        placement: Final placement tensor (normalized coordinates).
        feedback_history: List of ViolationFeedback from each repair iteration.
        metrics: Aggregated metrics dict.
        intermediate_placements: Optional list of intermediate placements.
    """

    placement: Tensor
    feedback_history: list[ViolationFeedback] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    intermediate_placements: list[Tensor] = field(default_factory=list)


class VSRLoop:
    """Verifier-Guided Selective Re-noising loop controller.

    Implements the closed-loop refinement procedure:
    1. Generate initial placement via diffusion backbone
    2. Decode and verify
    3. If violations: select offending macros, compute re-noising strength,
       selectively re-noise, then re-denoise
    4. Repeat until legal or budget exhausted

    Args:
        backend: Diffusion model backend (e.g., ChipDiffusionAdapter).
        verifier: Legality verifier.
        selector: Macro selection strategy.
        strength: Re-noising strength schedule.
        budget: Repair budget configuration.
        denoise_steps: Number of denoising steps for repair re-denoising.
        save_intermediates: Whether to save intermediate placements.
    """

    def __init__(
        self,
        backend: DiffusionBackend,
        verifier: Verifier,
        selector: MacroSelector,
        strength: StrengthSchedule,
        budget: RepairBudget | None = None,
        denoise_steps: int = 100,
        save_intermediates: bool = False,
    ):
        self.backend = backend
        self.verifier = verifier
        self.selector = selector
        self.strength = strength
        self.budget = budget or RepairBudget()
        self.denoise_steps = denoise_steps
        self.save_intermediates = save_intermediates

    def run(self, cond: Any, **sample_kwargs: Any) -> VSRResult:
        """Run the full VSR-Place closed-loop refinement.

        Args:
            cond: Conditioning information for the diffusion model
                (e.g., netlist graph, macro features).
            **sample_kwargs: Additional arguments for the initial sampling.

        Returns:
            VSRResult with final placement and diagnostics.
        """
        start_time = time.time()

        # Step 1: Initial generation
        x = self.backend.sample(cond, **sample_kwargs)

        feedback_history: list[ViolationFeedback] = []
        intermediates: list[Tensor] = []
        loop_count = 0

        while True:
            # Step 2: Decode and verify
            centers, sizes = self.backend.decode_placement(x, cond)
            feedback = self.verifier(centers, sizes)
            feedback_history.append(feedback)

            if self.save_intermediates:
                intermediates.append(x.clone())

            # Step 3: Check termination
            if self.budget.is_exhausted(loop_count, feedback_history):
                break

            if feedback.is_legal():
                break

            # Step 4: Select offending macros
            mask = self.selector.select(feedback)

            if not mask.any():
                # No macros selected (shouldn't happen if not legal, but be safe)
                break

            # Step 5: Compute re-noising strength
            alpha = self.strength.compute_alpha(feedback, mask, loop_iter=loop_count)

            # Step 6: Selective re-noising
            x_hat_0 = x  # Current placement is the clean estimate
            x_renoised = selective_renoise(x_hat_0, mask, alpha)

            # Step 7: Determine restart timestep from max alpha among selected macros
            max_alpha = alpha[mask].max().item()
            start_timestep = max_alpha  # Will be mapped to diffusion schedule by backend

            # Step 8: Re-denoise
            x = self.backend.denoise_from(
                x_renoised, cond,
                start_timestep=start_timestep,
                num_steps=self.denoise_steps,
            )

            loop_count += 1

        elapsed = time.time() - start_time

        # Compile metrics
        final_feedback = feedback_history[-1]
        metrics = {
            "elapsed_time": elapsed,
            "num_repair_loops": loop_count,
            "num_verifier_calls": len(feedback_history),
            **final_feedback.global_stats,
            "violation_history": [
                fb.global_stats["total_violations"] for fb in feedback_history
            ],
        }

        return VSRResult(
            placement=x,
            feedback_history=feedback_history,
            metrics=metrics,
            intermediate_placements=intermediates,
        )
