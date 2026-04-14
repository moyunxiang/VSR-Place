"""Modified sampling loop with verifier hooks for VSR-Place.

Provides intra-sampling verification: the verifier is called at configurable
points during the denoising process, and selective re-noising is applied
when violations are detected.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from vsr_place.backbone.adapter import ChipDiffusionAdapter
from vsr_place.conditioning.mask_only import MaskOnlyRepair
from vsr_place.verifier.feedback import ViolationFeedback
from vsr_place.verifier.verifier import Verifier


@dataclass
class VerifySchedule:
    """Controls when verification happens during denoising.

    Args:
        mode: 'every_k' (every k steps), 'late_only' (last N steps),
              'timestep_range' (between t_start and t_end).
        every_k: Verify every k denoising steps (for mode='every_k').
        late_steps: Number of final steps to verify (for mode='late_only').
        t_start: Start verifying when t drops below this (for mode='timestep_range').
        t_end: Stop verifying when t drops below this.
    """

    mode: str = "every_k"
    every_k: int = 10
    late_steps: int = 50
    t_start: float = 0.5
    t_end: float = 0.01

    def should_verify(self, step_idx: int, total_steps: int, t: float) -> bool:
        if self.mode == "every_k":
            return step_idx % self.every_k == 0
        elif self.mode == "late_only":
            return step_idx >= (total_steps - self.late_steps)
        elif self.mode == "timestep_range":
            return self.t_end <= t <= self.t_start
        return False


def vsr_guided_sampling(
    adapter: ChipDiffusionAdapter,
    cond: Any,
    verifier: Verifier,
    repair: MaskOnlyRepair,
    num_samples: int = 1,
    num_steps: int = 100,
    verify_schedule: VerifySchedule | None = None,
    max_repairs_per_step: int = 1,
) -> tuple[Tensor, list[ViolationFeedback]]:
    """Run diffusion sampling with intra-sampling verifier-guided repair.

    This modifies the standard denoising loop to:
    1. At configurable points, predict x_0 and verify it
    2. If violations found, selectively re-noise offending macros
    3. Continue denoising from the re-noised state

    Args:
        adapter: ChipDiffusion backbone adapter.
        cond: PyG Data object with netlist graph.
        verifier: Legality verifier.
        repair: MaskOnlyRepair instance (Variant A).
        num_samples: Number of placement samples.
        num_steps: Total denoising steps.
        verify_schedule: When to run verification during denoising.
        max_repairs_per_step: Max repair iterations per verification point.

    Returns:
        Tuple of:
            - (B, V, 2) final placement in normalized coords
            - List of ViolationFeedback from each verification
    """
    if verify_schedule is None:
        verify_schedule = VerifySchedule()

    cond = cond.to(adapter.device)
    num_vertices = cond.x.shape[0]
    feedback_history: list[ViolationFeedback] = []

    scheduler = adapter.scheduler
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    # Start from pure noise
    x = torch.randn(num_samples, num_vertices, 2, device=adapter.device)

    with torch.no_grad():
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t.expand(num_samples)
            t_next_batch = t_next.expand(num_samples)

            # Predict noise
            eps_pred = adapter.model(x, cond, t_batch)

            # Predict x_0 for verification
            if verify_schedule.should_verify(i, len(timesteps), t.item()):
                alpha_t = scheduler.alpha(t_batch).view(-1, 1, 1)
                sigma_t = scheduler.sigma(t_batch).view(-1, 1, 1)
                x0_pred = (x - sigma_t * eps_pred) / alpha_t.clamp(min=1e-8)

                # Verify each sample in batch
                for b_idx in range(num_samples):
                    x0_single = x0_pred[b_idx]  # (V, 2)
                    centers, sizes = adapter.decode_placement(x0_single, cond)
                    feedback = verifier(centers, sizes)
                    feedback_history.append(feedback)

                    if not feedback.is_legal():
                        # Apply repair in normalized coordinate space
                        x0_repaired, mask, max_alpha = repair.repair(
                            x0_single, feedback, loop_iter=0,
                        )
                        # Convert back to x_t space at current noise level
                        # x_t = alpha(t) * x0 + sigma(t) * eps
                        eps_for_repair = torch.randn_like(x0_repaired)
                        x_repaired = (
                            alpha_t[0] * x0_repaired + sigma_t[0] * eps_for_repair
                        )
                        # Only update repaired macros in x_t
                        mask_expanded = mask.unsqueeze(-1)  # (V, 1)
                        x[b_idx] = torch.where(mask_expanded, x_repaired, x[b_idx])

            # Standard denoising step
            z = torch.randn_like(x) if i < len(timesteps) - 2 else torch.zeros_like(x)
            x = scheduler.step(eps_pred, t_batch, t_next_batch, x, z)

    return x, feedback_history
