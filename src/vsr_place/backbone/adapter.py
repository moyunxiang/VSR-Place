"""Adapter wrapping ChipDiffusion model for VSR-Place usage.

This adapter provides a clean interface for VSR-Place to interact with
the ChipDiffusion diffusion model without modifying its source code.
"""

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

# Add ChipDiffusion to path
_CHIPDIFFUSION_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "chipdiffusion"
if str(_CHIPDIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHIPDIFFUSION_ROOT))


class ChipDiffusionAdapter:
    """Adapter for using ChipDiffusion as VSR-Place's diffusion backbone.

    Wraps model loading, sampling, denoising, and coordinate conversion.
    Conforms to the DiffusionBackend protocol expected by VSRLoop.

    Args:
        model: A loaded ChipDiffusion model (CondDiffusionModel or subclass).
        scheduler: CosineScheduler instance (for continuous models).
        device: Torch device.
    """

    def __init__(self, model: Any, scheduler: Any = None, device: str | torch.device = "cpu"):
        self.model = model
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: dict | None = None,
        device: str = "cpu",
    ) -> "ChipDiffusionAdapter":
        """Load a ChipDiffusion model from a checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            config: Model config dict. If None, tries to load from checkpoint.
            device: Device to load model on.

        Returns:
            ChipDiffusionAdapter instance.
        """
        # Import ChipDiffusion modules
        from diffusion.models import ContinuousDiffusionModel, GuidedDiffusionModel
        from diffusion.schedulers import CosineScheduler

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if config is None:
            config = checkpoint.get("config", checkpoint.get("cfg", {}))

        model_config = config.get("model", config)
        family = config.get("family", "continuous_diffusion")

        model_cls = {
            "continuous_diffusion": ContinuousDiffusionModel,
            "guided_diffusion": GuidedDiffusionModel,
        }.get(family, ContinuousDiffusionModel)

        model = model_cls(**model_config)

        # Load state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        scheduler = CosineScheduler()

        return cls(model=model, scheduler=scheduler, device=device)

    def sample(
        self,
        cond: Any,
        num_samples: int = 1,
        num_steps: int = 100,
        **kwargs: Any,
    ) -> Tensor:
        """Generate placements from noise using the diffusion model.

        Args:
            cond: PyG Data object with netlist graph.
            num_samples: Number of placement samples to generate.
            num_steps: Number of denoising steps.

        Returns:
            (B, V, 2) tensor of generated placements in normalized coords [-1, 1].
        """
        cond = cond.to(self.device)
        num_vertices = cond.x.shape[0]

        with torch.no_grad():
            if self.scheduler is not None:
                # Continuous model path
                self.scheduler.set_timesteps(num_steps)
                x = torch.randn(num_samples, num_vertices, 2, device=self.device)
                timesteps = self.scheduler.timesteps

                for i in range(len(timesteps) - 1):
                    t = timesteps[i].expand(num_samples)
                    t_next = timesteps[i + 1].expand(num_samples)
                    eps_pred = self.model(x, cond, t)
                    z = torch.randn_like(x) if i < len(timesteps) - 2 else torch.zeros_like(x)
                    x = self.scheduler.step(eps_pred, t, t_next, x, z)
            else:
                # Discrete model path - use built-in reverse_samples
                samples, _ = self.model.reverse_samples(
                    num_samples, None, cond, **kwargs
                )
                x = samples

        return x

    def predict_x0(self, x_t: Tensor, cond: Any, t: Tensor) -> Tensor:
        """Predict clean x_0 from noisy x_t at timestep t.

        For cosine schedule: x_0 = (x_t - sigma(t) * eps) / alpha(t)

        Args:
            x_t: (B, V, 2) noisy placement.
            cond: PyG Data object.
            t: (B,) timestep values.

        Returns:
            (B, V, 2) predicted clean placement.
        """
        cond = cond.to(self.device)
        with torch.no_grad():
            eps_pred = self.model(x_t, cond, t)

        if self.scheduler is not None:
            alpha_t = self.scheduler.alpha(t)  # (B,)
            sigma_t = self.scheduler.sigma(t)  # (B,)
            # Reshape for broadcasting: (B, 1, 1)
            alpha_t = alpha_t.view(-1, 1, 1)
            sigma_t = sigma_t.view(-1, 1, 1)
            x0_pred = (x_t - sigma_t * eps_pred) / alpha_t.clamp(min=1e-8)
        else:
            # Discrete schedule
            t_idx = t.long() - 1
            alpha_bar = self.model._alpha_bar[t_idx].view(-1, 1, 1)
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)

        return x0_pred

    def denoise_from(
        self,
        x_start: Tensor,
        cond: Any,
        start_timestep: float,
        num_steps: int = 50,
        **kwargs: Any,
    ) -> Tensor:
        """Resume denoising from an intermediate noisy state.

        This is the key method for VSR-Place: after selective re-noising,
        we resume denoising from the noise level corresponding to start_timestep.

        Args:
            x_start: (B, V, 2) partially noised placement.
            cond: PyG Data object.
            start_timestep: Noise level in [0, 1] to start denoising from.
                0 = clean, 1 = full noise. Maps to cosine schedule t.
            num_steps: Number of denoising steps to use.

        Returns:
            (B, V, 2) denoised placement.
        """
        cond = cond.to(self.device)
        b = x_start.shape[0]

        if start_timestep <= 1e-6:
            return x_start  # Already clean

        with torch.no_grad():
            if self.scheduler is not None:
                # Build a timestep schedule from start_timestep down to ~0
                self.scheduler.set_timesteps(num_steps)
                all_timesteps = self.scheduler.timesteps

                # Filter to only timesteps <= start_timestep
                # The scheduler timesteps go from high (noisy) to low (clean)
                valid_mask = all_timesteps >= (1.0 - start_timestep) * all_timesteps[0]
                # Actually: scheduler timesteps go from ~1 down to ~0
                # We want those <= start_timestep
                start_idx = 0
                for i, ts in enumerate(all_timesteps):
                    if ts.item() <= start_timestep + 1e-6:
                        start_idx = i
                        break

                timesteps = all_timesteps[start_idx:]
                x = x_start

                for i in range(len(timesteps) - 1):
                    t = timesteps[i].expand(b)
                    t_next = timesteps[i + 1].expand(b)
                    eps_pred = self.model(x, cond, t)
                    z = torch.randn_like(x) if i < len(timesteps) - 2 else torch.zeros_like(x)
                    x = self.scheduler.step(eps_pred, t, t_next, x, z)
            else:
                # Discrete model: map start_timestep to discrete step
                max_steps = self.model.max_diffusion_steps
                start_step = int(start_timestep * max_steps)
                x = x_start
                for t_val in range(start_step, 0, -1):
                    t = torch.full((b,), t_val, device=self.device, dtype=torch.long)
                    eps_pred = self.model(x, cond, t)
                    x = self._discrete_denoise_step(x, eps_pred, t_val)

        return x

    def _discrete_denoise_step(self, x_t: Tensor, eps_pred: Tensor, t: int) -> Tensor:
        """Single discrete denoising step (DDPM-style)."""
        alpha_bar_t = self.model._alpha_bar[t - 1]
        alpha_bar_prev = self.model._alpha_bar[t - 2] if t > 1 else torch.tensor(1.0)
        beta_t = self.model._beta[t - 1]

        mean = (1 / torch.sqrt(1 - beta_t)) * (
            x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_pred
        )

        if t > 1:
            sigma = self.model._sigma[t - 1]
            z = torch.randn_like(x_t)
            return mean + sigma * z
        return mean

    def decode_placement(
        self, x_normalized: Tensor, cond: Any
    ) -> tuple[Tensor, Tensor]:
        """Convert normalized coords [-1, 1] to absolute coordinates.

        Args:
            x_normalized: (B, V, 2) or (V, 2) placement in [-1, 1].
            cond: PyG Data object with chip_size and instance sizes.

        Returns:
            Tuple of:
                - centers: absolute center coordinates
                - sizes: absolute (width, height) per macro
        """
        single = x_normalized.dim() == 2
        if single:
            x_normalized = x_normalized.unsqueeze(0)

        # chip_size from cond
        if hasattr(cond, "chip_size"):
            chip_size = cond.chip_size
            if chip_size.dim() == 1 and chip_size.shape[0] == 4:
                # [xmin, ymin, xmax, ymax]
                canvas_w = (chip_size[2] - chip_size[0]).item()
                canvas_h = (chip_size[3] - chip_size[1]).item()
                offset_x = chip_size[0].item()
                offset_y = chip_size[1].item()
            elif chip_size.dim() == 1 and chip_size.shape[0] == 2:
                canvas_w = chip_size[0].item()
                canvas_h = chip_size[1].item()
                offset_x = 0.0
                offset_y = 0.0
            else:
                canvas_w = canvas_h = 1.0
                offset_x = offset_y = 0.0
        else:
            canvas_w = canvas_h = 2.0
            offset_x = offset_y = -1.0

        # De-normalize placement: x_abs = (x_norm + 1) / 2 * canvas + offset
        centers = (x_normalized + 1.0) / 2.0
        centers[..., 0] = centers[..., 0] * canvas_w + offset_x
        centers[..., 1] = centers[..., 1] * canvas_h + offset_y

        # De-normalize sizes: cond.x is normalized as 2 * (size / chip_size)
        # So absolute size = cond.x / 2 * chip_size
        norm_sizes = cond.x  # (V, 2)
        abs_sizes = norm_sizes.clone()
        abs_sizes[:, 0] = norm_sizes[:, 0] / 2.0 * canvas_w
        abs_sizes[:, 1] = norm_sizes[:, 1] / 2.0 * canvas_h

        if single:
            centers = centers.squeeze(0)

        return centers, abs_sizes

    def encode_placement(self, centers: Tensor, cond: Any) -> Tensor:
        """Convert absolute coordinates back to normalized [-1, 1].

        Args:
            centers: (B, V, 2) or (V, 2) absolute center coordinates.
            cond: PyG Data object with chip_size.

        Returns:
            Normalized placement tensor.
        """
        single = centers.dim() == 2
        if single:
            centers = centers.unsqueeze(0)

        if hasattr(cond, "chip_size"):
            chip_size = cond.chip_size
            if chip_size.dim() == 1 and chip_size.shape[0] == 4:
                canvas_w = (chip_size[2] - chip_size[0]).item()
                canvas_h = (chip_size[3] - chip_size[1]).item()
                offset_x = chip_size[0].item()
                offset_y = chip_size[1].item()
            elif chip_size.dim() == 1 and chip_size.shape[0] == 2:
                canvas_w = chip_size[0].item()
                canvas_h = chip_size[1].item()
                offset_x = 0.0
                offset_y = 0.0
            else:
                canvas_w = canvas_h = 1.0
                offset_x = offset_y = 0.0
        else:
            canvas_w = canvas_h = 2.0
            offset_x = offset_y = -1.0

        x_norm = centers.clone()
        x_norm[..., 0] = (centers[..., 0] - offset_x) / canvas_w * 2.0 - 1.0
        x_norm[..., 1] = (centers[..., 1] - offset_y) / canvas_h * 2.0 - 1.0

        if single:
            x_norm = x_norm.squeeze(0)

        return x_norm

    @staticmethod
    def noise_level_to_timestep(alpha: float) -> float:
        """Convert a re-noising strength alpha to a cosine schedule timestep.

        For cosine schedule: sigma(t) = sin(π/2 * t), so noise fraction = sin²(π/2 * t).
        Given alpha (noise variance fraction), solve for t:
            t = (2/π) * arcsin(sqrt(alpha))

        Args:
            alpha: Noise level in [0, 1]. 0 = clean, 1 = full noise.

        Returns:
            Corresponding timestep t in [0, 1].
        """
        import math
        alpha = max(0.0, min(1.0, alpha))
        return (2.0 / math.pi) * math.asin(math.sqrt(alpha))

    @staticmethod
    def timestep_to_noise_level(t: float) -> float:
        """Convert a cosine schedule timestep to noise level.

        Args:
            t: Timestep in [0, 1].

        Returns:
            Noise level alpha in [0, 1].
        """
        import math
        return math.sin(math.pi / 2.0 * t) ** 2
