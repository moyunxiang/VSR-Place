"""Adapter wrapping ChipDiffusion model for VSR-Place usage.

This adapter provides a clean interface for VSR-Place to interact with
the ChipDiffusion diffusion model without modifying its source code.
"""

import os
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
        model_config: dict | None = None,
        device: str = "cuda",
    ) -> "ChipDiffusionAdapter":
        """Load a ChipDiffusion model from a checkpoint.

        Uses ChipDiffusion's own Checkpointer for correct loading.

        Args:
            checkpoint_path: Path to .ckpt checkpoint file.
            model_config: Model constructor kwargs. If None, uses default large config.
            device: Device to load model on.

        Returns:
            ChipDiffusionAdapter instance.
        """
        from diffusion.models import ContinuousDiffusionModel
        from diffusion.schedulers import CosineScheduler
        from common.checkpoint import Checkpointer

        if model_config is None:
            model_config = _default_large_config(device=device)

        model = ContinuousDiffusionModel(**model_config).to(device)

        # Use ChipDiffusion's Checkpointer for correct state_dict loading
        checkpointer = Checkpointer()
        checkpointer.register({"model": model})
        loaded = checkpointer.load(checkpoint_path)

        if loaded:
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        else:
            print(f"WARNING: Could not load checkpoint from {checkpoint_path}, using random weights")

        model.eval()
        scheduler = CosineScheduler()

        return cls(model=model, scheduler=scheduler, device=device)

    @classmethod
    def from_eval_config(
        cls,
        checkpoint_path: str,
        task: str = "v1.61",
        device: str = "cuda",
    ) -> "ChipDiffusionAdapter":
        """Load model by mimicking ChipDiffusion's eval.py setup.

        This reads a benchmark sample to determine input_shape, then
        constructs the model with matching config.

        Args:
            checkpoint_path: Path to .ckpt file.
            task: Dataset task name (to determine input_shape).
            device: Device.

        Returns:
            ChipDiffusionAdapter instance.
        """
        from diffusion import utils as cd_utils

        # Load one sample to get input shape
        original_cwd = os.getcwd()
        os.chdir(str(_CHIPDIFFUSION_ROOT))
        try:
            _, val_set = cd_utils.load_graph_data_with_config(task)
            sample_shape = val_set[0][0].shape  # (V, 2)
        finally:
            os.chdir(original_cwd)

        config = _default_large_config()
        config["input_shape"] = tuple(sample_shape)
        config["device"] = device

        return cls.from_checkpoint(checkpoint_path, model_config=config, device=device)

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
                self.scheduler.set_timesteps(num_steps)
                x = torch.randn(num_samples, num_vertices, 2, device=self.device)
                timesteps = self.scheduler.timesteps

                for i in range(len(timesteps) - 1):
                    t = timesteps[i].expand(num_samples)
                    t_next = timesteps[i + 1].expand(num_samples)
                    eps_pred = self.model(x, cond, t)
                    z = torch.randn_like(x) if i < len(timesteps) - 2 else torch.zeros_like(x)
                    result = self.scheduler.step(eps_pred, t, t_next, x, z)
                    # step returns (x_t_minus_one, predicted_x0) tuple
                    x = result[0] if isinstance(result, tuple) else result
            else:
                samples, _ = self.model.reverse_samples(
                    num_samples, None, cond, **kwargs
                )
                x = samples

        return x

    def predict_x0(self, x_t: Tensor, cond: Any, t: Tensor) -> Tensor:
        """Predict clean x_0 from noisy x_t at timestep t.

        For cosine schedule: x_0 = (x_t - sigma(t) * eps) / alpha(t)
        """
        cond = cond.to(self.device)
        with torch.no_grad():
            eps_pred = self.model(x_t, cond, t)

        alpha_t = self.scheduler.alpha(t).view(-1, 1, 1)
        sigma_t = self.scheduler.sigma(t).view(-1, 1, 1)
        x0_pred = (x_t - sigma_t * eps_pred) / alpha_t.clamp(min=1e-8)

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

        Args:
            x_start: (B, V, 2) or (V, 2) partially noised placement.
            cond: PyG Data object.
            start_timestep: Noise level in [0, 1] to start denoising from.
                0 = clean, 1 = full noise.
            num_steps: Number of denoising steps to use.

        Returns:
            (B, V, 2) denoised placement.
        """
        if x_start.dim() == 2:
            x_start = x_start.unsqueeze(0)

        cond = cond.to(self.device)
        b = x_start.shape[0]

        if start_timestep <= 1e-6:
            return x_start

        with torch.no_grad():
            self.scheduler.set_timesteps(num_steps)
            all_timesteps = self.scheduler.timesteps

            # Find the first timestep <= start_timestep
            # Timesteps go from ~1 (noisy) down to ~0 (clean)
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
                result = self.scheduler.step(eps_pred, t, t_next, x, z)
                x = result[0] if isinstance(result, tuple) else result

        return x

    def decode_placement(
        self, x_normalized: Tensor, cond: Any
    ) -> tuple[Tensor, Tensor]:
        """Convert normalized coords [-1, 1] to absolute coordinates.

        ChipDiffusion normalizes:
          placement: x_norm = 2 * (x_abs - offset) / chip_size - 1
          sizes:     s_norm = 2 * s_abs / chip_size

        Returns:
            Tuple of (centers, sizes) in absolute coordinates.
        """
        single = x_normalized.dim() == 2
        if single:
            x_normalized = x_normalized.unsqueeze(0)

        canvas_w, canvas_h, offset_x, offset_y = self._get_canvas_params(cond)

        centers = x_normalized.clone()
        centers[..., 0] = (x_normalized[..., 0] + 1.0) / 2.0 * canvas_w + offset_x
        centers[..., 1] = (x_normalized[..., 1] + 1.0) / 2.0 * canvas_h + offset_y

        norm_sizes = cond.x  # (V, 2)
        abs_sizes = norm_sizes.clone()
        abs_sizes[:, 0] = norm_sizes[:, 0] / 2.0 * canvas_w
        abs_sizes[:, 1] = norm_sizes[:, 1] / 2.0 * canvas_h

        if single:
            centers = centers.squeeze(0)

        return centers, abs_sizes

    def encode_placement(self, centers: Tensor, cond: Any) -> Tensor:
        """Convert absolute coordinates back to normalized [-1, 1]."""
        single = centers.dim() == 2
        if single:
            centers = centers.unsqueeze(0)

        canvas_w, canvas_h, offset_x, offset_y = self._get_canvas_params(cond)

        x_norm = centers.clone()
        x_norm[..., 0] = (centers[..., 0] - offset_x) / canvas_w * 2.0 - 1.0
        x_norm[..., 1] = (centers[..., 1] - offset_y) / canvas_h * 2.0 - 1.0

        if single:
            x_norm = x_norm.squeeze(0)

        return x_norm

    def get_canvas_size(self, cond: Any) -> tuple[float, float]:
        """Get absolute canvas dimensions from a cond object.

        Returns:
            (canvas_width, canvas_height) in absolute coordinates.
        """
        canvas_w, canvas_h, _, _ = self._get_canvas_params(cond)
        return canvas_w, canvas_h

    def _get_canvas_params(self, cond: Any) -> tuple[float, float, float, float]:
        """Extract canvas width, height, and offset from cond."""
        if hasattr(cond, "chip_size") and cond.chip_size is not None:
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
                canvas_w = canvas_h = 2.0
                offset_x = offset_y = -1.0
        else:
            # Default: normalized space [-1, 1] -> canvas is 2x2
            canvas_w = canvas_h = 2.0
            offset_x = offset_y = -1.0
        return canvas_w, canvas_h, offset_x, offset_y

    def legalize(self, x: Tensor, cond: Any, mode: str = "opt", **kwargs) -> Tensor:
        """Apply ChipDiffusion's legalization to a placement.

        Args:
            x: (B, V, 2) or (V, 2) placement in normalized coords.
            cond: PyG Data object.
            mode: 'opt' (opt-adam) or 'scheduled'.

        Returns:
            Legalized placement tensor.
        """
        from diffusion import legalization

        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)

        x = x.to(self.device).detach().clone().requires_grad_(True)
        cond = cond.to(self.device)

        if mode == "opt":
            x_legal, _, _ = legalization.legalize_opt(x, cond, **kwargs)
        else:
            x_legal, _, _ = legalization.legalize(x, cond, **kwargs)

        if single:
            x_legal = x_legal.squeeze(0)

        return x_legal.detach()

    @staticmethod
    def noise_level_to_timestep(alpha: float) -> float:
        """Convert re-noising strength alpha to cosine schedule timestep.

        sigma(t) = sin(π/2 * t), noise fraction = sin²(π/2 * t) = alpha
        => t = (2/π) * arcsin(sqrt(alpha))
        """
        import math
        alpha = max(0.0, min(1.0, alpha))
        return (2.0 / math.pi) * math.asin(math.sqrt(alpha))

    @staticmethod
    def timestep_to_noise_level(t: float) -> float:
        """Convert cosine schedule timestep to noise level."""
        import math
        return math.sin(math.pi / 2.0 * t) ** 2


def _default_large_config(input_shape=(61, 2), device="cpu") -> dict:
    """Default model config matching ChipDiffusion Large+v2.

    Derived from checkpoints/large-v2/config.yaml shipped with the
    official pretrained checkpoint.
    """
    from omegaconf import OmegaConf

    backbone_params = OmegaConf.create({
        "edge_features": 4,
        "cond_node_features": 2,
        "hidden_size": 256,
        "hidden_node_features": [256, 256, 256],
        "attention_node_features": [256, 256, 256],
        "layers_per_block": 2,
        "input_encoding_dim": 32,
        "dropout": 0.0,
        "num_heads": 4,
        "mlp_num_layers": 2,
        "mlp_size_factor": 4,
        "ff_num_layers": 2,
        "ff_size_factor": 1,
        "att_implementation": "flash",
        "dir_att_input": True,
        "conv_params": {
            "layer_type": "gat",
            "heads": 4,
            "concat": True,
        },
        "in_node_features": input_shape[1],
        "out_node_features": input_shape[1],
        "t_encoding_dim": 32,
        "device": device,
        "mask_key": "is_ports",
    })

    return {
        "backbone": "att_gnn",
        "backbone_params": backbone_params,
        "input_shape": tuple(input_shape),
        "t_encoding_type": "sinusoid",
        "t_encoding_dim": 32,
        "max_diffusion_steps": 1000,
        "noise_schedule": "cosine",
        "mask_key": "is_ports",
        "use_mask_as_input": True,
        "num_classes": 10,
        "device": device,
        "guidance_mode": "none",
        "legality_guidance_weight": 0.0,
        "hpwl_guidance_weight": 0.0,
        "grad_descent_steps": 0,
    }
