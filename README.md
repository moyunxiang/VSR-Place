# VSR-Place: Verifier-Guided Selective Re-noising for Constraint-Satisfying Macro Placement

A closed-loop refinement framework that bridges diffusion sampling and non-differentiable placement verification for chip macro placement.

## Overview

VSR-Place builds on top of [ChipDiffusion](https://github.com/vint-1/chipdiffusion) (ICML 2025) as a pretrained backbone. Instead of relying solely on differentiable legality potentials, it introduces a **verifier-in-the-loop** mechanism:

1. Generate a placement via diffusion model
2. Run a non-differentiable legality verifier to detect violations
3. Selectively re-noise only the offending macros
4. Re-denoise to repair the layout
5. Repeat until legal or budget exhausted

The key insight: layout violations are **sparse and localized**, so targeted re-noising is more effective than global resampling.

## Project Structure

```
src/vsr_place/
├── verifier/       # Non-differentiable legality verifier
├── renoising/      # Selective re-noising engine
├── backbone/       # ChipDiffusion adapter
├── loop/           # Closed-loop controller
├── conditioning/   # Repair variants (A: mask-only, B/C: extensions)
└── metrics/        # Legality, HPWL, violation tracking
```

## Quick Start

```bash
# Install (CPU, for development/testing)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Generate demo visualizations
python scripts/visualize.py
```

### Full Setup (GPU required)

```bash
# One-command environment setup
bash scripts/setup_env.sh

# Download pretrained checkpoints
bash scripts/download_checkpoints.sh

# Run experiments
python scripts/run_vsr.py --config configs/methods/vsr_mask_only.yaml
```

## Method

**Core formula** — selective re-noising for offending macro $i$:

$$x_i' = \sqrt{1 - \alpha_i} \cdot \hat{x}_{0,i} + \sqrt{\alpha_i} \cdot \epsilon_i$$

Non-offending macros are kept fixed: $x_i' = \hat{x}_{0,i}$

**Variants:**
- **Variant A** (implemented): Mask-only repair — violation severity determines which macros to re-noise
- **Variant B** (extension): Feature-level conditioning
- **Variant C** (extension): Rich structured conditioning via violation encoder

## References

- [Chip Placement with Diffusion Models](https://github.com/vint-1/chipdiffusion) (ICML 2025)
- [RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)
- [Universal Guidance for Diffusion Models](https://arxiv.org/abs/2302.07121)
