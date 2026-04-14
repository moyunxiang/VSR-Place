# VSR-Place Development Log

#### 2026-04-14 00:00 HKT — Project kickoff and plan creation
**Context**: New research project VSR-Place. Repo only has `proposal.md` and `CLAUDE.md`. Need to implement Verifier-Guided Selective Re-noising for macro placement on top of ChipDiffusion backbone.
**Actions**:
- Read `proposal.md` (443 lines) — full research proposal
- Explored codebase: confirmed empty repo (no code, no deps, no data)
- Searched for ChipDiffusion upstream: confirmed at https://github.com/vint-1/chipdiffusion (ICML 2025)
- Designed 8-phase implementation plan
- Created plan file, got user approval
**Results**: Plan approved. 8 phases: scaffolding → verifier → renoising → adapter → loop → eval → ablations → extensions
**Decisions / Assumptions**:
- Assumption: ChipDiffusion repo is accessible and its eval pipeline works out of the box. Verification: clone + run eval on one benchmark.
- Decision: Use git submodule for ChipDiffusion (not fork). Reason: keep upstream changes trackable, no source modification needed for Variant A.
- Decision: Start with Variant A (mask-only repair) as core deliverable. Variants B/C are extensions.
**Next**: Phase 0 — initialize git, create directory structure, add ChipDiffusion submodule, set up environment.

#### 2026-04-14 00:15 HKT — Phase 0/1/2/4 scaffolding and core implementation
**Context**: Plan approved. Begin implementing project structure and core modules.
**Actions**:
- `git init` — initialized git repo
- Created full directory structure: `src/vsr_place/{verifier,renoising,backbone,loop,conditioning,metrics}`, `configs/`, `scripts/`, `tests/`, `third_party/`, `results/`
- Created `.gitignore`, `pyproject.toml`, `setup.py` (shim for old pip)
- **Phase 1 (Verifier)**: Implemented `geometry.py` (pairwise overlap, boundary violation, spacing violation — all vectorized PyTorch), `feedback.py` (ViolationFeedback dataclass with is_legal/offending_macros/top_k), `verifier.py` (Verifier class with batch support)
- **Phase 2 (Re-noising)**: Implemented `selector.py` (GlobalSelector, ThresholdSelector, TopKSelector, AdaptiveThresholdSelector), `strength.py` (FixedStrength, SeverityAdaptiveStrength, ScheduledStrength), `renoise.py` (selective_renoise core formula + batch version)
- **Phase 4 partial (Loop)**: Implemented `vsr_loop.py` (VSRLoop controller with DiffusionBackend protocol), `budget.py` (RepairBudget with max_loops/early_stop/min_improvement)
- **Conditioning**: Implemented `mask_only.py` (Variant A)
- **Metrics**: Implemented `legality.py`, `hpwl.py` (list + vectorized), `violations.py`, `runtime.py`
- **Tests**: 42 tests across `test_geometry.py`, `test_verifier.py`, `test_selector.py`, `test_renoise.py`
- Installed torch 2.11.0+cpu, pytest 9.0.3
- Ran: `PYTHONPATH=src pytest tests/ -v`
**Results**: 42/42 tests passed in 0.24s. All geometry primitives, verifier, selectors, and re-noising logic verified.
**Decisions / Assumptions**:
- Used `setup.py` shim because pip 22.0.2 doesn't support PEP 660 editable installs with pure pyproject.toml. Verification: works after adding shim.
- Verifier uses protrusion length (not area) for boundary violations — simpler and gradient-free. Consistent with proposal's "not requiring gradients."
- Spacing violation computation: `max(gap_x, gap_y)` for AABB gap, violation = `max(0, min_spacing - gap)`.
**Next**: Phase 3 — implement ChipDiffusion backbone adapter (requires cloning the ChipDiffusion repo). Then wire up the full closed-loop with real backbone.

#### 2026-04-14 00:30 HKT — Fix UNKNOWN.egg-info + Phase 3/5 complete (all local work)
**Context**: Old pip (22.0.2) couldn't parse pyproject.toml properly → registered as "UNKNOWN". Need to fix install and complete all GPU-independent work.
**Actions**:
- Fixed `setup.py` shim: added explicit `name`, `version`, `package_dir`, `packages`. Reinstalled → now `vsr-place 0.1.0`
- `git submodule add https://github.com/vint-1/chipdiffusion.git third_party/chipdiffusion` — pinned at commit `c6fb7e3`
- Studied ChipDiffusion API thoroughly:
  - `ContinuousDiffusionModel`: forward pass `(x, cond, t) → eps_pred`, coords in [-1,1]
  - `CosineScheduler`: `alpha(t)=cos(πt/2)`, `sigma(t)=sin(πt/2)`, t∈[0,1]
  - `GuidedDiffusionModel.guidance_force_xhat`: gradient descent on legality + HPWL potentials
  - Data format: PyG Data with `x=(V,2)` sizes, `edge_index`, `edge_attr=(E,4)` pin offsets
- **Phase 3 (Adapter)**: Implemented `backbone/adapter.py` (`ChipDiffusionAdapter` — from_checkpoint, sample, predict_x0, denoise_from, decode/encode_placement, noise_level↔timestep mapping) and `backbone/sampling.py` (intra-sampling VSR with VerifySchedule)
- **Integration tests**: `tests/test_loop.py` — 7 tests with MockBackend (loop convergence, early stop, budget, metrics)
- **Experiment scripts**: `scripts/run_vsr.py` (config-driven entry point, dry-run mode), `scripts/visualize.py` (placement + convergence plots), `scripts/aggregate_results.py` (CSV + LaTeX tables)
- **Configs**: `configs/defaults.yaml`, `configs/methods/{baseline_unguided, baseline_guided, vsr_mask_only}.yaml`
- **Setup scripts**: `scripts/setup_env.sh`, `scripts/download_checkpoints.sh`, `scripts/download_benchmarks.sh`
- Installed matplotlib, pandas, seaborn. Ran demo visualization → 2 figures generated
- Commands: `pytest tests/ -v` → 49/49 passed (0.22s); `python3 scripts/run_vsr.py --dry-run` → config printed OK; `python3 scripts/visualize.py` → 2 PNG saved
**Results**: All 49 tests pass. All local-runnable code complete. Two demo visualizations generated (placement heatmap + convergence curve).
**Decisions / Assumptions**:
- Adapter wraps ChipDiffusion without modifying source. Uses model's `__call__` and scheduler directly rather than `reverse_samples` (which doesn't expose intermediate states). Verification: will need GPU test.
- Cosine schedule mapping: `t = (2/π) * arcsin(√α)` for noise_level→timestep. Verification: will compare with ChipDiffusion's own scheduler output on GPU.
- Assumption: ChipDiffusion checkpoint format has `model_state_dict` key. Verification: inspect actual checkpoint file on GPU machine.
**Next**: All remaining work requires GPU:
1. Load ChipDiffusion checkpoint and verify adapter produces valid samples
2. Run ChipDiffusion baselines to reproduce published numbers
3. Run VSR-Place end-to-end on real benchmarks
4. Run ablation suite
