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

#### 2026-04-15 00:00 HKT — 补全 GPU 实验链路 + AutoDL 部署准备
**Context**: 本地所有 CPU 代码已完成。补全 GPU 所需的完整实验链路：数据加载、baseline 脚本、checkpoint 下载。
**Actions**:
- 研究 ChipDiffusion eval.py 完整调用链：`load_graph_data_with_config(task)` → `val_set[i]` → `(x, cond)` → model 采样 → metrics
- 重写 `scripts/run_vsr.py`：
  - 新增 `load_benchmark_data(task)` 函数，调用 ChipDiffusion 的 `utils.load_graph_data_with_config()`
  - 新增 `run_single_sample()` 函数，处理 VSR 和 baseline 两种模式
  - 新增 CLI 参数：`--checkpoint`, `--task`, `--no-vsr`
  - 完整结果保存：per-sample metrics + summary
- 新增 `scripts/run_baseline.py`：直接调用 ChipDiffusion 的 eval.py（subprocess），支持 unguided/guided/guided_legalized/macro_only 四种模式
- 更新 `scripts/download_checkpoints.sh`：使用 gdown 从 Google Drive 下载 Large+v2 checkpoint，自动 symlink 到 ChipDiffusion 的 logs/ 目录
- 更新 `configs/defaults.yaml`：新增 `benchmark.task` 字段
- 49/49 测试通过，dry-run 正常
**Results**: 实验链路代码完整。在 AutoDL 上只需：clone → install → download checkpoint → run。
**Decisions / Assumptions**:
- Verifier 使用 canvas_width=2.0, canvas_height=2.0（ChipDiffusion 归一化坐标 [-1,1] 对应宽度 2）。Verification: 在 GPU 上验证 decode_placement 输出范围。
- load_benchmark_data 需要 chdir 到 ChipDiffusion 目录（因为其内部用相对路径）。Verification: 在 AutoDL 上实际运行验证。
- run_baseline.py 用 subprocess 调用 ChipDiffusion eval.py 而非 import，避免 Hydra config 冲突。
**Next**: Push 到 GitHub → 在 AutoDL 上 clone + 跑实验。

#### 2026-04-15 00:30 HKT — 补全所有缺口 + requirements.txt
**Context**: 之前的代码有 6 个缺口无法完成全部实验。逐一修复。
**Actions**:
- **Fix 1 - adapter.from_checkpoint()**: 重写为使用 ChipDiffusion 的 `Checkpointer` 类加载 checkpoint（checkpoint 格式是 `{"model": state_dict, ...}`）。新增 `_default_large_config()` 匹配 Large+v2 模型参数。新增 `from_eval_config()` 可自动从 benchmark 数据推断 input_shape。
- **Fix 2 - verifier canvas**: `run_vsr.py` 中改为从 `adapter.get_canvas_size(cond)` 动态获取，不再写死。adapter 新增 `get_canvas_size()` 和 `_get_canvas_params()` 方法。
- **Fix 3 - VSR + legalizer**: adapter 新增 `legalize()` 方法调用 ChipDiffusion 的 `legalization.legalize_opt()`。`run_vsr.py` 新增 `--legalize` 参数。
- **Fix 4 - run_ablations.py**: 实现完整消融实验脚本，含 proposal 中定义的全部 4 个维度 21 个变体（scope×8, strength×5, budget×4, constraints×4）。支持 `--dry-run`, `--list`, `--ablations` 筛选。
- **Fix 5 - run_all.sh**: 一键跑全部实验：4 个基线 + VSR + VSR+legalizer + 消融套件。
- **Fix 6 - requirements.txt**: 完整依赖列表，包含 ChipDiffusion 的额外依赖（performer-pytorch, termcolor, rich 等）。
- 49/49 测试通过
**Results**: 所有已知缺口已修复。AutoDL 上只需 `bash scripts/run_all.sh <ckpt> <task> <seed>` 即可跑完全部实验。
**Decisions / Assumptions**:
- Checkpoint 加载使用 ChipDiffusion 的 Checkpointer 而非直接 torch.load，保证与训练时的保存格式一致。
- Legalization 默认用 opt 模式（opt-adam），参数沿用 ChipDiffusion 的默认值。Verification: GPU 上实际运行验证。
- scheduler.step() 返回值可能是 tuple (x, x0_pred)，adapter 中做了兼容处理。
**Next**: Push 到 GitHub，准备 AutoDL 部署。
