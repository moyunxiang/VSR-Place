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

#### 2026-04-15 21:00 HKT — AutoDL 部署 + 首次 GPU 实验成功
**Context**: 在 AutoDL RTX 4090 (24GB) 上部署并跑通完整实验链路。
**Actions**:
- SSH 连接 AutoDL（connect.bjb1.seetacloud.com:43348）
- 克隆仓库（GitHub 被墙，用 ghfast.top 镜像加速）
- 安装依赖，49/49 测试通过
- Checkpoint: 本地下载 large-v2.ckpt (73MB, 6.28M params) → scp 上传（Google Drive 在国内被墙）
- 数据生成：`generate.py` 单线程生成 v1.61 的 2 个验证样本（`00000005.pickle` 含 5 个 (x,cond) tuple, `00000100.pickle` 含更多）
- 修复 8 个 GPU 运行时问题：
  1. `diffusion/` 目录加入 sys.path（ChipDiffusion 用 bare import 如 `import pos_encoding`）
  2. Mock torchvision/moviepy（版本冲突/未安装，VSR-Place 不需要）
  3. 自写数据加载器支持 ChipDiffusion 的 batch pickle 格式（`list of (x, cond) tuples`）
  4. dataset config 需要 `train_samples`/`val_samples` 字段（旧格式），非 `num_train_samples`
  5. scheduler timesteps 需要 `.to(device)` 移到 GPU
  6. `fill_diagonal_` 在某些 PyTorch 版本报错，改用 index 赋值
  7. VSRLoop 启动时需 `cond = cond.to(backend.device)`（PyG Data.to() 返回新对象）
  8. `denoise_from` 返回 (1,V,2) 需 squeeze batch 维
- 跑通基线：Sample 1: 307 violations, Sample 2: 98 violations (3.4s)
- 跑通 VSR-Place：4 repair loops/sample, Sample 1: 355 viol, Sample 2: 102 viol (5.3s)
**Results**: 完整链路验证通过。Baseline 和 VSR-Place 均可在 GPU 上端到端运行。
**Decisions / Assumptions**:
- 用 ghfast.top 作为 GitHub 镜像（国内 AutoDL 无法直连 GitHub）
- 数据加载绕过 ChipDiffusion 的 `load_graph_data_with_config`（依赖太多），自写 pickle 解析器
- Assumption: VSR-Place violations 比 baseline 高是因为样本太少 + denoise_steps 不够。Verification: 用更多数据 + 调参验证。
**Next**: 生成 20 个验证样本 → 跑完整对比实验 + 消融。

#### 2026-04-16 00:30 HKT — 首轮完整实验（20 样本）+ 消融结果
**Context**: 在 AutoDL RTX 4090 上跑 20 样本的完整 baseline + VSR-Place + 6 个消融实验。
**Actions**:
- 修复 3 个运行时问题：
  1. `run_ablations.py` import 路径修正（`from scripts.run_vsr` → `sys.path + from run_vsr`）
  2. dataset config val_samples 写成了 pickle 文件数（1）而非实际样本数（20）
  3. jinja2 版本升级
- 生成 20 个 v1.61 验证样本（350s，单个 pickle 含 20 tuple）
- 跑 Baseline (unguided, no VSR)：20 samples, seed=42
- 跑 VSR-Place (α=0.3, budget=4)：20 samples, seed=42
- 跑 6 个消融：strength {0.1, 0.3, 0.5} × budget {1, 2, 8}
**Results**:

| 方法 | Avg Violations | Pass Rate | Time |
|------|---------------|-----------|------|
| Baseline (no VSR) | **184.55** | 0% | 30s |
| VSR α=0.1, budget=4 | 206.70 | 0% | 35s |
| VSR α=0.3, budget=4 | 204.50 | 0% | 47s |
| VSR α=0.5, budget=4 | 204.65 | 0% | 60s |
| VSR α=0.3, budget=1 | 207.80 | 0% | 34s |
| VSR α=0.3, budget=2 | 204.85 | 0% | 38s |
| VSR α=0.3, budget=8 | 202.45 | 0% | 65s |

**关键发现：VSR-Place 目前比 baseline 更差（+10-12% violations）。**

**Decisions / Assumptions**:
- 根因分析（3 个可能原因）：
  1. `denoise_from` 只用 50 步 re-denoise，不足以将重噪声后的 placement 恢复到合理状态
  2. 模型在混合噪声级别输入上表现差（部分 macro 有噪声、部分没有，GNN 训练时所有节点同一噪声级别）
  3. 选择性重噪声后直接在归一化坐标空间操作，但 start_timestep 映射到 cosine schedule 的 noise level 可能不准确
- 这些问题在 proposal 的 Risk 栏（第 1、2 条）中已预期
- 更多 budget 迭代（8 loops）略有改善（202.45 vs 207.80），说明方向不完全错但修复力度不够
- Assumption: 核心问题是 re-denoise 不够深，需要更多 denoising 步数或完整 re-denoising。Verification: 增加 denoise_steps 到 100/200 看效果。
**Next**: 
1. 增加 denoise_steps（当前 50 → 试 100, 200）看是否改善
2. 尝试 intra-sampling 模式而非 post-sampling
3. 分析具体哪些 macro 被 re-noise 了、re-denoise 后是变好还是变差

#### 2026-04-16 01:00 HKT — 诊断并修复 timestep 转换 bug
**Context**: 深入分析首轮实验 VSR 比 baseline 差的根因。
**Actions**:
- 对比 ChipDiffusion `schedulers.py` 的 `add_noise: x_t = cos(πt/2)*x + sin(πt/2)*ε` 与我们 `renoise.py` 的 `x' = sqrt(1-α)*x + sqrt(α)*ε`
- 确认：当 α = sin²(πt/2) 时两个公式数学等价，**re-noising 公式本身没错**
- 发现真正的 bug：`vsr_loop.py:159-160` 将 noise fraction `max_alpha` 直接作为 `start_timestep` 传给 `denoise_from`，但没有做 cosine schedule 的坐标转换
  - 例如 α=0.3 时，正确 timestep = (2/π)*arcsin(√0.3) ≈ 0.377，我们传的是 0.3
  - denoiser 以为噪声是 sin²(π*0.3/2) ≈ 0.206，实际是 0.3 → 噪声水平不匹配
- `noise_level_to_timestep()` 方法已存在于 adapter.py 但从未被调用
- 修复：
  1. `vsr_loop.py`: 添加 `(2/π)*arcsin(√α)` 转换
  2. `adapter.py`: `denoise_from` 默认步数 50 → 100
  3. `configs/defaults.yaml`: `denoise_steps` 50 → 100
- 49/49 测试通过
**Results**: Bug 已修复，待 AutoDL 验证。
**Decisions / Assumptions**:
- 直接内联转换公式而非调用 `noise_level_to_timestep()` 静态方法，避免循环 import
- denoise_steps=100 与初始采样步数一致，保证足够的去噪质量
- Assumption: 修复 timestep 转换后 VSR violations 应低于 baseline。Verification: AutoDL 跑 20 样本对比。
**Next**: push → AutoDL 拉取 → 重跑 baseline vs VSR-Place 对比。
