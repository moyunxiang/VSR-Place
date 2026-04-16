# VSR-Place Implementation Plan

## Context

基于 `proposal.md`，本项目实现 **Verifier-Guided Selective Re-noising (VSR-Place)**——一个闭环精炼框架，在预训练的 ChipDiffusion 扩散模型基础上，通过不可微分的合法性验证器引导局部重噪声，修复芯片宏单元布局中的约束违反。

当前仓库状态：**全新仓库**，仅有 `proposal.md` 和 `CLAUDE.md`，无任何代码。

ChipDiffusion 上游仓库：[vint-1/chipdiffusion](https://github.com/vint-1/chipdiffusion)（ICML 2025），提供扩散模型代码、预训练权重和评估协议。

---

## Goal

实现 VSR-Place 的核心方法（Variant A: mask-only repair），在 ICCAD/ISPD 基准上验证四个研究假设（H1-H4），产出可复现的实验结果。

## Non-Goals

- 不重新训练 ChipDiffusion backbone（核心方法是 inference-time 的）
- 不实现完整工业 DRC 验证
- Variant B/C（feature-level / rich conditioning）为延伸目标，不在核心交付范围内

---

## Project Structure

```
vsr_place/
├── CLAUDE.md / proposal.md / plan.md / log.md
├── pyproject.toml                    # 依赖管理
├── environment.yaml                  # Conda 环境
├── README.md
├── third_party/
│   └── chipdiffusion/                # Git submodule (pinned commit)
├── src/vsr_place/
│   ├── __init__.py
│   ├── verifier/                     # Phase 1
│   │   ├── geometry.py               # 矩形重叠/边界检查原语
│   │   ├── feedback.py               # ViolationFeedback 数据类
│   │   └── verifier.py               # 主验证器 V(x) -> 结构化反馈
│   ├── renoising/                    # Phase 2
│   │   ├── selector.py               # 宏选择策略 (threshold/top-k/global)
│   │   ├── strength.py               # 重噪声强度调度
│   │   └── renoise.py                # 选择性重噪声核心公式
│   ├── backbone/                     # Phase 3
│   │   ├── adapter.py                # ChipDiffusion 适配器
│   │   └── sampling.py               # 修改后的采样循环 (带验证器钩子)
│   ├── loop/                         # Phase 4
│   │   ├── vsr_loop.py               # 闭环控制器
│   │   └── budget.py                 # 修复预算和终止条件
│   ├── conditioning/                 # Phase 4 (Variant A) / Phase 7 (B/C)
│   │   ├── mask_only.py              # Variant A
│   │   ├── feature_cond.py           # Variant B (延伸)
│   │   └── violation_encoder.py      # Variant C (延伸)
│   └── metrics/                      # Phase 5
│       ├── legality.py
│       ├── violations.py
│       ├── hpwl.py
│       └── runtime.py
├── configs/                          # Hydra YAML 配置
│   ├── defaults.yaml
│   ├── backbone/ benchmarks/ methods/ ablations/
├── scripts/
│   ├── setup_env.sh                  # 环境一键安装
│   ├── download_checkpoints.sh       # 下载预训练权重
│   ├── download_benchmarks.sh        # 下载基准数据集
│   ├── run_baseline.py               # 跑基线实验
│   ├── run_vsr.py                    # 跑 VSR-Place 实验
│   ├── run_ablations.py              # 跑消融实验
│   ├── aggregate_results.py          # 汇总结果到表格
│   └── visualize.py                  # 布局可视化
├── tests/
│   ├── conftest.py                   # 共享 fixtures
│   ├── test_geometry.py
│   ├── test_verifier.py
│   ├── test_selector.py
│   ├── test_renoise.py
│   └── test_loop.py
└── results/                          # .gitignore, 实验输出
```

---

## Milestones

### Phase 0: Project Scaffolding ✅ DONE

**目标**：仓库初始化，环境配置，验证 ChipDiffusion 可运行。

**任务**：
1. 初始化 git 仓库，创建上述目录结构
2. 添加 ChipDiffusion 为 git submodule（`third_party/chipdiffusion/`），pin 到特定 commit
3. 创建 `environment.yaml`（继承 ChipDiffusion 依赖：PyTorch 2.2.1, torch-geometric 2.4.0, hydra-core, wandb 等）
4. 编写 `scripts/setup_env.sh`
5. 编写 `scripts/download_checkpoints.sh` 和 `scripts/download_benchmarks.sh`
6. 验证 ChipDiffusion 自带的 eval 脚本在至少一个基准上能运行
7. 记录基线结果作为参考数字
8. 创建 `plan.md`、`log.md`

**验收标准**：
- 环境安装成功，ChipDiffusion eval 可运行
- 至少一个基准电路的基线结果已记录

---

### Phase 1: Standalone Verifier ✅ DONE

**目标**：独立实现并测试非微分合法性验证器。

**关键实现**：
- `geometry.py`：矢量化矩形重叠面积 / 边界越界面积 / 对距离矩阵 / 间距违反检查（PyTorch GPU）
- `feedback.py`：`ViolationFeedback` 数据类
  - `pairwise_matrix: Tensor (N×N)` — 对间违反严重度
  - `severity_vector: Tensor (N,)` — 逐宏聚合严重度（= Σ_j A_ij + b_i）
  - `boundary_violations: Tensor (N,)` — 逐宏边界违反量
  - `global_stats: dict` — 总违反数、总重叠面积、通过/失败
  - 方法：`is_legal()`, `offending_macros(threshold)`, `top_k_offending(k)`
- `verifier.py`：`Verifier(canvas_w, canvas_h, min_spacing)` → `__call__(placements, sizes) -> ViolationFeedback`
  - 处理归一化坐标 [-1,1] 和绝对坐标
  - 支持批处理 (B 个布局)

**验收标准**：
- 单元测试 100% 通过（已知重叠/边界越界/边缘情况）
- N=100 宏在 GPU 上 <10ms

---

### Phase 2: Selective Re-noising Engine ✅ DONE

**目标**：实现选择性重噪声机制。

**关键实现**：
- `selector.py`：`ThresholdSelector`, `TopKSelector`, `GlobalSelector`, `AdaptiveThresholdSelector` → 返回 `(B, N)` 二值掩码
- `strength.py`：`FixedStrength(alpha)`, `SeverityAdaptiveStrength(alpha_min, alpha_max)`, `ScheduledStrength(alpha_schedule)` → 返回 `(B, N)` alpha 张量
- `renoise.py`：核心公式 `selective_renoise(x_hat_0, mask, alpha, noise=None) -> x_renoised`
  - 选中宏：`x_i' = sqrt(1-α_i) * x̂₀ᵢ + sqrt(α_i) * εᵢ`
  - 未选中宏：`x_i' = x̂₀ᵢ`（保持不变）

**验收标准**：
- 未选中宏的输出与输入 bit-identical
- alpha 值在有效范围内
- 统计测试验证重噪声方差

---

### Phase 3: ChipDiffusion Backbone Adapter ✅ DONE

**目标**：创建适配器层封装 ChipDiffusion 模型。

**关键实现**：
- `adapter.py`：`ChipDiffusionAdapter`
  - `load_benchmark(name)` — 通过 ChipDiffusion 数据管线加载网表
  - `sample(cond, num_steps, guidance_config)` — 标准采样
  - `predict_x0(x_t, cond, t)` — 单步 x₀ 预测
  - `denoise_from(x_start, cond, start_timestep, num_steps)` — **关键方法**：从任意中间状态恢复去噪
  - `decode_placement / encode_placement` — 坐标转换
- `sampling.py`：带验证器钩子的采样循环

**关键技术挑战**：
- **噪声级别映射**：重噪声 alpha 到扩散 timestep 的映射。ChipDiffusion 使用 cosine schedule：`alpha_bar_t = cos²(πt/2)`。重噪声后需找到对应 `t_renoise` 使去噪器看到一致输入。
- **混合噪声级别**：选择性重噪声后，部分宏是干净的、部分有噪声。GNN 训练时所有宏同一噪声级别。处理方案：(a) 使用所有重噪声宏的最大噪声级别作为重启 timestep；(b) 保持噪声级别温和。

**验收标准**：
- Adapter 产出与 ChipDiffusion 原始 eval pipeline 一致的结果
- `denoise_from` 在 t=0 返回输入（no-op），t=T 返回合理采样

---

### Phase 4: Closed-Loop Controller + Variant A ✅ DONE

**目标**：实现完整 VSR-Place 闭环和 Variant A。

**关键实现**：
- `vsr_loop.py`：`VSRLoop(adapter, verifier, selector, strength, budget, frequency)`
  - `run(cond) -> (placement, feedback_history, metrics)`
  - 两种模式：**intra-sampling**（去噪过程中验证）和 **post-sampling**（完整去噪后验证再重噪声）
- `budget.py`：`RepairBudget(max_loops, early_stop_on_legal, min_improvement_threshold)`
- `mask_only.py`（Variant A）：仅用违反严重度选择宏 → 重噪声 → 不修改模型条件特征

**验收标准**：
- VSR-Place Variant A 在 ICCAD 基准上比 ChipDiffusion unguided 提高合法性
- 合成数据上闭环收敛到合法布局
- 预算耗尽和早停逻辑正确

---

### Phase 5: Baselines & Evaluation Pipeline 🔄 IN PROGRESS

**目标**：跑全部基线，构建完整对比管线。
**进度**：
- v1.61 合成数据：VSR-Place (177.20) 超越 baseline (184.30)，-3.8%。RePaint + α=0.08 + denoise=200 + budget=8。
- ISPD2005 baseline（Colab H100 跑出）：8 电路，avg legality=0.986, avg HPWL ratio=0.748。
- **下一步**：在 ISPD2005 上跑 VSR-Place，目标改善 adaptec2 的 8.3% violation。
- **注意**：IBM ICCAD04 不可用（官方源离线），只用 ISPD2005。

**基线**：
1. ChipDiffusion (unguided)
2. ChipDiffusion (guided)
3. ChipDiffusion (guided + legalizer)
4. VSR-Place (ours)
5. VSR-Place + legalizer

**指标**：
- 约束指标：合法性/通过率、违反计数、重叠面积、边界违反数
- 质量指标：HPWL、归一化 HPWL
- 效率指标：运行时间、额外验证器调用次数、修复循环数

**验收标准**：
- 基线数字与 ChipDiffusion 公布结果一致（合理方差内）
- VSR-Place 结果在 3 个种子上可复现
- 至少 ICCAD04 完整结果表

---

### Phase 6: Ablation Suite

**目标**：运行 proposal 中定义的全部消融实验。

**消融维度**：
| 维度 | 变体 |
|------|------|
| 重噪声范围 | global / selective / top-k (3,5,10) / threshold (0.01,0.05,0.1) |
| 重噪声强度 | fixed (0.1,0.3,0.5,0.7) / severity-adaptive |
| 验证器信号 | scalar / object-wise / object-wise + pairwise |
| 修复频率 | every step / every k steps / late-stage only |
| 修复预算 | 1 / 2 / 4 / 8 loops |
| 约束集 | boundary-only / overlap-only / both / both+spacing |

**验收标准**：所有消融实验完成，可明确回答 H1-H4

---

### Phase 7: Extensions (Variant B/C)（延伸目标）

- Variant B：将逐宏严重度 r_i 追加到 GNN 节点特征
- Variant C：轻量级 GNN 编码违反图，产出嵌入加到 ChipDiffusion 内部表示

### Phase 8: ISPD2005 + Final Results（延伸目标）

- 扩展到 ISPD2005 基准
- 产出论文级别结果表和图

---

## Dependency Graph

```
Phase 0 (Scaffolding)
  ├──> Phase 1 (Verifier)     ──┐
  ├──> Phase 2 (Re-noising)   ──┼──> Phase 4 (Loop + Variant A) ──> Phase 5 (Eval)
  └──> Phase 3 (Adapter)      ──┘                                     ├──> Phase 6 (Ablations)
                                                                       ├──> Phase 7 (Ext B/C)
                                                                       └──> Phase 8 (ISPD)
```

Phase 1/2/3 **可并行**开发。Phase 4 依赖三者全部完成。

---

## Risks

| 风险 | 严重度 | 缓解措施 |
|------|--------|----------|
| ChipDiffusion API 不支持中间状态恢复去噪 | 高 | 用模型 forward pass + scheduler 参数自建去噪循环 |
| 混合噪声级别输入导致 GNN 性能下降 | 高 → **已触发** | 首轮实验 VSR violations 比 baseline 高 10-12%。需增加 denoise_steps 或改用全局 re-denoise |
| 选择性重噪声反而损害 HPWL | 中 → **已触发** | re-denoise 50步不够恢复。待测 100/200 步 |
| 验证器性能瓶颈 | 中 | GPU 矢量化实现；仅每 k 步验证；提前 profiling |
| 基准数据格式不兼容 | 中 | 复用 ChipDiffusion 自带解析工具；优先 ICCAD04 |
| 改进边际或为负 | 中 | 研究风险；消融帮助诊断；可报告负面结果 |

## Rollback

- 每个 Phase 独立 branch，合并前验证
- Phase 0 的 ChipDiffusion submodule pin 到特定 commit
- 所有实验配置化，参数更改不影响代码
- `log.md` 完整记录所有决策和假设

---

## Acceptance Criteria (Overall)

1. VSR-Place Variant A 在 ICCAD 基准上比 ChipDiffusion guided 提高合法性/通过率
2. 选择性重噪声比全局重噪声更高效（H2）
3. 结构化反馈比标量惩罚更有效（H1）
4. 无需重训练 backbone 即可获得可测量的提升（H4）
5. 所有实验通过配置文件 + 固定种子可复现
6. `log.md` 完整记录开发过程

---

## Verification Plan

1. **单元测试**：`pytest tests/` 覆盖 verifier、selector、renoise 核心逻辑
2. **集成测试**：闭环在合成数据上收敛
3. **基线复现**：ChipDiffusion 结果与公布数字一致
4. **实验复现**：固定种子，3 次运行结果一致
5. **可视化检查**：布局可视化确认合法修复效果

---

## Deliverables

- 可运行的 VSR-Place 代码（Variant A）
- ICCAD04 全套基准实验结果（5 种方法 × 多个电路 × 3 种子）
- 完整消融实验结果
- 论文级结果表和可视化图
- `log.md` 开发日志
