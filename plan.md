# NeuralVSR — 2-Week Sprint Plan (NeurIPS 2026)

> Pivot from original VSR-Place. Details in `proposal_v2.md`.
> Prior achievements archived below for reference.

## Goal

投稿 **NeurIPS 2026 Main Track**。在 2 周内完成 NeuralVSR 全套实验（训练 + ISPD2005 评估 + 消融 + 对比 baseline + 论文写作）。

## Non-Goals

- 不重训 ChipDiffusion backbone
- 不追求 SOTA HPWL（只要 VSR 不恶化 HPWL 即可）
- 不支持 IBM ICCAD04（数据不可用）
- 不做 Variant B/C（原 proposal 里的延伸目标）

---

## 核心论文故事

**标题**：NeuralVSR: Amortized Verifier-Guided Repair for Diffusion-Generated Placements

**一句话贡献**：用一个 50K 参数的 GNN 替代 20000 步的 gradient descent legalization，zero-shot 泛化到真实 ISPD2005 电路，违规减少 37-62%。

---

## 2 周时间表

### Week 1: Core Implementation + Training

#### Day 1-2（周一、周二）: GNN 架构 + 合成数据

**目标**：写好 `NeuralVSR` 模型和合成训练数据生成器。

**任务**：
- [ ] `src/vsr_place/neural/model.py`：3 层 GAT，输入 `(x, node_features, edge_index, edge_features)`，输出 per-macro displacement
- [ ] `src/vsr_place/neural/dataset.py`：合成违规电路生成器
  - 参考 ChipDiffusion 的 `data-gen/generate.py` 但更快（纯 Python，无 diffusion）
  - 随机 macro 大小（log-normal）、随机网表、随机放置
  - 目标：1M 训练对（`(x_bad, V(x_bad), x_good)`），4090 上 1 小时生成
- [ ] `src/vsr_place/neural/train.py`：训练循环（PyTorch），L2 loss 预测 `Δx = x_good - x_bad`
- [ ] 本地小样本测试（100 样本，验证 loss 下降）

**产出**：跑得通的训练脚本。

#### Day 3-4（周三、周四）: 训练 + 调参

**目标**：训出一个能用的 NeuralVSR。

**任务**：
- [ ] 租 RTX 4090（AutoDL 无卡模式生成数据 + 有卡模式训练）
- [ ] 生成 1M 训练对（约 5h）
- [ ] 训练 3-5 个配置（GNN 深度、隐藏维度、迭代次数）
- [ ] 在合成验证集上对比 NeuralVSR vs hand-crafted repulsive force
  - **Pass criterion**：NeuralVSR 在合成数据上比 hand-crafted 好 ≥10%
- [ ] 选出最佳 checkpoint

**产出**：`checkpoints/neural_vsr_best.pt`（~200KB）

#### Day 5-6（周五、周六）: ISPD2005 首次验证

**目标**：证明 NeuralVSR 在真实电路上能 work。

**任务**：
- [ ] 在 adaptec1, adaptec3, bigblue1 上跑（这些 24GB 能跑）
- [ ] 对比：
  1. Guided baseline（已有结果）
  2. Hand-crafted repulsive force（已有结果：-37~-62%）
  3. **NeuralVSR（新）**
- [ ] **Pass criterion**：NeuralVSR ≥ hand-crafted 或差 <5%
- [ ] 记录结果，如果 NeuralVSR 明显差则回 Day 3 调参

**产出**：`results/neural_vsr/ispd2005_small.json`

#### Day 7（周日）: 租 A100，跑剩余 5 电路

**目标**：补齐 ISPD2005 full table。

**任务**：
- [ ] 租 **A100 80GB** on AutoDL（~¥30/h × 8h = ¥240）
- [ ] 跑 adaptec2, adaptec4, bigblue2, bigblue3, bigblue4 的 guided baseline + NeuralVSR
- [ ] 3 种子（42, 123, 300）
- [ ] **Pass criterion**：至少 5/8 电路有改善

**产出**：完整 ISPD2005 表格。

---

### Week 2: Ablations + Baselines + Writing

#### Day 8-9（下周一、二）: 消融实验

**任务**：
- [ ] GNN 深度消融：1 / 2 / 3 / 5 层
- [ ] Hidden dim 消融：16 / 32 / 64 / 128
- [ ] 迭代次数 K：1 / 5 / 10 / 20
- [ ] 训练数据分布消融：matched vs mismatched
- [ ] No-violation-features ablation（证明 verifier 信号重要）
- [ ] **Pass criterion**：找到 1-2 个有趣的 finding 写进 paper

**产出**：`results/neural_vsr/ablations.json`

#### Day 10（下周三）: Baseline 对比

**任务**：
- [ ] 跑 DREAMPlace（标准 EDA baseline）on ISPD2005
- [ ] 报 ChipDiffusion + their 20000-step legalizer（论文里的数字，直接引用 + 复跑 1-2 个电路验证）
- [ ] 制作最终对比表（5 baselines × 8 circuits × violations/HPWL/time）

**产出**：Table 1 的最终数据。

#### Day 11-12（下周四、五）: 论文撰写

**任务**：
- [ ] Introduction（1.5 page）
- [ ] Related Work（1 page）
- [ ] Method（2 pages）：架构图 + 训练流程 + 推理流程
- [ ] Experiments（2-3 pages）：主表、消融、生成-到-真实泛化
- [ ] Discussion + Broader Impact（0.5 page）
- [ ] 画图：架构、收敛曲线、布局可视化、违规热图

**产出**：9-page NeurIPS draft

#### Day 13（下周六）: Buffer

**用途**：
- 补跑失败的实验
- 改图
- 内部 review 一轮

#### Day 14（下周日）: 投稿

**任务**：
- [ ] 最终 review
- [ ] 上传 OpenReview
- [ ] Supplementary（code + checkpoint + extra ablations）

---

## 硬件需求明细

### 训练阶段（Week 1 Day 3-4）

| 资源 | 规格 | 用途 | 时间 | 费用 |
|------|------|------|------|------|
| GPU | **RTX 4090 24GB** | GNN 训练 | 10h | ¥25 |
| CPU | 任意 | 合成数据生成（无卡模式） | 5-10h | ¥1-5 |

**AutoDL 镜像**：PyTorch 2.6 / CUDA 12.8 / Python 3.10（同之前）

### ISPD2005 评估阶段（Week 1 Day 5-7 + Week 2 消融）

| 资源 | 规格 | 用途 | 时间 | 费用 |
|------|------|------|------|------|
| GPU | **RTX 4090 24GB** | 小电路评估 + 消融 | 15h | ¥38 |
| GPU | **A100 80GB** | 大电路（adaptec2/4, bigblue2/3/4） | 8h | ¥240 |

**关键决策**：Week 1 Day 7 必须租到 A100，否则 5/8 电路跑不出来。

### 总预算

| 项 | 费用 |
|----|------|
| GPU 租用 | ¥303 |
| 数据存储 / 传输 | ¥20 |
| 缓冲 | ¥100 |
| **合计** | **¥423** |

---

## 关键文件清单

### 新增代码

| 文件 | 说明 |
|------|------|
| `src/vsr_place/neural/__init__.py` | 模块入口 |
| `src/vsr_place/neural/model.py` | NeuralVSR GNN 架构 |
| `src/vsr_place/neural/dataset.py` | 合成数据生成 + Dataset 类 |
| `src/vsr_place/neural/train.py` | 训练循环 |
| `src/vsr_place/neural/infer.py` | 推理/替换 `local_repair.py` |
| `scripts/train_neural_vsr.py` | 训练入口 |
| `scripts/eval_neural_vsr.py` | 评估入口 |
| `tests/test_neural_model.py` | 模型单元测试 |

### 修改的代码

| 文件 | 改动 |
|------|------|
| `scripts/run_vsr.py` | 增加 `--neural-repair <ckpt>` 参数 |
| `src/vsr_place/renoising/local_repair.py` | 保留作 baseline，新增 `neural_repair_loop` |

### 产出文档

| 文件 | 说明 |
|------|------|
| `docs/paper_results.md` | 更新为 NeuralVSR 主结果 |
| `paper/main.tex` | NeurIPS 论文 LaTeX |
| `paper/figures/` | 所有图 |
| `checkpoints/neural_vsr_best.pt` | 训练好的模型 |

---

## 验收标准（每日 gate）

每个阶段有明确通过条件，不满足就回退或调整：

| Gate | 标准 | 不满足的话 |
|------|------|----------|
| Day 2 end | 训练脚本跑通 | 简化 loss / 更小数据 |
| Day 4 end | 合成 val 上 NeuralVSR > hand-crafted | 换架构（MLP？Graph Transformer？） |
| Day 6 end | ISPD2005 上 NeuralVSR ≥ hand-crafted - 5% | 加训练数据 / 微调真实电路 |
| Day 7 end | ≥5/8 电路有改善 | 接受当前结果，focus on 小电路故事 |
| Day 9 end | ≥3 个消融发现 | 挑能讲的故事 |
| Day 12 end | 论文 draft 完成 | buffer day 补 |

---

## 风险与备选方案

| 风险 | 应对 |
|------|------|
| NeuralVSR 训不好（比 hand-crafted 差） | 备选 A：保持 hand-crafted，把 ML 故事改为"learned noise schedule for verifier-guided sampling"；备选 B：换用 diffusion 模型做 repair |
| A100 租不到 | 用 RTX 4090 能跑的 3 个电路 + 报告 OOM 限制，作为 honest limitation |
| 训练数据生成慢 | 用更小的合成电路（100 macros 而非 500） |
| HPWL 退化严重 | 加 HPWL-aware loss term（还是能写进论文） |
| 投稿日期不对 | 如果 NeurIPS 2026 deadline 已过，转 ICLR 2027（2026 年 9-10 月 deadline） |

---

## 回滚策略

如果 Day 6 NeuralVSR 明显比 hand-crafted 差：
- **Plan B**：把论文重新包装为 **"Verifier-Signal Aware Diffusion Guidance"**，用当前的 hand-crafted repulsive force 作为一种 instance，同时实现另外 2-3 种 instance（比如 learned, gradient-based, MPC-based），做比较。这样论文贡献变成**框架 + 多实例对比**而非单个学习方法，也能发。

---

# ───────────────────────────────────────
# 以下为原 VSR-Place 计划（归档）
# ───────────────────────────────────────

## 原 Plan 成果总结

已完成（作为 NeuralVSR 的基础设施）：
- ✅ Phase 0: 仓库搭建 + ChipDiffusion submodule
- ✅ Phase 1: Verifier (geometry, feedback, main verifier class)
- ✅ Phase 2: Selective Re-noising 引擎（hand-crafted）
- ✅ Phase 3: ChipDiffusion Backbone Adapter（含 guided sampling）
- ✅ Phase 4: Closed-Loop Controller + Variant A
- ✅ Phase 5.1: timestep 修复 + RePaint 集成
- ✅ Phase 5.2: ISPD2005 数据管线 + 首次评估
- ✅ 发现 hand-crafted post-processing 有效（-37~-62%）
- ⚠️ 发现 RePaint re-denoise 方法与 guided 不兼容

## 原 VSR-Place 实验成果

### v1.61 合成（unguided）
- Baseline: 184.30 violations
- VSR-Place: 177.20 (-3.8%)

### ISPD2005（guided + hand-crafted post-proc）

| 电路 | Baseline | VSR | 改善 |
|------|----------|-----|------|
| adaptec1 | 19,985 | 8,405 | **-57.9%** |
| adaptec3 | 42,205 | 16,375 | **-61.2%** |
| bigblue1 | 11,157 | 6,996 | **-37.3%** |

这些数字是 NeuralVSR 的对照基线。如果 NeuralVSR 做不到这个水平，就用 Plan B 回滚。
