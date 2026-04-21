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

#### Day 1-2: GNN 架构 + 合成数据 ✅ 完成

**任务完成情况**：
- [x] `src/vsr_place/neural/model.py`：3 层 GATv2, hidden=128, heads=4, **103K params**
- [x] `src/vsr_place/neural/dataset.py`：合成违规电路生成器（log-normal sizes, cond_binomial edges）
- [x] `src/vsr_place/neural/train.py`：训练循环（AdamW, cosine LR, L2 loss）
- [x] 本地 smoke test 通过（50 样本 2 epochs）
- [x] 新增 8 个 unit tests，57/57 全通过

**产出**：`src/vsr_place/neural/{model.py, dataset.py, train.py, infer.py}`

#### Day 3-4: 训练 + 调参 ⚠️ 部分完成

**任务完成情况**：
- [x] 租 RTX 4090D（connect.westc.seetacloud.com）
- [x] 多轮训练迭代：v1 (legal target), v2 (teacher, 2000 合成样本), v5 (scale-invariant, 50 样本)
- [x] **Teacher distillation 工作**：val loss 从 0.22 降到 8e-6（v2）
- [x] 加入 scale-invariant feature normalization
- [ ] ~~Pass criterion：NeuralVSR 在合成 val 上好 ≥10%~~ — **失败于 ISPD2005 上泛化**

**产出**：6 个训练好的 checkpoints（`checkpoints/neural_vsr/{v1_2k, v2_teacher, v5_norm, real_v1, real_traj_v1, best}.pt`）

#### Day 5-6: ISPD2005 首次验证 ❌ Pass criterion 失败

**任务完成情况**：
- [x] 在 adaptec1 上跑评估脚本
- [x] 对比 guided baseline / hand-crafted / NeuralVSR
- [ ] ~~Pass criterion：NeuralVSR ≥ hand-crafted - 5%~~ — **所有变体都比 baseline 更差**

**实际结果**（adaptec1，单种子）：

| 方法 | Violations | vs baseline |
|------|-----------|-------------|
| Guided baseline | 18,700 | — |
| Hand-crafted | 8,400 | **-55%** ✅ |
| NeuralVSR v2 (纯合成) | 23,150 | -23% ❌ |
| NeuralVSR v5 (scale-invariant 合成) | 20,741 | -12% ❌ |
| NeuralVSR real_v1 (3 真+扰动) | 34,512 | -84% ❌ |
| NeuralVSR real_traj_v1 (3 真×30 traj) | 20,771 | -10% ❌（K=10, 最好） |

**根因分析**：
1. v1-v5 合成训练：sim-to-real gap 无法闭合（合成分布 != ISPD2005 分布）
2. real_v1：3 个真样本 + 扰动增强，增强后多样性仍不够
3. real_traj_v1：轨迹蒸馏（每样本 30 步 = 90 pairs），但 3 电路 × 1 = 只见过 adaptec1
4. **核心瓶颈：24GB 只能稳定生成 3 个真实 guided placements**，其余 OOM

**产出**：`results/neural_vsr/eval_jsons/*.json`（所有 K ablation 结果）

#### Day 7: 租 A100，跑剩余 5 电路 + 重训 🔴 待做（关键）

**必须用 A100 80GB 的原因**：
- 24GB 对 guided sampling 的 5 个电路（adaptec2/4, bigblue2/3/4）完全 OOM
- 24GB 甚至无法稳定跑 adaptec3/bigblue1 的多种子（累积 OOM）
- NeuralVSR 训练需要 ≥50 真实样本，24GB 拿不到

**Day 7 任务**：
- [ ] 租 A100 80GB（~¥30/h × 4-6h = ¥120-180）
- [ ] 生成 8 电路 × 8 种子 = **64 个真实 guided placements**
- [ ] 用轨迹蒸馏扩成 64 × 30 = **1920 training pairs**
- [ ] 重训 `real_traj_v2`（充分数据版本）
- [ ] 在全 8 电路上评估 baseline / hand-crafted / NeuralVSR（3 seeds 各）
- [ ] **Pass criterion**：NeuralVSR ≥ hand-crafted - 5%（跨 ≥5 电路）

**如果 Day 7 仍然失败 → 启动 Plan B**（见文末回滚策略）

---

### A100 开机前的 Pre-flight Checklist（**全部 ✅**）

在租 A100 之前所有不需要 A100 的工作都已完成：

- [x] **代码完整**：model / dataset / train / infer / eval 全部就绪
- [x] **Hand-crafted baseline 已验证**：-37~-62% on 3 个能跑的电路
- [x] **训练管线已调通**：teacher distillation + trajectory distillation 两种 target，都能让 loss 收敛
- [x] **Scale-invariant 归一化已加入**：model 支持任意 canvas 大小
- [x] **Eval 脚本完整**：`scripts/eval_neural_vsr.py` 三方对比
- [x] **ISPD2005 数据管线就绪**：`parse_ispd2005.py` 可一键下载+解析
- [x] **数据备份**：73MB checkpoint + 104MB ISPD tarball 在本地缓存，scp 直接给 A100
- [x] **SSH 恢复流程**：`docs/autodl_ssh_setup.md` + `docs/restore_autodl_env.md`
- [x] **现有 trained checkpoints**：6 个 NeuralVSR 变体已存 git，方便对比

### A100 上只需要做的事（串行，**~4-6h**）

```bash
# 1. 配 SSH（0.5h）
# 2. clone + deps + restore caches（0.5h）
scp local_artifacts/large-v2.ckpt autodl:/root/autodl-tmp/VSR-Place/checkpoints/large-v2/
scp local_artifacts/ispd2005dp.tar.xz autodl:/tmp/
# extract + parse（~5min）

# 3. 生成完整真实数据（1-1.5h）
python scripts/gen_ispd_placements.py \
    --checkpoint checkpoints/large-v2/large-v2.ckpt \
    --circuits 0 1 2 3 4 5 6 7 \
    --seeds 0 1 2 3 4 5 6 7 \
    --output data/ispd_placements_full.pkl
# 预期 64 样本 × ~17s = 18min 实际 guided sampling + 开销 = 30min

# 4. 训练（0.5h）
python scripts/train_neural_vsr_real.py \
    --data data/ispd_placements_full.pkl \
    --output checkpoints/neural_vsr/real_traj_v2.pt \
    --epochs 100 --batch-size 8 --trajectory-steps 30

# 5. 评估（1-2h）
python scripts/eval_neural_vsr.py \
    --checkpoint checkpoints/large-v2/large-v2.ckpt \
    --neural checkpoints/neural_vsr/real_traj_v2.pt \
    --circuits 0 1 2 3 4 5 6 7 \
    --seeds 42 123 300
# 24 (circuit, seed) 对 × ~30s = 12min + 开销

# 6. Commit + push 结果, 关机
```

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
