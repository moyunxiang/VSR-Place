# Plan — VSR-Place NeurIPS 2026 投稿

> 旧计划（NeuralVSR）已弃用，见 git 历史 (commit 之前)。
> 当前路线：**Verifier-Guided Selective Repair**（hand-crafted 物理力场为主），论文已经 9 页 NeurIPS 格式编译通过。

---

## Goal

**投 NeurIPS 2026 Main Track**。剩下的工作：补完 component ablation（reviewer 必问），内部 review，定稿投稿。

## Non-Goals

- 不再追 NeuralVSR（旧路线归档）
- 不再要求支持 ICCAD04（数据不可用）
- 不试 bigblue2/bigblue4 backbone OOM 突破（已写为 limitation）

---

## 论文故事（已锁定）

**标题**：*Verifier-Guided Selective Repair for Diffusion-Generated Macro Placements*

**一句话贡献**：非可微 verifier 输出**结构化反馈**（severity scalar + 邻接 overlap graph + 边界向量），驱动 100-step 物理力场 → 在 6 ISPD2005 电路 4 seeds 上 median −55% violations + 89% strict-Pareto improvement (λ=4) + 1.6-3.6× 比 ChipDiffusion 自家 legalizer 更快。

**核心数据**（已验证）：
- VSR-post 在 24 (circuit, seed) 对中 12 个严格 Pareto 改善 baseline，CD-std 0/24
- λ=4 时严格 Pareto 飙到 16/18 (89%)
- Wilcoxon p<0.001 vs 5 个 baseline 的双指标全部碾压
- 显存证据：bigblue2/4 OOM 的瓶颈是 ChipDiffusion 自家 attention，不是 VSR

---

## 已完成（截至 2026-04-26）

### Phase 1 — GPU 实验（done）
- ✅ E1 main: 6 circuits × 4 seeds × 7 methods = 24 (circuit, seed) 全部 OK
- ✅ E4 timestep sweep: 6 × 5 × 3 = 90 rows
- ✅ E5 mem profile: 3 circuits × 3 dtypes
- ✅ AutoDL #1 关机

### Phase 2 — 本地分析（done）
- ✅ L1 lambda sweep（已塞入 §4.4，加 fig_lambda_sweep）
- ✅ L2 step budget（§4.5，加 fig_step_budget）
- ✅ L3 failure cases（§4.6）
- ✅ L4 runtime/memory consolidation
- ✅ L5 use_mask ablation
- ✅ L6 hpwl_weight=0 ablation
- ✅ Toy 2D 全量 (CPU 30 min, 5 seeds × 50 prompts × 5 methods = 1250 rows)

### Phase 3 — 论文（done）
- ✅ NeurIPS 2026 sty + 真引用，9 页 content 编译通过
- ✅ Author: Anonymous（无 email），Wilcoxon 表星号格式修好
- ✅ Table 1 / 4 / 9 全部 reformat 不溢出
- ✅ Fig 1 / 4 / 8 / 9 全部用最新数据

---

## Phase 4 — GPU #2 (done)

**G1 component ablation 完成**：6 circuits × 3 seeds × 8 variants = 144 runs

关键发现（结果在 `results/vsr_extra/component_ablation.json`）：

| 变体 | Δviol | ΔHPWL | 严格 Pareto |
|---|---|---|---|
| **full (control)** | **−50.1%** | **−5.1%** | **9/18** |
| − overlap signal | +79.2% (worse) | −85.0% | 0/18 |
| − boundary signal | −82.9% | −8.9% | 9/18 |
| − attractive force | −51.4% | +138.4% (catastrophic) | 0/18 |
| severity soft-mask | −21.8% | +43.7% | 1/18 |
| random_select / uniform_select | = full（饱和情况） | = full | 9/18 |

**结论**：
1. 移除 overlap 信号 / repulsive 力 → 违规反增 79%（结构必需）
2. 移除 attractive 力 → HPWL 翻倍 +138%（必需）
3. severity-weighted soft mask 比 binary 差 28pp（**binary mask 才对**）
4. boundary signal 在饱和 draft 上其实可省（次要发现）

塞进 main.tex §4.7 + Table 6。

成本：~¥25 × 50 min A800。

---

## Phase 5 — Final（剩余人工事项）

- [x] G1 results import 进 §4.7 (commit `21d632d`)
- [x] supplement.tex 用新 ablations 数据填实 (commit 待推)
- [x] main.tex self-review 一轮
- [ ] 内部 review 一轮（你 / 同事）
- [ ] NeurIPS abstract registration（截止前）
- [ ] 上传 OpenReview + 完整 supplement
- [ ] author / affiliation block（投稿时填）

---

## 仓库当前状态（Phase 5 起点）

```
paper/main.tex          ← 9 页正文 + 2 页 refs (NeurIPS 2026 sty)
paper/main.pdf          ← 0 undefined refs
paper/supplement.tex    ← 11 sections（含 G1 完整表 + failure-mode 列表 + toy 2D 数字）
paper/supplement.pdf    ← 5 页
paper/figures/*.pdf     ← 11 figures
results/ispd2005/*.json ← 全部 GPU 实验数据
results/vsr_extra/*     ← lambda sweep / step / failure / G1 component ablation
results/toy/*           ← toy 2D 全量 (1250 rows)
scripts/                ← 20+ 脚本
```

GitHub: https://github.com/moyunxiang/VSR-Place（最新 HEAD `21d632d`，supp update 待推）

---

## 总成本

| 阶段 | 时长 | 费用 |
|---|---|---|
| AutoDL #1（E1/E4/E5） | 3.5h × A800 | ¥105 |
| AutoDL #2（G1 component） | 50 min × A800 | ¥25 |
| 本地 CPU | — | 0 |
| **合计** | | **¥130** |

两台 AutoDL 都已关机。预算上限 ¥200，实际 65%。
