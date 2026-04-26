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
- [x] **NeurIPS reviewer round 1 全部回应**（见 review_response_plan.md + log Phase 5）
  - F1-F12 文本修正（abstract、claims、tone-down、limitations）
  - F4 circuit-level Wilcoxon (n=6) 替换主表
  - W2 tuned baselines: cg legality_weight + RePaint t_start sweep → supplement Table，main 段落引用
  - W2/Q5 **independent classical force-directed legalizer baseline** (FD-pure + FD+spring) → main §4.4(iii) + supplement §sec:supp-classical
  - W4 verifier-evaluation circularity 透明段落 → main §4.4
  - W5/E4 full-design HPWL with cells fixed at .pl → supplement Table，main 段落引用
  - W7 selector saturated-draft regime artifact 透明段落 → main §4.4
  - W9/Q7 no_grad OOM 源代码精确行号引用 (chipdiffusion/diffusion/models.py:1501-1550) → main Limitations + appendix §sec:supp-mem
  - F11 bigblue3 overlap-area: 进 full_metrics 表（6 circuits 全有 overlap_area / max_overlap）
  - **Suggestion 6 principled λ selection**: LOCO CV，6/6 folds 都选 λ̂=8，held-out 6/6 strict-Pareto → main §4.6 + supplement §sec:supp-lambda-loco
- [ ] **AutoDL 关机**（password policy 阻止本地 SSH，需要用户手动: web 控制台或 `ssh autodl_main 'shutdown -h now'`）
- [ ] 内部 review 一轮（你 / 同事）
- [ ] NeurIPS abstract registration（截止前）
- [ ] 上传 OpenReview + 完整 supplement
- [ ] author / affiliation block（投稿时填）

## Phase 5 round 4 — Round-2 reviewer (5/10) 闭环（commit `42082f0`）

GPU 实验全部完成（24 trials × 13 methods，30 min on AutoDL A800 80GB，~¥15 增量）：

- [x] **R1/Q1 决定性下游 pipeline**: raw→cd-sched vs vsr8→cd-sched on 24 trials → median Δv −35.9% (vs raw -28.5%) + 略低 full-HPWL → main §4.6 + supplement §sec:supp-round2
- [x] **R2/Q3/Q6 λ=8 main 4 seeds**: median Δv=-52.2%/Δh_macro=-34.1%/Δh_full=+8.9%, residual ≈18k → table_lambda8_main.tex
- [x] **R3/Q4 cg/RePaint 4 seeds**: w∈{2,8} + t∈{0.3,0.5} on 24 trials → table_baseline_sweeps_24.tex
- [x] **R4/Q5 classical FD on 24 trials**: FD-pure (-54%/+144%), FD+spring (-52%/+34%), VSR-post (-52%/-34%) → table_classical_legalizer_24.tex
- [x] **W7 inconsistencies**: §4.1 (n=24) vs §4.4 (n=6) 调和；Table 2 caption + §4.7 文案修正
- [x] **W8 RePaint-bin overclaim**: 删 "p<0.05 both metrics" claim，标注 HPWL p=0.075 透明
- [x] **W9/Q8 non-differentiable**: 全文替换为 "piecewise differentiable, vanishing gradient in inactive regions"
- [x] **W10/Q7 adaptec3 +73%**: Limitations 加专门段落 + 引 §sec:downstream_pipeline 作为 mitigation evidence
- [x] **S6 Fig 1 标签**: "Legal placement" → "Repaired placement"
- [x] **S7 strictly dominates**: §4.2 标题改 + §4.4 措辞软化

未做（reviewer 提到但工作量大）：
- [ ] multiple-comparison correction (Bonferroni/BH) — 仅在 §4.4 footnote 注明 "p-values are marginal, no correction"
- [ ] DREAMPlace / RePLACE 实测对比 — 在 Limitations 里说明输入格式不匹配 macros-only

仓库当前状态：
```
paper/main.pdf       ← 18 页（body 9 + refs + appendix 8）
paper/supplement.pdf ← 9 页
results/vsr_extra/round2_review.json (24 rows × 13 methods)
4 张新 paper figure tables (downstream_pipeline / lambda8_main / classical_legalizer_24 / baseline_sweeps_24)
GitHub HEAD: commit 42082f0
```

总成本：~¥165（前 ¥150 + round 4 ¥15）。

## Phase 5 实测 — Reviewer 回应数据
- cg-tuned median: Δv=−7.1%, Δh=+14.2%（每电路最优 hp 后仍弱于 VSR）
- RePaint-tuned median: Δv=**+109.4%**, Δh=−52.8%（HPWL "改善" 来自 macros 塌缩）
- VSR-post (λ=2) median: Δv=−55.0%, Δh=−10.9%
- 6/6 电路 VSR-post 在 Δv 上严格胜过 best-tuned cg/RePaint
- Full-design HPWL median: VSR-post +10.7%, VSR-intra-soft −1.6%；adaptec3 例外（+73%，dense cell-macro coupling 限制）

## 仓库 layout 更新
```
paper/main.pdf           ← 14 页（body 9 页 + refs + appendix）
paper/supplement.pdf     ← 7 页（新增 §"Tuned baselines" + §"Full-design HPWL"）
paper/figures/table_baseline_sweeps.tex  ← NEW
paper/figures/table_full_metrics.tex     ← NEW（4-26 早些时候）
results/vsr_extra/baseline_sweeps.json   ← 60 行（cg+repaint sweep）
results/vsr_extra/full_metrics.json      ← 24 行（full-design HPWL + overlap area）
scripts/run_baseline_sweeps.py            ← NEW
scripts/run_full_metrics.py               ← NEW（已 run）
scripts/analyze_baseline_sweeps.py        ← NEW
scripts/analyze_full_metrics.py           ← NEW
```

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
