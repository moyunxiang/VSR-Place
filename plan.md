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

## 当前在做 — Phase 4（GPU #2）

**G1 component ablation**：6 circuits × 3 seeds × 8 variants ≈ 144 runs ≈ 30 min A800

8 variants：
- `full` — VSR-post λ=2 (control)
- `binary_mask` — boolean mask 替代 severity-soft
- `no_overlap` / `no_boundary` — verifier 信号去其一
- `no_attract` / `no_repulsive` — force 项去其一
- `random_select` — 随机 K macros
- `uniform_select` — 全 macros 同样 force

**目标**：reviewer 问 "structured feedback necessary?" 时 directly answer with table。

预算：~¥15。脚本 `scripts/run_component_ablation.py` 已写 + 本地 mock preflight 通过。

---

## Phase 5 — Final（未做）

- [ ] 把 G1 component ablation 结果 import 进 §4.7
- [ ] 内部 review 一轮（你 / 同事）
- [ ] NeurIPS abstract registration（早于 paper deadline）
- [ ] 上传 OpenReview + 完整 supplement

---

## 仓库当前状态

```
paper/main.tex          ← 9 page content + 2 page refs (NeurIPS 2026 sty)
paper/main.pdf          ← 编译输出，0 undefined refs
paper/supplement.tex    ← 7 sections
paper/figures/*.pdf     ← 9 figures
results/ispd2005/*.json ← 全部 GPU 实验数据
results/vsr_extra/*     ← Phase 2 local analysis
results/toy/*           ← toy 2D 全量
scripts/                ← 18+ 脚本，含 preflight + bootstrap
```

GitHub: https://github.com/moyunxiang/VSR-Place（当前 HEAD `9522efd`）

---

## AutoDL 当前状态（Phase 4 GPU run）

- Host: connect.nma1.seetacloud.com:54595
- A800 80GB confirmed
- Bootstrap 流程同 `scripts/bootstrap_autodl.sh` + `push_to_autodl.sh`
- 配置预算：~¥30（含 buffer）
