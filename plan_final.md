# Final Sprint Plan: Framework Paper + Double Submission

**Date**: 2026-04-22
**Strategy**: Route 1 (framework paper) × Double submission (NeurIPS main + ICCAD/DAC)
**Core result**: Hand-crafted HPWL-aware VSR achieves -50.3% violations with -1.5% HPWL on 6 ISPD2005 circuits (3 seeds).

---

## Paper Narrative

**Title**: *"Verifier-Guided Constraint Projection: A Framework for Post-Processing Diffusion-Generated Placements"*

**Key contributions**:
1. **Framework** formalizing post-diffusion constraint repair:
   - Verifier produces structured violation feedback (per-macro severity + pairwise overlap graph)
   - Selective repair operator modifies only offending macros
   - HPWL-aware formulation preserves wirelength via netlist attraction
2. **Hand-crafted instance** (repulsive-attractive physics):
   - 100 iterations of force integration
   - Single `hpwl_weight` hyperparameter controls violation-HPWL Pareto
   - Simple (~50 LOC), fast (~1.5s/circuit), universal (works on any macro count)
3. **Neural instance attempts** (negative result, methodology contribution):
   - 6 training strategies (synthetic teacher, real teacher, trajectory distillation, GT legal, residual learning)
   - All fail to beat hand-crafted → insight: constraint+HPWL optimization requires >>15 samples
   - Discussion: amortization limits in low-data regimes
4. **Empirical study**: ISPD2005 (6/8 circuits), 3 seeds each, full Pareto sweep

---

## Timeline (14 days = 2 weeks)

### Days 1-3: Finalize experiments (cheap CPU work + some A800)

**Must have**:
- [ ] DREAMPlace baseline (install + run on 6 circuits) — industry standard comparison
- [ ] Try bigblue2/bigblue4 with memory optimizations (gradient checkpointing / fp16):
  - Target: get 2 more circuits if possible
  - If impossible: report as documented limitation
- [ ] Run multiple `hpwl_weight` values at 3 seeds each (Pareto table): w ∈ {0, 0.25, 0.5, 1, 2, 4}
- [ ] Measure ChipDiffusion's own 20,000-step legalizer on same circuits (compute baseline)

**Nice to have**:
- [ ] Longer hand-crafted: 500 iters vs 100 (ablation)
- [ ] Top-k offender mask vs all offenders (ablation)

### Days 4-5: Figures + Tables

- [ ] Figure 1: Framework diagram (verifier → selector → repair operator)
- [ ] Figure 2: Placement visualizations (before/after on 2-3 circuits)
- [ ] Figure 3: Pareto curve (violations vs HPWL for different hpwl_weight)
- [ ] Figure 4: Per-circuit improvement bar chart
- [ ] Figure 5: Convergence curves (violation/HPWL vs repair iteration)
- [ ] Table 1: Main results (6 circuits × metrics × 3 methods)
- [ ] Table 2: Timing comparison vs ChipDiffusion legalizer + DREAMPlace
- [ ] Table 3: Neural VSR attempts ablation (negative results)

### Days 6-9: NeurIPS paper writing

- [ ] Abstract (0.5 page)
- [ ] Introduction (1.5 pages) — problem, gap, contributions
- [ ] Related Work (0.75 page) — diffusion, constraint satisfaction, placement
- [ ] Method (2 pages) — framework + hand-crafted + neural attempts
- [ ] Experiments (2.5 pages) — setup, main results, ablations
- [ ] Discussion (0.75 page) — limitations, why learning fails here
- [ ] Broader Impact + Refs

### Days 10-12: ICCAD/DAC adaptation

- [ ] Rewrite for EDA audience (less ML background, more EDA rigor)
- [ ] Add ChipDiffusion's legalizer timing comparison prominently
- [ ] Strengthen placement metrics discussion
- [ ] 6-page format compliance

### Days 13-14: Polish & Submit

- [ ] Internal review
- [ ] Supplementary materials (code, checkpoints, extra ablations)
- [ ] NeurIPS submission (OpenReview)
- [ ] ICCAD/DAC submission

---

## Immediate Next Actions (Today)

1. **Install DREAMPlace on AutoDL A800** (DREAMPlace is from UCSD, open-source)
2. **Write helper script `run_all_final_experiments.py`** that runs:
   - All 6 circuits × 3 seeds × 6 hpwl_weight values = 108 runs (each ~5s) = ~10 min
   - ChipDiffusion legalizer on 6 circuits × 1 seed (long, ~30 min)
3. **Try memory optimizations for bigblue2/4** (best-effort):
   - `torch.utils.checkpoint` on ChipDiffusion's att_gnn blocks
   - fp16 casting during guided sampling
4. **Generate visualization figures** (placement plots before/after)

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| DREAMPlace install fails on AutoDL | Use literature-reported numbers, cite clearly |
| bigblue2/4 still OOM after memory opt | Report as "6/8 circuits, limitation due to diffusion model's attention complexity" |
| NeurIPS reviewers reject due to "no strong ML contribution" | Lean into methodology paper framing; emphasize generality of framework |
| Paper writing time overrun | Cut non-essential ablations; focus on main table + figure quality |

---

## Success Criteria

- **Minimum**: Solid ICCAD/DAC submission with 6-circuit Pareto study (expected outcome: 70% acceptance)
- **Stretch**: NeurIPS main track submission competitive (expected outcome: 30-40% acceptance)
- **Bonus**: bigblue2/4 added to results via memory tricks
