# VSR-Place: Paper-Level Experimental Results (updated 2026-04-21)

---

## Headline Result

**Hand-crafted VSR-Place reduces ISPD2005 placement violations by 37–85% over ChipDiffusion guided baseline on 6 of 8 circuits (3 seeds each).**

## Experimental Setup

| Item | Value |
|------|-------|
| Backbone | ChipDiffusion Large+v2 (6.29M params, official checkpoint) |
| Guidance | `opt` mode (Adam-based constrained optimization, 20 grad steps) |
| Diffusion steps | 1000 |
| VSR method | Post-processing `local_repair_loop` (repulsive force on offending macros) |
| Repair steps | 100 iterations, step_size=0.3 |
| Dataset | ISPD2005 Bookshelf (macro-only) |
| Hardware | NVIDIA A800 80GB PCIe |
| Seeds | 42, 123, 300 |

## Main Results Table (averaged over 3 seeds)

| Circuit | #Macros | Baseline violations | VSR violations | **Improvement** |
|---------|---------|---------------------|-----------------|-----------------|
| adaptec1 | 543 | 19,460 | 8,761 | **-55.0%** |
| adaptec2 | 566 | 36,016 | 11,329 | **-68.5%** |
| adaptec3 | 723 | 39,763 | 16,397 | **-58.8%** |
| adaptec4 | 1,329 | 109,565 | 36,691 | **-66.5%** |
| bigblue1 | 560 | 11,187 | 7,063 | **-36.9%** |
| bigblue3 | 1,298 | 208,926 | 30,566 | **-85.4%** |
| bigblue2 | 23,084 | — | — | OOM on 80GB (too many edges) |
| bigblue4 | 8,170 | — | — | OOM on 80GB |

**Coverage**: 6 of 8 ISPD2005 circuits (3 seeds × 6 = 18 runs, all successful).
**Average improvement across 6 circuits**: **-61.9%** violations.

## NeuralVSR Status ⚠️

4 training variants attempted, none beat hand-crafted:

| Variant | Training data | Val loss | ISPD2005 best result |
|---------|---------------|----------|----------------------|
| v2 (synthetic teacher) | 2000 random circuits | 0.001 | -23% (worse) |
| v5 (scale-invariant) | 50 synthetic | 8e-6 | -12% (worse) |
| real_v1 (3 real + perturb) | 3 × 50 aug = 153 | 0.069 | -57% (worse) |
| real_traj_v1 (3 real × 30 traj) | 90 pairs | 0.004 | -10% (worse) |
| real_traj_v2 (15 real × 30 traj) | 450 pairs | 0.035 | **-17% (worse)** |

**Diagnosis**: training loss always converges, but does not generalize to ISPD2005. Root cause is data distribution mismatch; even 15 real placements from 4 circuits (best we could get on A800 before OOM on larger circuits) are insufficient.

## Key Technical Insights

1. **Diffusion model guidance still leaves 37–85% residual violations**. Guided sampling alone is insufficient.
2. **Post-processing with a simple repulsive-force model recovers most of this**. Hand-crafted method is:
   - Simple (~50 lines of code)
   - Fast (~1–2 seconds per circuit)
   - Universal (works on every circuit size we tested)
3. **Amortized learning via a small GNN is harder than expected**. Even with ISPD2005-distribution training data, the model fails to generalize. Our diagnosis points to data quantity (≤15 training placements) as the bottleneck, not the method itself.
4. **OOM on A800 80GB** is a real constraint for ChipDiffusion's guided sampling on very large circuits (bigblue2 with 23K macros, bigblue4 with 8K macros). Future work needs multi-GPU or gradient checkpointing.

## Limitations

- **bigblue2 / bigblue4** results unavailable due to 80GB OOM during ChipDiffusion guided sampling
- NeuralVSR does not beat the hand-crafted baseline within our current training data budget
- HPWL measurement deferred (only violations reported)

## Cost Analysis

| Method | Time per circuit | Avg improvement |
|--------|------------------|---------------------|
| ChipDiffusion guided baseline (20 grad steps) | ~9s | — |
| + Hand-crafted VSR post-proc | +1.5s | **-61.9%** |

VSR post-processing adds minimal compute overhead for large violation reductions.

## Files

- Raw results: `results/ispd2005/final/a800_full.json` (6 circuits × 3 seeds)
- Older 24GB partial results: `results/ispd2005/final/all_results.json`
- NeuralVSR checkpoints: `checkpoints/neural_vsr/real_traj_v2.pt` (best of 5 variants)
- NeuralVSR training logs: `results/neural_vsr/training_logs/`
- Implementation: `src/vsr_place/renoising/local_repair.py` (hand-crafted), `src/vsr_place/neural/` (GNN)
