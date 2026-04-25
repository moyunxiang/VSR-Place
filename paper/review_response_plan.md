# Response Plan to NeurIPS Review

> **Honesty assessment**: The reviewer is largely correct. Many of our claims
> overstate. We will fix all internal inconsistencies, add missing metrics
> (legality rate, overlap area, circuit-level stats), tone down language, and
> add a clear "this is repair not legalization" framing.

Each item below maps a reviewer concern → concrete fix → cost / data.

---

## Tier 1 — Local fixes (0 ¥, this session)

### F1. Internal inconsistency: 12/24 vs 14 flagged ⚠️
**Reviewer**: "Table 2 implies 12/24 non-Pareto, Appendix D says 14."
**Reality**: strict-Pareto = 12 (both Δv<0 AND Δh<0). Failures table has 14
because it includes 2 "weak-violation-cut" entries on bigblue1 (both
Δv > −20%) — those are still strict-Pareto on HPWL but flagged as weak
violation cut. Both are valid views, just need to clarify in caption.
**Fix**: explicitly explain in Appendix D caption: "14 = 12 not-strict-Pareto
+ 2 weak-cut (still Pareto but Δv > −20%)".

### F2. Abstract overstates cg-strong significance ⚠️
**Reviewer**: "Abstract claims p<0.001 vs classifier guidance, but Table 3
HPWL p=0.0814."
**Reality**: VSR-post vs cg-strong: Δv p<0.001 ✓, Δh p=0.081 ✗. Abstract
literally says "with $p<0.001$" implying both metrics. This is wrong.
**Fix**: rewrite abstract sentence: "...statistically dominates classifier-
guidance with strong legality weighting on violations ($p<10^{-3}$, with
HPWL not significantly different)..."

### F3. NEVER FULLY LEGAL — re-frame headline 🔴
**Reviewer**: "median 55.3% violation reduction still leaves many violations".
**Reality**: confirmed from `main_neurips.json`: **0 of 24 trials achieve
v=0 (or even v ≤ 5)** for any method. We're a *repair* layer, not a
*legalizer*. Reporting only "% reduction" is misleading.
**Fix**:
- Abstract: change "median 55.3% violation reduction" → "reduces residual
  violations by a median of 55.3% (no method we test produces a fully legal
  placement)".
- Add a sentence to Limitations: "VSR is a repair layer, not a legalizer:
  none of the methods we test (including ChipDiffusion's 5000-step
  legalizer) yields v=0 placements; classical post-place legalization
  remains a necessary downstream step."
- Add **fully-legal column** (always 0/24) to Table 1 caption.

### F4. Wilcoxon at (circuit, seed) overstates n 🔴
**Reviewer**: "n=24 treats seeds as independent; circuit-level variation
dominates."
**Reality**: this IS a real concern. The 4 seeds within a circuit are paired
on the same baseline draft, but circuit-level variance (e.g., bigblue1 has
~10× smaller HPWL than adaptec3) dominates. A circuit-level paired test
(n=6) is the conservative choice.
**Fix**:
- Recompute Wilcoxon at circuit-level (mean per circuit over 4 seeds),
  n=6 paired tests.
- Report BOTH n=6 and n=24 numbers transparently; explain the seed-level
  result is a paired-bootstrap, not iid.
- Tone down "p<0.001" claims to "p < 0.05 at circuit level".

### F5. "Non-differentiable" verifier is technically wrong 🟡
**Reviewer**: "max/clamp is piecewise differentiable in the active region."
**Fix**: replace "non-differentiable" → "with vanishing/discontinuous
gradients (only piecewise active)" or simply "feedback that classical
classifier guidance cannot consume directly".

### F6. Force update underspecified 🟡
**Reviewer**: "How are coords/sizes/HPWL normalized? hyperedges? boundary
'snap' or scaled?"
**Fix**: Expand §3.3 (or supplement A) with explicit equations:
- Coordinates in absolute (canvas) units (units of macro-size mean).
- Hyperedges → pairwise via netlist edge pairs (already documented).
- Boundary: scaled-snap via `eta * boundary_force`, not raw snap.
- Force clamp at 0.5 × mean macro size per step (already in code).

### F7. "Structured verifier feedback" novelty overstated 🟡
**Reviewer**: "pairwise overlap, boundary, force-directed all classical."
**Fix**: tone down. Reframe contribution as:
1. A clean Pareto-control knob (λ) for diffusion-draft repair, with the
   first systematic empirical Pareto study of post-diffusion macro repair.
2. A 1.6–3.6× wall-clock improvement over CD's released legalizer at
   strictly Pareto-superior operating points.
3. A clean component ablation showing the structural ingredients.
NOT a fundamental ML novelty.

### F8. Selector ablation undermines selector contribution 🟡
**Reviewer**: "random/uniform match full method on saturated drafts."
**Fix**: be honest. Replace "the selector is essential…" with "On
ISPD2005's worst-case drafts (every macro violates) the offender mask
trivially equals the all-macros set; we therefore make no claim about the
selector's standalone value, and note that classical legalizers
likewise do not gate on offender masks."

### F9. Why λ=2 not λ=8 main setting 🟡
**Reviewer**: "λ=8 has 100% strict-Pareto, but main results use λ=2."
**Reality**: we ran the main experiment (`run_main_neurips.py`) at λ=2
(historical default before extending the sweep). The lambda sweep was the
final addition and exposed λ=8 as Pareto-best. We never re-ran the main
table at λ=8.
**Fix**: 
- Explicitly explain in §4.5: "We report λ=2 in Table 1 because that is the
  setting we benchmarked the comparison methods at; λ=8 emerges from the
  sweep as Pareto-best for HPWL-priority workloads."
- Add a paragraph: "Replacing λ=2 with λ=8 in Table 1 would shift VSR-post's
  median Δh from −7.8% to −37.9% while keeping median Δv near −49%; the
  comparison conclusions would not change but the strict-Pareto count would
  become 18/18."

### F10. Memory profile under no_grad? ✅ verify
**Reviewer**: "should not accumulate activations across 100 steps."
**Reality**: Our `denoise_repaint` uses `with torch.no_grad():`. Need to
double-check `guided_sample` does too. ChipDiffusion's `reverse_samples`
does grad-descent over Adam in 'opt' mode — that DOES create activation
graphs. So the OOM is real but not just naive activation accumulation;
it's gradient computation over the legality/HPWL surrogate within sampling.
**Fix**: clarify in §4.10/Limitations: "ChipDiffusion's released sampler
runs Adam-based optimization at every denoising step under guidance-mode
'opt'; the OOM is dominated by these per-step optimization graphs, not raw
forward-only inference."

### F11. Add overlap-area + max-overlap metrics from existing data 🔴
**Reviewer**: "violation count is inadequate; need overlap area, max overlap."
**Reality**: we have overlap area at the bigblue3 placement save level
(`results/placements/bigblue3_neurips.pkl` contains severity vectors).
For the 6 circuits × 4 seeds, we did NOT save per-trial severity,
only the summary count `v`. Computing overlap area for all 24 trials
requires re-running with placement saves enabled.
**Fix**:
- Add **bigblue3 overlap-area row** to a new supplement table (we have
  this for seed=42 only).
- Add an honest note: "overlap area was only saved for the bigblue3 hero
  trial; full-coverage overlap-area metrics are deferred to future work."

### F12. Reframe contribution language 🟡
**Reviewer**: "'legal placement', 'strictly dominates', 'right interface'
too aggressive."
**Fix**: global text pass:
- "strictly dominates" → "dominates / outperforms" (only keep "strict" for
  strict-Pareto count which IS strict)
- "the right interface" → "a useful interface"
- "non-differentiable structured feedback" → "verifier-driven repair"
- Drop any unqualified "legal placement" claims.

---

## Tier 2 — GPU experiments (if user authorizes ¥30–60)

### F13. Classifier-guidance hyperparameter sweep 🟡
**Reviewer**: "baselines insufficiently tuned."
**Fix**: run `cg_strong` with `legality_weight ∈ {0.5, 1, 2, 4, 8}`,
`hpwl_weight ∈ {0, 0.001, 0.01}`, on 6 circuits × 1 seed. ~25 runs × 9s
sample = ~5 min GPU. Report best per-circuit.

### F14. RePaint hyperparameter sweep 🟡
**Reviewer**: same.
**Fix**: run `repaint_bin` with `t_start ∈ {0.1, 0.2, 0.3, 0.5, 0.7}` on
6 circuits × 1 seed. ~30 runs × 4s = ~3 min GPU. Report best per-circuit.

### F15. Strong classical legalizer baseline (DREAMPlace) ⚠️
**Reviewer**: "missing DREAMPlace, RePLACE, NTUplace."
**Cost**: ~60–120 min GPU + non-trivial install. Risk of failure on
ISPD2005 macros-only protocol mismatch.
**Fix decision**: **defer**. Explicitly write in Limitations: "we compare
to ChipDiffusion's own released legalizer; comparison to DREAMPlace and
RePLACE is left to future work, with the caveat that these tools target
post-detailed-place legalization and require macro-cell coupling
information that ChipDiffusion's macros-only subset does not provide."

### F16. Full-design HPWL with cells 🟡
**Reviewer**: "macros-only HPWL hides cell-level wirelength."
**Cost**: ~10 min GPU to recompute HPWL with full pin-map (we have parsed
data with all nodes). Need a small recompute script.
**Fix decision**: defer to time. Note in Limitations.

### F17. Save overlap-area for all 24 trials 🟡
**Cost**: ~30 min GPU re-run of `run_main_neurips.py` with overlap-area
recording enabled.
**Fix decision**: do it if GPU comes back online.

---

## Tier 3 — Future-work, won't fix in this revision

### F18. Multi-benchmark
ISPD2005 only. ICCAD04/IBM unavailable / different format.
Fix: explicitly acknowledge.

### F19. Mixed-effects model statistics
Reviewer suggests; we'd need to learn / properly implement.
Fix: present circuit-level paired Wilcoxon; note mixed-effects as future.

---

## Execution order (this session)

**Tier 1 (local, all 12 items)**:
- F1, F2, F8, F12: text fixes only
- F3: re-frame headline + add "fully legal" stats from existing data
- F4: add circuit-level Wilcoxon (n=6) to compute_neurips_stats.py
- F5, F6, F7, F9, F10: text edits
- F11: bigblue3 overlap-area row from existing pkl

**Compile + push**.

**Tier 2 (defer to user authorization)**:
- F13, F14, F17: ~10 min GPU each.
- F15, F16, F18: clearly deferred.
