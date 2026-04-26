# NeurIPS-Style Review: Verifier-Guided Selective Repair for Diffusion-Generated Macro Placements

## 1. Summary

The paper proposes VSR, a verifier-guided repair layer for diffusion-generated macro placements. A verifier computes per-macro severity, pairwise overlap structure, and boundary protrusions; a masked force-directed operator then moves violating macros using repulsion, boundary correction, and netlist attraction, controlled by a scalar lambda. The latest version explicitly frames VSR as a repair layer rather than a complete legalizer, adds lambda=8 results, adds a downstream `VSR -> ChipDiffusion legalizer` experiment, and expands baseline comparisons.

## 2. Strengths

- The paper is now much more honest about scope: VSR is positioned as a repair/preconditioning layer, not a full legalizer.
- The added downstream experiment directly addresses the most important practical question: whether VSR helps before a legalizer.
- The lambda=8 4-seed result and LOCO selection make the lambda story more concrete than before.
- The expanded force-directed baseline on the full 24 trials is useful and shows VSR's HPWL advantage over simple repulsion/spring heuristics.
- The statistical framing is improved: circuit-level n=6 testing is now clearly described as the primary conservative test, with seed-level n=24 relegated to the appendix.
- The paper now avoids the earlier overclaim that the verifier is strictly non-differentiable; it correctly describes the signals as piecewise differentiable.

## 3. Weaknesses

- The main practical issue remains: even after VSR, the median residual violation count is still about 18k. This is far from a usable legal placement.
- The new downstream pipeline result is positive but weak. `raw -> VSR(lambda=8) -> cd-sched` improves median residual violation reduction from -28.5% to -35.9% and full-design HPWL from +8.4% to +7.1%, but the gains are small and not consistently better per circuit.
- The downstream pipeline still does not produce fully legal placements. Calling the experiment "decisive" is too strong; it only shows modest preconditioning benefit for one weak legalizer.
- The post-pipeline macro-only HPWL is actually worse with VSR preconditioning: +226.2% versus +215.7% median. The paper should not overstate the quality benefit.
- Full-design HPWL remains problematic. At lambda=8, VSR-post has median full-design HPWL +8.9%, with adaptec3 at +100.0% and adaptec4 at +30.3%.
- The main table still uses lambda=2 even though LOCO and the lambda=8 table indicate lambda=8 is the better validated setting. The historical explanation is not a scientific reason.
- Novelty remains limited. The method is a well-engineered combination of classical force-directed placement ingredients, not a fundamentally new constraint-handling method.
- Baseline comparison to mature physical-design flows remains missing. DREAMPlace/RePlAce are dismissed as not directly applicable to macros-only output, but the practical claim ultimately requires comparison to a real placement/legalization pipeline.
- The paper contains some stale or confusing material: the old seed-42 tuned-baseline sweep remains in the supplement even though a newer 24-trial sweep replaces it.
- Some language is still too strong, especially "decisive downstream pipeline experiment" and "Pareto-superior" when several conclusions depend on medians and not consistent per-circuit dominance.

## 4. Technical Soundness

The repair operator is technically plausible and now specified more clearly. The force terms, clamping, degree normalization, and piecewise-differentiability discussion are reasonable.

However, the technical contribution should be framed as a practical heuristic/preconditioner. The method does not solve legality, and the downstream legalizer experiment only partially supports the claim that VSR improves final outcomes. The strongest supported claim is: VSR can reduce overlap-related violations while preserving or improving macro-only HPWL better than simple force baselines and ChipDiffusion's released legalizer settings.

The paper still needs more caution around statistical and median-based claims. With only 6 circuits, p-values are fragile, and median improvements can hide circuit-level regressions.

## 5. Novelty & Significance

The novelty is modest. The ingredients are classical: overlap repulsion, boundary correction, netlist springs, and masking. The more novel part is the empirical study of using these as a post-diffusion repair/preconditioning interface.

The significance is improved by the downstream experiment, but not enough to make the method clearly impactful. The actual post-pipeline gains are small, and no tested pipeline reaches full legality.

## 6. Experimental Evaluation

The evaluation is now substantially stronger than the previous versions. Important improvements include:

- lambda=8 evaluated on the same 24 trials.
- force-directed baselines evaluated on the same 24 trials.
- cg/RePaint variants evaluated on the same 24 trials.
- a downstream `VSR -> cd-sched` pipeline comparison.
- residual violation counts reported.

Remaining issues:

- The downstream result should be reported more conservatively: VSR helps median violation reduction by 7.4 percentage points and median full-design HPWL by 1.3 percentage points, but the per-circuit pattern is mixed.
- The downstream legalizer is still ChipDiffusion's weak legalizer, which itself does not achieve legality. A standard legalizer or placement flow is still missing.
- The largest circuits remain excluded from diffusion sampling due to backbone OOM.
- Full-design HPWL results are mixed and sometimes severe.
- The old seed-42 sweep should be removed or clearly marked as superseded to avoid confusion.

## 7. Clarity

The paper is clearer and more defensible than before. Figure 1 now says "Repaired placement" rather than "Legal placement," and the verifier discussion is more accurate.

Remaining clarity issues:

- The phrase "decisive downstream pipeline experiment" is too strong for a modest median improvement.
- The lambda=2 main-table choice remains confusing now that lambda=8 is validated.
- The supplement has both old seed-42 tuned baselines and new 24-trial baseline sweeps; this should be cleaned up.
- Some statements rely on cross-circuit medians while the per-circuit table shows mixed wins and losses.

## 8. Questions for Authors

- Does `VSR -> cd-sched` achieve full legality on any trial? If not, how many residual violations remain after the pipeline?
- Are the downstream improvements statistically significant at the circuit level?
- Why is lambda=2 still the main result if lambda=8 is selected by LOCO and has better Pareto behavior?
- Can the downstream pipeline be tested with a stronger external legalizer or real placement flow?
- Why does VSR preconditioning improve median full-design HPWL only slightly while worsening macro-only HPWL after cd-sched?
- Should the old seed-42 tuned-baseline table be removed now that 24-trial baseline sweeps are available?
- How sensitive are downstream results to choosing cd-std versus cd-sched as the second-stage legalizer?

## 9. Suggestions for Improvement

- Move lambda=8 to the main table or explicitly present VSR as a lambda-parametrized family, not a lambda=2 method.
- Rename "decisive downstream pipeline experiment" to something more neutral, such as "downstream preconditioning experiment."
- Report downstream residual violation counts and full legality rates explicitly.
- Add circuit-level statistical tests for the downstream pipeline comparison.
- Remove or demote the old seed-42 baseline sweep to avoid conflicting with the newer 24-trial sweep.
- State clearly that the downstream benefit is modest and mixed per circuit.
- If possible, add one external legalization/placement-flow comparison, even if only on the tractable circuits.

## 10. Overall Recommendation

Score: 5/10.

The latest version is meaningfully stronger and fixes several prior reviewer-level objections. However, the method still does not produce legal placements, the downstream benefit is modest, the novelty is limited, and the strongest practical experiment still relies on a weak legalizer that also fails to legalize. I would view this as a solid empirical workshop-style contribution or borderline conference paper, but not yet a strong NeurIPS acceptance.

Confidence: 4/5.

The paper now provides enough evidence to judge the contribution. My main concern is not missing implementation detail; it is that the demonstrated practical impact remains too small relative to the claims.
