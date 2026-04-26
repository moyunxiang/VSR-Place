# NeurIPS-Style Review: Verifier-Guided Selective Repair for Diffusion-Generated Macro Placements

## 1. Summary

The paper proposes VSR, a repair layer for diffusion-generated macro placements. Given a ChipDiffusion draft, a verifier computes per-macro severity, pairwise overlap structure, and boundary protrusion. A masked force-directed repair operator then moves offending macros using repulsive, boundary, and netlist-attractive forces, controlled by a scalar lambda. The updated version explicitly frames VSR as a repair layer rather than a complete legalizer, and evaluates it on 6 ISPD2005 circuits with comparisons against ChipDiffusion legalizers, classifier guidance, RePaint-style repair, and additional force-directed/tuned baselines.

## 2. Strengths

- The paper now correctly states that VSR is not a full legalizer. This is an important correction because none of the tested methods reaches zero violations.
- The added circuit-level Wilcoxon analysis is more appropriate than the previous seed-level-only significance test.
- The paper now includes useful robustness additions: full-design HPWL with fixed cells, tuned classifier-guidance/RePaint sweeps, a classical force-directed baseline, lambda leave-one-circuit-out selection, overlap-area checks, and more precise force-update details.
- The lambda sweep is informative and shows that the method exposes a practical quality tradeoff.
- The implementation is simple and computationally cheap relative to the diffusion backbone and ChipDiffusion's 5000-step legalizer.

## 3. Weaknesses

- The main practical weakness remains severe: VSR still leaves roughly 18,000 median violations after repair. This is not close to a usable macro placement. The paper admits a downstream legalizer is required, but does not evaluate the actual downstream pipeline.
- The contribution is now honestly framed as combining classical ingredients, but this makes the NeurIPS-level novelty weaker. Pairwise repulsion, boundary forces, netlist springs, and masking are standard legalization/placement ideas.
- The paper does not show that VSR improves final legal placement quality after a real downstream legalizer. The relevant practical comparison should be: raw diffusion draft + legalizer versus VSR-repaired draft + legalizer, measuring final HPWL, runtime, legality, and perhaps routing/timing.
- Baseline coverage improved, but remains incomplete. The added classical force-directed baseline is not a mature macro legalizer, and the paper still lacks a strong comparison to a standard placement/legalization flow.
- Some added baseline sweeps are not fully comparable. The tuned classifier-guidance/RePaint sweep appears to be per-circuit at seed 42 only, while the main claims use 4 seeds. The classical force-directed baseline is reported on 15 cached drafts and only 4 circuits, not the same 24 trials and 6 circuits as the main experiment.
- The main table still uses lambda = 2 even though the paper's own LOCO procedure always selects lambda = 8. The explanation that lambda = 2 came from an earlier script is not a scientific justification. If lambda = 8 is validated, the paper should either use it consistently or present VSR as a family of operating points rather than a single method.
- The updated text still contains internal inconsistencies. Section 4.1 says all paired comparisons use Wilcoxon with n = 24, while Section 4.4 says the conservative primary test is circuit-level n = 6. Table 2 says there are 12 non-strict-Pareto trials, but then describes 12 non-Pareto plus 2 weak-violation-cut cases, which totals 14. Section 4.7 says all 12 failures are HPWL-inflation failures, but Appendix F includes weak-violation-cut cases with improved HPWL.
- Some statistical claims remain too strong. The conservative n = 6 test is very small, p-values are marginal, and there is no multiple-comparison correction. Also, Table 3 reports VSR-post vs. RePaint-binary HPWL p = 0.0747, so the claim that VSR outperforms RePaint-binary on both metrics under p < 0.05 is not supported.
- The paper still calls the verifier "non-differentiable" and says max/clamp terms offer no useful gradient at the boundary. This is overstated: overlap penalties are piecewise differentiable and are routinely used in placement optimization.
- Full-design HPWL weakens the story. VSR-post has median full-design HPWL increase of +10.7%, and adaptec3 worsens by +73.2%. This suggests macros-only HPWL may hide important cell-coupling costs.

## 4. Technical Soundness

The repair operator is technically plausible as a heuristic. The added specification of repulsive force, boundary force, attractive force, degree normalization, and per-step clamping improves reproducibility.

However, the theoretical framing remains weak. The method is essentially a masked force-directed macro repair heuristic. That is not incorrect, but the paper should avoid presenting this as a fundamentally new constraint-handling primitive. The strongest technical interpretation is: a carefully tuned lightweight repair step can improve ChipDiffusion drafts before downstream legalization.

The remaining concern is that the method is evaluated mostly against the same violation definition it uses internally. The added overlap-area metric helps, but a true independent evaluator would be a downstream legalizer or physical-design flow.

## 5. Novelty & Significance

Novelty is modest. The updated paper explicitly admits the components are classical, which is accurate. The novel part is mostly the empirical packaging around diffusion-output repair and the lambda-controlled Pareto analysis.

The significance is limited unless the authors show that this repair layer improves final legal placement outcomes. Reducing violations from very bad to still very bad is not enough for a strong placement paper, even if HPWL is better than the released ChipDiffusion legalizer.

## 6. Experimental Evaluation

The updated evaluation is substantially stronger than before, but still not fully convincing.

Major remaining gaps:

- No fully legal placements are produced by any method.
- No downstream legalizer experiment shows whether VSR actually helps the final usable placement.
- Baselines are still not matched on the same full set of circuits/seeds.
- Full-design HPWL shows a nontrivial degradation for VSR-post, especially adaptec3.
- The largest circuits are excluded from diffusion sampling, so scalability remains unresolved at the benchmark sizes where repair would matter most.
- The n = 6 statistical test is conservative but underpowered; conclusions should be phrased cautiously.

The lambda sweep and LOCO result are useful, but they also make the main-table lambda = 2 choice awkward. The paper should either rerun the main 4-seed table at lambda = 8 or explicitly define lambda = 2 as one operating point rather than the main method.

## 7. Clarity

The paper is clearer than the previous version and is more honest about limitations. However, several claims should still be softened.

Confusing or problematic parts:

- Figure 1 still labels the output as "Legal placement," which conflicts with the explicit statement that no method reaches full legality.
- Section 4.1 and Section 4.4 disagree about whether the primary Wilcoxon test uses n = 24 or n = 6.
- The Table 2 / Appendix F failure-count explanation is confusing and mathematically inconsistent as written.
- Phrases like "strictly dominates" and "right interface" remain too strong for the evidence.

## 8. Questions for Authors

- What happens if VSR output is fed into a standard downstream legalizer? Does final legal HPWL improve compared with raw diffusion + same legalizer?
- Why not make lambda = 8 the main setting if LOCO consistently selects it?
- Can the main results be rerun for lambda = 8 on all 24 trials?
- Why are the tuned cg/RePaint baselines only reported for seed 42?
- Why is the classical force-directed baseline not evaluated on the same 6 circuits and 24 trials?
- How many violations remain per circuit after VSR at lambda = 8, not just lambda = 2?
- Does the full-design HPWL degradation on adaptec3 persist after downstream legalization?
- Can the authors remove or justify the remaining "non-differentiable" claim more carefully?

## 9. Suggestions for Improvement

- Add the decisive experiment: raw diffusion + downstream legalizer versus VSR + same downstream legalizer.
- Use lambda = 8 as the validated main setting, or present results as a lambda-controlled Pareto family rather than privileging lambda = 2.
- Evaluate all added baselines on the same circuits and seeds as the main experiment.
- Fix the n = 24 / n = 6 statistics inconsistency.
- Fix the Table 2 and Appendix F failure-count inconsistency.
- Replace "legal placement" with "repaired placement" wherever outputs still contain violations.
- Tone down "strictly dominates" unless both metrics are statistically significant under the conservative test.
- Discuss full-design HPWL degradation more directly, especially adaptec3.

## 10. Overall Recommendation

Score: 5/10.

The updated paper is clearly improved. It is more honest, includes stronger diagnostics, and fixes several earlier statistical and framing issues. However, the core contribution remains an engineering repair heuristic built from classical placement ideas, and the method still leaves many thousands of hard-constraint violations. Without showing improvement after a real downstream legalizer, the practical significance remains limited.

Confidence: 4/5.

The main evidence is now easier to judge. My remaining concerns are not about missing minor details; they are about whether the contribution is novel and practically meaningful enough for NeurIPS.
