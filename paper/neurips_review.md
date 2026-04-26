# NeurIPS-Style Review: Verifier-Guided Selective Repair for Diffusion-Generated Macro Placements

## Priority Fixes

1. **The biggest remaining issue is practical impact.** VSR and the downstream `VSR -> cd-sched` pipeline still produce no fully legal placements: full legality is 0/24, and the downstream pipeline still leaves about 22k residual violations. The paper should make this the central limitation and avoid any phrasing that suggests the method solves legalization.

2. **The downstream pipeline result is weak and should be framed as negative/mixed, not as strong evidence.** The updated paper now reports the honest numbers: `raw -> cd-sched` vs. `raw -> VSR -> cd-sched` gives residual violations 23,824 vs. 22,089, full-design HPWL +8.4% vs. +7.1%, and circuit-level Wilcoxon p-values `p_v=0.562`, `p_hf=1.000`. This is not statistically significant and the per-circuit wins are mixed.

3. **The paper still needs a stronger external legalization/placement-flow comparison if it wants to claim practical placement value.** ChipDiffusion's legalizers are weak and also fail to legalize. A comparison to a more standard downstream legalizer or placement flow would substantially improve credibility.

4. **The novelty should be toned down further.** The method is a well-engineered combination of classical pairwise repulsion, boundary correction, netlist springs, and masking. The contribution is empirical packaging and analysis for diffusion-output repair, not a fundamentally new constraint-solving method.

5. **The lambda story is improved but still slightly awkward.** The paper now reports both lambda=2 and lambda=8 in Table 1, which helps. However, the statistical tests are still at lambda=2 while LOCO selects lambda=8. The paper should clearly state that lambda=2 is only a conservative/reference point and not the recommended operating point.

## 1. Summary

The paper proposes VSR, a verifier-guided repair layer for diffusion-generated macro placements. A verifier computes per-macro severity, pairwise overlap structure, and boundary protrusions; a masked force-directed operator then moves violating macros using repulsion, boundary correction, and netlist attraction, controlled by a scalar lambda. The latest version frames VSR as a lambda-controlled family of repair operators rather than a single legalizer, reports both lambda=2 and lambda=8, adds a downstream pipeline test, and expands baseline comparisons.

## 2. Strengths

- The paper is now substantially more honest about scope: VSR is clearly positioned as a repair/preconditioning layer, not a full legalizer.
- The abstract now reports the most important negative downstream result: `VSR -> cd-sched` is not significant and both pipelines leave roughly 22k residual violations.
- Reporting lambda=2 and lambda=8 side-by-side in Table 1 makes the lambda-controlled family much clearer.
- The downstream pipeline test, cd-std sensitivity check, full 24-trial force-directed baseline, and 24-trial cg/RePaint comparisons meaningfully strengthen the empirical section.
- The statistical framing is much improved: circuit-level n=6 testing is the primary conservative test, with seed-level n=24 treated only as a paired bootstrap.
- The previous incorrect non-differentiability framing has been mostly fixed; the verifier is now described as piecewise differentiable.

## 3. Weaknesses

- The main practical weakness remains severe: neither VSR nor VSR followed by ChipDiffusion's legalizer produces a fully legal placement on any of the 24 trials.
- The downstream result is not significant and only modestly improves medians. It reduces median residual violations from 23,824 to 22,089 and full-design HPWL from +8.4% to +7.1%, but per-circuit wins are mixed.
- The post-pipeline macros-only HPWL is worse with VSR preconditioning: +226.2% versus +215.7%. This undercuts any broad claim that VSR improves final placement quality.
- Full-design HPWL remains problematic before downstream legalization: at lambda=8, VSR-post has median full-design HPWL +8.9%, with adaptec3 at +100.0% and adaptec4 at +30.3%.
- The strongest comparisons are still against ChipDiffusion-specific legalizers and simple force-directed baselines. A mature external physical-design flow is missing.
- The method's novelty is limited because the core ingredients are standard force-directed/legalization ideas.
- The largest circuits are still excluded from diffusion sampling due to the ChipDiffusion backbone's memory behavior, so scalability remains unresolved.
- The main claim that the VSR Pareto frontier "empirically dominates" ChipDiffusion's legalizer should be phrased carefully because full legality is never achieved and downstream gains are not significant.

## 4. Technical Soundness

The repair operator is technically plausible and better specified than in earlier versions. The force terms, clamping, degree normalization, pairwise overlap signal, and lambda tradeoff are reasonable engineering choices.

The main limitation is not correctness of the heuristic; it is that the heuristic does not get close to solving the hard constraint. Reducing violations by about half is useful as a repair signal, but the resulting placements remain far from legal. The downstream experiment shows only weak evidence that this repair improves the final legalizer outcome.

## 5. Novelty & Significance

Novelty is modest. The method combines classical overlap repulsion, boundary correction, netlist attraction, and offender masking. The diffusion-specific contribution is the empirical study of using these signals to repair diffusion outputs.

Significance is limited by practical utility. If the method does not produce legal placements and does not significantly improve a downstream legalizer, then its main value is diagnostic/preconditioning rather than a strong placement result.

## 6. Experimental Evaluation

The evaluation is now much stronger than previous versions. The paper now includes:

- lambda=2 and lambda=8 on the main 24-trial setup.
- full 24-trial force-directed baselines.
- full 24-trial cg/RePaint variants.
- downstream `raw -> cd-sched` vs. `raw -> VSR -> cd-sched`.
- cd-std downstream sensitivity.
- residual violation counts and full legality rates.

Remaining problems:

- The downstream result is not statistically significant.
- The downstream legalizers still fail to legalize every trial.
- The improvement over raw downstream legalization is small in absolute terms.
- No mature external legalization/placement-flow baseline is used.
- Full-design HPWL contains severe outliers.

## 7. Clarity

The paper is much clearer than before. The abstract is now quite transparent about the downstream result being non-significant, which is good.

Remaining clarity issues:

- The paper should avoid words like "dominates" unless carefully scoped to the reported metrics and not interpreted as final placement quality.
- The lambda=2 statistical-test choice should be justified as a reference setting, while lambda=8 should be clearly identified as the recommended HPWL-priority setting.
- The downstream pipeline section should be treated as a limitation/diagnostic result, not as strong validation.

## 8. Questions for Authors

- Can the authors test `VSR -> external legalizer` with a stronger placement/legalization flow, even on a subset of circuits?
- Why does VSR preconditioning not significantly improve downstream residual violations or full-design HPWL?
- What kinds of violations remain after VSR and after `VSR -> cd-sched`? Are they many tiny overlaps or large structural failures?
- Would iterative VSR/legalizer alternation reduce the remaining 22k violations, or does the method plateau?
- Are the downstream results different if the objective is overlap area rather than violation count?
- How should a practitioner choose lambda if the target is final legal placement quality rather than pre-legalizer macro HPWL?

## 9. Suggestions for Improvement

- Reframe the paper's central claim as: "VSR is a lightweight repair/preconditioning operator that improves macro-only Pareto metrics, but current downstream legalization remains unresolved."
- Add an error analysis of the residual 18k-22k violations: count, area, max overlap, and spatial clustering.
- Add at least one stronger external legalization/placement-flow experiment if feasible.
- If external legalization is infeasible, explicitly state that final placement usability is not established.
- Add downstream significance tests and per-circuit win/loss summary directly beside the downstream table.
- Replace broad "dominates" language with metric-specific language such as "improves macro-only violation/HPWL Pareto tradeoff relative to ChipDiffusion legalizer settings."

## 10. Overall Recommendation

Score: 5/10.

The paper is now much more rigorous and transparent. It has evolved into a defensible empirical study of diffusion-output repair. However, the core practical result remains weak: the method does not legalize, downstream legalizers still do not legalize, and the downstream benefit is small and statistically non-significant. This is likely a solid workshop or borderline conference contribution, but still not a strong NeurIPS acceptance.

Confidence: 4/5.

The current version provides enough evidence to judge the work. The main concern is no longer missing experiments, but that the added experiments reveal limited practical impact.
