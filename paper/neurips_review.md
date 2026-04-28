# NeurIPS-Style Review: Verifier-Guided Selective Repair for Diffusion-Generated Macro Placements

## Priority Fixes

1. **Fix Table 1 immediately.** The text and caption claim Table 1 reports VSR-post at both `lambda=2` and `lambda=8`, but the actual table currently only shows one `VSR-post` column pair. This is a serious consistency bug because the paper's lambda-family framing depends on that table.

2. **Clarify the DREAMPlace experiment setup.** The main text says DREAMPlace has "24 trials, 7 pre-conditioners"; the supplement text says "12 trials (6 circuits x 2 seeds)"; the table caption says four pre-conditioners and 24 trials. These statements conflict. Make the exact number of trials, seeds, and pre-conditioners consistent everywhere.

3. **Reframe the DREAMPlace result as a strong baseline result, not a VSR win.** DREAMPlace plus any force-directed preconditioner roughly halves residual violations, but VSR does not outperform FD-pure or FD+spring. In fact, median residuals are raw 20,075, FD-pure 9,545, FD+spring 9,773, VSR 9,876. The conclusion should be that generic force preconditioning helps DREAMPlace, not that VSR is uniquely effective downstream.

4. **Do not claim practical legalization success.** Even with DREAMPlace, full legality is still 0/24. This remains the central practical limitation and should be emphasized.

5. **Be precise about multiple-comparison correction.** The paper now reports BH and Bonferroni, which is good. But clarify whether BH is applied separately per metric column or across all 16 p-values in Table 3. Current wording says `m=8 comparisons`, while the table contains two metrics for eight comparisons.

## 1. Summary

The paper proposes VSR, a verifier-guided repair layer for diffusion-generated macro placements. A verifier computes per-macro severity, pairwise overlap structure, and boundary protrusions; a masked force-directed operator then moves violating macros using repulsion, boundary correction, and netlist attraction, controlled by a scalar lambda. The latest version adds multiple-comparison-corrected statistics and an external DREAMPlace downstream pipeline.

## 2. Strengths

- The paper is now much more transparent about limitations: no method, including DREAMPlace pipelines, achieves full legality.
- Adding DREAMPlace as an external legalizer is a substantial improvement over relying only on ChipDiffusion's weak legalizers.
- The DREAMPlace result is informative: force-directed preconditioning clearly helps reduce residual violations relative to raw diffusion outputs.
- Reporting BH-FDR and Bonferroni-adjusted p-values improves statistical rigor.
- The paper now acknowledges that VSR does not outperform classical FD preconditioners in the DREAMPlace downstream metric.
- The violation breakdown and iterative VSR/legalizer alternation analysis are useful and help explain why residual violations persist.

## 3. Weaknesses

- The paper still does not demonstrate practical legalization. Full legality remains 0/24 across all tested pipelines.
- The most important downstream result does not uniquely support VSR. DREAMPlace after FD-pure or FD+spring performs as well as or slightly better than DREAMPlace after VSR.
- The method's novelty is further weakened by the DREAMPlace experiment: a classical force-directed preconditioner captures the downstream benefit.
- Table 1 is currently inconsistent with the text/caption because it does not actually show both lambda=2 and lambda=8.
- The DREAMPlace experiment description has conflicting trial/preconditioner counts.
- The paper still relies heavily on macros-only metrics, while full-design HPWL can degrade substantially before downstream legalization.
- The largest ISPD2005 circuits remain excluded from diffusion sampling because of the ChipDiffusion backbone.
- The phrase "empirically dominates ChipDiffusion's released legalizer" should be scoped carefully because this does not imply final placement quality or full legality.

## 4. Technical Soundness

The VSR operator is technically plausible as a lightweight force-directed repair heuristic. The implementation details are now sufficiently clear.

The main technical issue is not correctness; it is attribution. The DREAMPlace results suggest the downstream improvement comes from broad force-directed preconditioning rather than the specific verifier-guided selective repair design. If VSR's distinct value is upstream macros-only HPWL preservation, the paper should make that the central claim and avoid implying downstream superiority.

## 5. Novelty & Significance

Novelty remains modest. The components are classical: overlap repulsion, boundary correction, netlist attraction, and macro masking. The empirical study is useful, but the external legalizer experiment shows that simple classical FD baselines are competitive with VSR in the most practical downstream setting.

Significance is limited by the lack of full legality and by the absence of VSR-specific downstream advantage.

## 6. Experimental Evaluation

The evaluation is now stronger than before:

- DREAMPlace external legalizer pipeline is added.
- cd-sched and cd-std downstream pipelines are reported.
- iterative VSR/legalizer alternation is analyzed.
- residual violation type/area breakdown is reported.
- multiple-comparison-corrected statistics are reported.

Remaining issues:

- DREAMPlace pipeline still leaves about 10k residual violations even after preconditioning.
- VSR does not significantly beat FD-pure or FD+spring under DREAMPlace.
- Table 1 and DREAMPlace setup descriptions need consistency fixes.
- Full legality rates should remain front-and-center in every downstream table.
- If possible, report DREAMPlace full-design HPWL or explain precisely why it is not comparable.

## 7. Clarity

The paper is much clearer and more honest than earlier versions, but there are still important clarity problems.

- Table 1/caption mismatch must be fixed.
- DREAMPlace trial count and preconditioner count must be made consistent.
- The statistical correction setup needs exact wording.
- The main conclusion should distinguish "VSR improves upstream macro-only Pareto metrics" from "generic force preconditioning helps DREAMPlace residual violations."

## 8. Questions for Authors

- Why does Table 1 not show lambda=8 despite the text and caption saying it does?
- How many DREAMPlace treatments were actually run: four preconditioners or seven? 12 trials or 24 trials?
- Does DREAMPlace report any placement legality metric internally, and how does it compare to the paper's verifier?
- Why does VSR not outperform FD-pure/FD+spring after DREAMPlace?
- If the remaining overlaps are structural, what specific structural constraints prevent full legalization?
- Can the paper report full-design HPWL after DREAMPlace, or is only residual violation count available?

## 9. Suggestions for Improvement

- Fix the Table 1 input so lambda=2 and lambda=8 are both shown, or change the caption/text.
- Rewrite the DREAMPlace section to say: "force-directed preconditioning helps; VSR is not better than classical FD in this downstream setting."
- Make full legality `0/24` unavoidable in the conclusion and limitations.
- Clarify BH-FDR correction scope: per metric across 8 comparisons, or across all 16 tests.
- Remove or soften "dominates" language unless it is explicitly limited to macros-only metrics against ChipDiffusion-specific legalizers.
- Add a concise table summarizing the core story: raw, VSR, FD, DREAMPlace+raw, DREAMPlace+FD, DREAMPlace+VSR, with residual count and full legality.

## 10. Overall Recommendation

Score: 5/10.

The paper is now a rigorous and honest empirical study, but the added DREAMPlace experiment cuts both ways: it strengthens the evaluation while showing that VSR is not uniquely better than simple classical force-directed preconditioning in the downstream setting. Since no tested pipeline reaches full legality and the novelty remains modest, I would still not recommend strong acceptance.

Confidence: 4/5.

The evidence is now rich enough to judge. The remaining concerns are mainly practical impact, novelty, and several fixable presentation inconsistencies.
