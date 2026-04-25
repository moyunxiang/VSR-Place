# NeurIPS-Style Review: Verifier-Guided Selective Repair for Diffusion-Generated Macro Placements

## 1. Summary

The paper proposes VSR, a verifier-guided repair layer for diffusion-generated macro placements. A non-learned verifier computes overlap and boundary violations, converts them into macro-level masks and pairwise overlap structure, and a 100-step force-based repair operator moves offending macros while trading legality against HPWL via a scalar lambda. The paper compares this against ChipDiffusion samples, classifier guidance, RePaint-style repair, and ChipDiffusion legalizers on 6 ISPD2005 circuits.

## 2. Strengths

- The problem is important: diffusion placement samples that violate hard geometric constraints are not directly usable.
- The method is simple, fast, and does not require retraining the diffusion backbone.
- The paper includes useful engineering evaluations: lambda sweep, step-budget sweep, component ablation, failure analysis, and memory profiling.
- The Pareto framing is appropriate in principle because placement quality is multi-objective.
- The paper is transparent that the method is domain-specific and does not claim a new diffusion training method.

## 3. Weaknesses

- The central result is not actual legalization. A median 55.3% violation reduction still leaves many hard-constraint violations. For macro placement, "fewer violations" is not equivalent to a legal placement.
- Baselines are weak and insufficiently tuned. VSR gets a lambda sweep, but classifier guidance and RePaint appear to use narrow or fixed settings. Standard placement/legalization baselines such as DREAMPlace, RePlAce, NTUPlace-style legalization, greedy macro legalization, or analytical macro legalizers are not actually benchmarked.
- The evaluation uses only 6 circuits, with 4 seeds each. Treating 24 seed/circuit pairs as independent for Wilcoxon tests is statistically questionable because circuit-level variation dominates.
- The method optimizes almost exactly the same verifier metric used for evaluation, while baselines may optimize different surrogates. This makes the violation-count comparison partially circular.
- The paper uses violation count rather than overlap area, max overlap, legality success rate, routed congestion, timing, or true full-design HPWL. Counting a tiny overlap and a catastrophic overlap equally is not adequate.
- The "structured verifier feedback" novelty is overstated. Pairwise overlap graphs, boundary protrusions, and force-directed repulsion are standard ingredients in placement legalization.
- The ablation undermines the selector contribution: random selector and uniform all-macro movement match the full method because the raw drafts are saturated with violations.
- There are internal inconsistencies: Table 2 implies 12/24 VSR-post trials are non-Pareto, while Appendix D says 14 trials are flagged. The abstract claims p < 0.001 on both metrics against classifier guidance, but Table 3 reports HPWL p = 0.0814 for VSR-post vs. cg-strong.
- The memory explanation is suspicious. Diffusion inference should not accumulate activations over 100 denoising steps unless the implementation stores them or lacks proper inference/no-grad handling.

## 4. Technical Soundness

The repair operator is plausible as a heuristic, but the technical framing is weak. The paper repeatedly calls the verifier "non-differentiable," yet overlap area with max/clamp terms is piecewise differentiable and has useful gradients when overlaps are positive. Classical legalizers exploit exactly this kind of structure.

The force update is underspecified. It is unclear how coordinates, macro sizes, canvas dimensions, and HPWL terms are normalized, how pairwise forces are symmetrized, how hyperedges are converted to pairwise edges, and whether the boundary force is really "snapped" or scaled by eta. Without these details, reproducibility and stability are hard to assess.

The method appears to be a force-directed macro legalizer with a mask. That can be useful, but the paper does not establish why this is technically superior to standard legalization methods, only to the specific released ChipDiffusion legalizer settings.

## 5. Novelty & Significance

Novelty is limited. The idea of using overlap and boundary diagnostics to drive force-based macro movement is close to classical legalization and force-directed placement. The diffusion-specific contribution is mostly the placement of this repair step after or inside sampling.

The significance would be stronger if VSR produced fully legal placements with competitive HPWL against mature placers/legalizers. As written, the method reduces violations but does not solve the hard-constraint problem.

## 6. Experimental Evaluation

The experimental evaluation is not convincing enough for the paper's claims.

Major missing evaluations:

- Absolute violation counts before and after repair.
- Fraction of samples that become fully legal.
- Total overlap area, max overlap, and boundary protrusion magnitude.
- Full-design HPWL including standard-cell connectivity, not only macro-only edge graphs.
- Strong classical legalization baselines.
- Fair hyperparameter sweeps for all baselines.
- Circuit-level statistical tests rather than seed-level pseudo-replication.
- Results on more benchmarks or explanation of why ISPD2005 macros-only is representative.

The lambda sweep is useful but creates another problem: lambda = 8 appears much better for strict Pareto performance than lambda = 2, yet the main results use lambda = 2. The paper needs a principled selection rule or validation protocol.

## 7. Clarity

The paper is mostly readable, but several claims are too aggressive relative to the evidence. Phrases such as "legal placement," "strictly dominates," and "right interface" are not justified when the method does not report full legality and has statistical issues.

Some details are missing or ambiguous:

- Exact construction of the macro-only netlist.
- Absolute HPWL scale and normalization.
- Whether baselines were tuned comparably.
- Whether memory profiling used proper inference mode.
- Exact residual violations after VSR.

## 8. Questions for Authors

- How many VSR outputs are fully legal, i.e. zero overlap and zero boundary violation?
- What are the absolute violation counts and total overlap areas before and after repair?
- Why is lambda = 2 used as the main setting when lambda = 8 appears to give better Pareto results?
- Were classifier guidance and RePaint tuned over comparable hyperparameter grids?
- How does VSR compare against standard macro legalization methods, not just ChipDiffusion's released legalizer?
- Why are seeds treated as independent samples in significance tests?
- Is the OOM issue still present under `torch.no_grad()` / inference mode with no stored intermediate activations?
- Why does Appendix D report 14 flagged failures while Table 2 implies 12?
- Why does the abstract claim p < 0.001 on both metrics against classifier guidance when Table 3 reports HPWL p = 0.0814?

## 9. Suggestions for Improvement

- Reframe the contribution as a heuristic repair/legalization layer, not as a general verifier-guided diffusion advance.
- Report full legality rates and overlap-area metrics, not only violation-count reduction.
- Add strong classical legalizers and simple force-directed baselines with comparable tuning.
- Use circuit-level statistics or a mixed-effects model; do not overstate p-values from 24 seed/circuit pairs.
- Tune all baselines fairly and report sensitivity curves.
- Include a principled validation protocol for choosing lambda.
- Fix internal inconsistencies in Pareto counts and statistical claims.
- Clarify the memory experiment and verify the backbone OOM is not an implementation artifact.

## 10. Overall Recommendation

Score: 4/10.

The paper addresses a real problem and presents a practical heuristic with some promising empirical behavior, but the technical novelty is modest, the claims are overstated, and the evaluation does not establish that the method solves macro-placement legality. The lack of full legality metrics and strong legalization baselines is a major weakness.

Confidence: 4/5.

The paper is clear enough to judge, and the main limitations are visible from the reported tables and claims. My confidence would increase further with access to the code and raw JSONs, especially for baseline tuning and memory profiling.
