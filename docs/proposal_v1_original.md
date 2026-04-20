# Verifier-Guided Selective Re-noising for Constraint-Satisfying Macro Placement

## 1. Title

**Verifier-Guided Selective Re-noising for Constraint-Satisfying Macro Placement**

---

## 2. Abstract

Macro placement is a core problem in chip physical design, where the goal is to place large macros within a chip canvas while optimizing quality metrics such as legality and wirelength. Recent diffusion-based approaches, most notably ChipDiffusion, have shown that diffusion models can generate high-quality macro placements and can be guided at inference time using differentiable surrogate objectives such as legality and half-perimeter wirelength (HPWL). However, these methods still rely on smooth approximations of constraints rather than executable non-differentiable verification signals.

We propose **Verifier-Guided Selective Re-noising (VSR-Place)**, a closed-loop refinement framework that bridges diffusion sampling and non-differentiable placement verification. Instead of relying solely on differentiable legality potentials, our method introduces a verifier-in-the-loop mechanism: after generating an intermediate placement, an external legality verifier detects violations, attributes them to specific objects or object pairs, and returns structured feedback. This feedback is then used to selectively re-noise only the offending macros or local regions, followed by re-denoising to repair invalid layouts while preserving already valid regions.

The key idea is that layout violations are often sparse and localized, so targeted re-noising is more effective than global resampling. Our method is designed to be fully compatible with pretrained diffusion placers such as ChipDiffusion, enabling direct comparison without retraining the backbone. We plan to evaluate the method under the same benchmark protocols used by ChipDiffusion, focusing on IBM/ICCAD-style and ISPD2005 macro placement benchmarks, and compare legality, violation count, final pass rate, HPWL tradeoff, and runtime. We expect verifier-guided selective re-noising to significantly improve final legality and violation reduction efficiency while preserving placement quality.

---

## 3. Motivation

Diffusion models have recently emerged as a promising direction for chip macro placement because they can model complex spatial distributions and generate placements in a zero-shot or low-supervision manner. ChipDiffusion demonstrates that diffusion sampling, when combined with inference-time guidance, can produce competitive placements without requiring a separately trained reward model. In particular, it uses differentiable surrogate objectives for legality and HPWL during sampling.

However, there remains a clear methodological gap between **differentiable guidance** and **real executable verification**:

- Diffusion guidance typically assumes a smooth objective or a gradient-accessible potential.
- Real placement constraints are often naturally expressed through discrete checks or programmatic verification.
- Constraint violations are not always best summarized by a single scalar penalty; instead, they often admit object-level or pairwise attribution.
- Once an invalid local structure appears during denoising, continuing standard denoising may not be sufficient to repair it efficiently.

This suggests a new question:

> Can a diffusion-based placer be improved by integrating a non-differentiable verifier into the generation loop, and can localized repair via selective re-noising outperform purely differentiable guidance?

This proposal aims to answer that question.

---

## 4. Problem Statement

We consider macro placement on a chip canvas.

### Input
- A netlist graph \( G = (V, E) \)
- Macro attributes such as width, height, and possibly pin-related features
- A rectangular chip boundary

### Output
- A 2D placement \( x = \{(x_i, y_i)\}_{i=1}^N \) for all macros

### Goal
Generate placements that:
1. satisfy legality constraints,
2. maintain good placement quality such as low HPWL,
3. can be improved through verifier-guided refinement without retraining the backbone model.

---

## 5. Key Idea

The central idea is to transform a non-differentiable verifier into a usable signal for diffusion refinement.

Instead of treating the verifier as a final post-processing filter, we place it **inside a closed loop**:

1. Generate an intermediate placement from a pretrained diffusion model.
2. Run a verifier to detect violations.
3. Convert verifier output into structured violation feedback.
4. Selectively re-noise only the offending macros or local regions.
5. Re-denoise to repair the layout.
6. Repeat until the layout becomes legal or a repair budget is exhausted.

This yields a repair-oriented diffusion procedure that preserves valid regions and focuses stochastic search on problematic parts of the placement.

---

## 6. Methodology

## 6.1 Backbone

We use **ChipDiffusion** as the generation backbone.

The reason for this choice is practical and methodological:
- it already provides an open-source diffusion pipeline for chip macro placement,
- it has released checkpoints and public evaluation protocols,
- it supports inference-time guidance with legality and HPWL surrogate objectives,
- it allows direct apples-to-apples comparison between the original sampling procedure and our verifier-guided refinement.

Our framework is designed to be inserted at inference time, so no retraining of the backbone is required for the main version of the method.

---

## 6.2 Verifier

We define a **non-differentiable legality verifier** \( V(x) \) over a candidate placement \( x \).

### MVP verifier
The minimum viable verifier checks:
- **boundary violations**: a macro lies partially outside the chip boundary,
- **pairwise overlap violations**: two macros overlap,
- optionally **minimum-spacing violations**: two macros are closer than a required margin.

This choice is intentional:
- it aligns well with the legality notion used in diffusion-based macro placement,
- it is easy to execute exactly,
- it does not require gradients,
- it supports localized attribution.

### Possible extension
A stronger version of the verifier may additionally incorporate:
- coarse routability or congestion proxy,
- channel blockage proxy,
- downstream compatibility with a legalizer or detailed placement tool.

The core proposal, however, is built around non-differentiable legality checking rather than full industrial DRC.

---

## 6.3 Structured Violation Feedback

A key methodological contribution is to avoid collapsing verifier output into a single scalar.

Instead, we extract three levels of feedback:

### (a) Pairwise violation structure
A matrix \( A \in \mathbb{R}^{N \times N} \) where:
- \( A_{ij} > 0 \) indicates that macros \( i \) and \( j \) violate a rule,
- the value can encode severity, such as overlap area or spacing shortfall.

### (b) Object-wise severity
A vector \( r \in \mathbb{R}^N \) where each \( r_i \) summarizes how problematic macro \( i \) is, e.g.
\[
r_i = \sum_j A_{ij} + b_i
\]
where \( b_i \) captures boundary-related severity.

### (c) Global summary
A small set of global statistics such as:
- total violation count,
- total overlap area,
- number of boundary-offending macros,
- whether all constraints pass.

This structured feedback is crucial because layout errors are local and sparse. It lets the diffusion sampler know **where** the layout is wrong rather than merely **how wrong** it is overall.

---

## 6.4 Selective Re-noising

Standard denoising updates the entire sample. Our method instead repairs only the problematic portion.

Let \( M \subseteq \{1, \dots, N\} \) denote the set of macros with violation severity above a threshold.

For each offending macro \( i \in M \), we apply re-noising:
\[
x_i' = \sqrt{1 - \alpha_i}\,\hat{x}_{0,i} + \sqrt{\alpha_i}\,\epsilon_i
\]
where:
- \( \hat{x}_{0,i} \) is the current clean estimate,
- \( \epsilon_i \sim \mathcal{N}(0, I) \),
- \( \alpha_i \) is the re-noising strength, potentially adaptive to violation severity.

For non-offending macros \( i \notin M \), we keep them fixed or nearly fixed:
\[
x_i' = \hat{x}_{0,i}
\]

This **selective re-noising** mechanism aims to:
- preserve already valid structure,
- avoid unnecessary perturbation,
- localize stochastic search,
- improve repair efficiency relative to global resampling.

---

## 6.5 Closed-Loop Refinement Procedure

The inference loop is conceptually:

1. Sample an intermediate placement using the diffusion backbone.
2. Decode the current placement estimate.
3. Run verifier \( V(x) \).
4. If the placement passes, continue normal denoising or terminate.
5. If violations exist:
   - compute structured feedback,
   - identify offending macros,
   - selectively re-noise those macros,
   - resume denoising under the updated state.
6. Repeat until:
   - all constraints pass,
   - loop budget is exhausted,
   - or no further improvement is observed.

This can be viewed as a verifier-driven local repair mechanism embedded inside diffusion sampling.

---

## 6.6 Optional Conditioning Variants

We consider several progressively stronger ways to inject verifier information:

### Variant A: Mask-only repair
Use violation severity only to decide which macros are re-noised.

### Variant B: Feature-level conditioning
Append object-wise verifier signals to macro features during denoising.

### Variant C: Rich structured conditioning
Use pairwise and object-level feedback through a lightweight violation encoder.

For a first paper version, Variant A is the cleanest and most implementation-friendly. Variants B/C can serve as extensions or ablations.

---

## 7. Research Hypotheses

We will test the following hypotheses:

### H1
Structured violation feedback is more informative than a scalar legality penalty.

### H2
Selective re-noising is more effective and compute-efficient than global re-noising.

### H3
A non-differentiable verifier loop can improve final legality and pass rate while preserving HPWL better than naive rejection or purely post-hoc legalization.

### H4
Verifier-guided refinement can be added to a pretrained diffusion backbone without retraining and still produce measurable gains.

---

## 8. Experimental Plan

## 8.1 Benchmarks

We plan to evaluate under the same benchmark protocol used by ChipDiffusion, focusing on:
- IBM / ICCAD-style macro placement benchmarks,
- ISPD2005 macro placement benchmarks.

The goal is not to redefine the benchmark but to ensure direct comparability with the public ChipDiffusion setup.

---

## 8.2 Baselines

We will compare against:

1. **ChipDiffusion (unguided)**
2. **ChipDiffusion (guided)**
3. **ChipDiffusion + original legalizer / post-processing**
4. **Ours: ChipDiffusion + verifier-guided selective re-noising**
5. **Ours + original legalizer**
   - to test whether verifier-loop repair and traditional legalization are complementary.

Optional stronger engineering baselines may include analytical or optimization-based placers, but the core comparison is against the ChipDiffusion family.

---

## 8.3 Metrics

We will report:

### Constraint-related metrics
- final legality / pass rate,
- total violation count,
- overlap area,
- number of boundary violations,
- success within a fixed repair budget.

### Quality-related metrics
- HPWL,
- normalized HPWL ratio.

### Efficiency-related metrics
- runtime,
- number of extra verifier calls,
- number of extra repair loops,
- performance under matched compute budget.

The main story of the paper will be the **legality–quality–compute tradeoff**.

---

## 8.4 Ablations

We will conduct systematic ablations on:

### Re-noising scope
- global re-noising,
- selective re-noising,
- top-k offending macros,
- thresholded offending set.

### Re-noising strength
- fixed strength,
- severity-adaptive strength.

### Verifier signal type
- scalar only,
- object-wise only,
- object-wise + pairwise.

### Repair frequency
- every step,
- every k steps,
- only late-stage refinement.

### Repair budget
- 1 loop,
- 2 loops,
- 4 loops,
- 8 loops.

### Constraint set
- boundary only,
- overlap only,
- boundary + overlap,
- boundary + overlap + spacing.

These ablations are important because they determine whether the gains come from the verifier signal itself, from selective localization, or simply from more computation.

---

## 9. Expected Contributions

We expect the paper to make three main contributions:

### (1) Methodological contribution
We introduce a principled way to bridge **non-differentiable executable verification** and **diffusion-based generation** for macro placement.

### (2) Algorithmic contribution
We propose **violation-aware selective re-noising**, a localized repair mechanism that preserves valid layout structure while resampling only the problematic macros.

### (3) Empirical contribution
We demonstrate that a verifier-in-the-loop refinement framework can improve final legality and pass rate under the same benchmark protocol and pretrained backbone used by ChipDiffusion, while preserving placement quality as measured by HPWL.

---

## 10. Why This Direction Is Promising

This direction is attractive for several reasons:

1. **Strong backbone availability**  
   ChipDiffusion already provides open code, pretrained checkpoints, and reproducible evaluation protocols.

2. **Clear methodological gap**  
   Existing diffusion guidance in chip placement relies on differentiable surrogate objectives rather than executable non-differentiable verification.

3. **Localized structure of violations**  
   Placement errors are naturally sparse and object-specific, making selective repair particularly well matched to the problem.

4. **Good experimental controllability**  
   The proposal admits clean, directly comparable experiments with strong baselines and measurable outcomes.

5. **Potential generalization**  
   The verifier-guided re-noising principle may later extend beyond chip placement to other structured layout domains.

---

## 11. Limitations and Risks

Several risks should be acknowledged:

### (a) Legality improvement may come at the cost of HPWL
Selective re-noising could repair legality while slightly worsening wirelength.

### (b) Too many verifier loops may increase runtime
A matched-compute comparison is necessary.

### (c) Gains may depend heavily on verifier quality
If verifier attribution is noisy or overly coarse, selective repair may be less effective.

### (d) Full industrial DRC is beyond the first-stage scope
The first version of the work should focus on legality and possibly coarse routability proxies, not full downstream signoff.

These limitations do not weaken the proposal; rather, they define a realistic and focused first paper.

---

## 12. Broader Methodological Positioning

This work sits at the intersection of:

- diffusion-based structured generation,
- constrained generative modeling,
- executable verification,
- closed-loop repair for design optimization,
- chip physical design and macro placement.

The broader message is:

> Instead of requiring every generation constraint to be differentiable, one can use executable verifiers as external feedback and couple them with selective stochastic repair inside the diffusion process.

This is the deeper methodological contribution beyond the chip placement application itself.

---

## 13. Related Work

### Diffusion for chip placement
- **ChipDiffusion** is the most directly relevant work. It formulates macro placement using diffusion models and uses inference-time guidance for legality and HPWL.
- It serves as both the backbone and the main baseline in this proposal.

### Placement optimization and legalization
- Classical analytical placers and legalizers remain strong engineering baselines.
- GPU-accelerated placers such as DREAMPlace provide a useful reference for downstream placement compatibility.

### Guidance in diffusion models
- **Universal Guidance** shows that diffusion models can be steered at inference time by arbitrary guidance functions.
- This motivates using external objectives during sampling, but most such approaches still assume a smooth score-like signal.

### Resampling and repair-style diffusion
- **RePaint** demonstrates that resampling selected regions during diffusion is an effective way to satisfy partial constraints in image generation.
- Our work transfers this intuition from image-space inpainting to object-space layout repair.

### Constrained generative modeling
- Prior constrained diffusion methods study projections, optimization alignment, or constrained sampling in more general settings.
- Our proposal differs in emphasizing:
  - executable non-differentiable verification,
  - localized object-level repair,
  - chip-placement-specific legality structure.

---

## 14. Proposed Paper Message

A concise way to summarize the paper is:

> Diffusion-based macro placement currently relies on differentiable surrogate guidance, but real placement constraints are often better expressed through executable verification. We propose verifier-guided selective re-noising, a closed-loop refinement mechanism that converts non-differentiable violation feedback into localized repair during diffusion sampling, improving legality while preserving placement quality.

---

## 15. References

1. **Chip Placement with Diffusion Models**
2. **ChipDiffusion GitHub repository**
3. **Universal Guidance for Diffusion Models**
4. **RePaint: Inpainting using Denoising Diffusion Probabilistic Models**
5. **DREAMPlace**
6. **ICCAD 2004 benchmark documentation**
7. **ISPD 2005 benchmark suite / associated benchmark documentation**
8. **OpenROAD detailed placement documentation**
9. **Representative constrained diffusion / projected diffusion / optimization-aligned diffusion papers**

---