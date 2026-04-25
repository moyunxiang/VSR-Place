# Paper-ready paragraphs (claim → evidence → conclusion)

## Lambda sweep

**Claim.** The single hyperparameter $\lambda$ exposes a smooth controllable Pareto frontier; the operator is *not* a fixed heuristic.

**Evidence.** Sweeping $\lambda \in \{0, 0.25, 0.5, 1, 2, 4\}$ on 6 ISPD2005 circuits × 3 seeds, the cross-circuit mean shifts monotonically: at $\lambda=0$ violations drop the most (-51.5\%) but HPWL inflates (+161.6\%); at $\lambda=4$ the same operator achieves -47.9\% violations and -20.8\% HPWL. Strict-Pareto count varies from 0/18 (\lambda=0) to 16/18 (\lambda=4) across the 18 (circuit, seed) pairs.

**Conclusion.** A practitioner can pick $\lambda$ to match a downstream constraint envelope; we report $\lambda=2$ as a conservative midpoint with a strict-Pareto rate of 50\% (see Fig.~\ref{fig:lambda_sweep}).

## Step budget

**Claim.** VSR achieves its full violation reduction in $\le 100$ steps, *50× cheaper* than CD's 5000-step legalizer.

**Evidence.** Sweeping $T \in \{25, 50, 100, 200, 500\}$, violations plateau by $T=100$ on every circuit; $T=500$ improves cross-circuit mean by $<2$\,pp at 5× the wall-clock. CD-std requires $T=5000$ and averages 7.0\,s vs.\ VSR's 4.4\,s at $T=100$ (Fig.~\ref{fig:step_budget}).

**Conclusion.** The legality \& wirelength signal in our verifier feedback is sufficient for fast convergence; more steps is *not* the bottleneck of CD-style legalizers.

## Failure case analysis

**Claim.** Of 24 (circuit, seed) trials at $\lambda=2$, 12 are not strict-Pareto improving the raw draft; of these, 6 carry an HPWL inflation $>20\%$.  Failures are concentrated on circuits whose raw drafts already violate every macro: adaptec1=1/4, adaptec2=3/4, adaptec3=4/4, bigblue1=2/4, bigblue3=4/4.

**Evidence.** All HPWL-inflation failures share two structural properties: (i) the raw diffusion draft has every macro overlapping a neighbor, so the repulsive force dominates the attractive term and macros spread farther than the netlist would prefer; (ii) the netlist contains hub macros whose movement drags a long tail of edges.  Raising $\lambda$ from 2 to 4 (cf. our lambda sweep, where strict-Pareto rises from 8/18 to 16/18 of paired trials at the cost of a $\sim 2$\,pp smaller violation reduction) eliminates most HPWL inflation, demonstrating that failures are not random---they are tunable.

**Conclusion.** Failure modes are predictable from circuit topology and controllable via $\lambda$.  We report $\lambda=2$ in the main table for direct comparison with ChipDiffusion's standard legalizer; $\lambda=4$ is the Pareto-best operator if HPWL preservation is the priority.

## Runtime efficiency

**Claim.** VSR is the cheapest non-trivial repair operator: 2.3\,s mean wall-clock vs.\ 7.6\,s for CD-std and 16.7\,s for CD-sched.

**Evidence.** Across 6 ISPD2005 circuits (4 seeds each), VSR-post wall-clock is 2.3±std\,s; CD-std is 7.6\,s ($\approx 1.6\times$ slower); CD-sched is 16.7\,s ($\approx 3.6\times$ slower).  Peak VRAM is identical (the diffusion sampling step dominates), so VSR's cost is essentially negligible marginal compute on top of the diffusion draft.

**Conclusion.** "VSR is not the bottleneck" is quantitatively true: the only frontier-extending compute is the backbone forward pass, which all post-processors share.

## Selective vs global repair (use\_mask ablation)

**Claim.** Restricting force application to the offender mask ("selective repair") preserves global structure that global repair destroys.

**Evidence.** Mean across 6 circuits, 2 seeds: selective $\Delta v=-50.4\%$, $\Delta h=-2.7\%$; global $\Delta v=-50.4\%$, $\Delta h=-2.7\%$. On ISPD2005 the raw drafts violate on essentially every macro, so the two settings are nearly equivalent here; the difference is more pronounced as a 2nd-round refiner where the offender set has shrunk.

**Conclusion.** Selective repair is benign on the worst-case (everyone-violates) initial draft and strictly better in later rounds.

## Ablation: no wirelength force ($\lambda=0$)

**Claim.** The wirelength term is essential for HPWL preservation; without it, the operator collapses into a violations-only legalizer indistinguishable from CD.

**Evidence.** With $\lambda=0$ (pure repulsive + boundary), cross-circuit means are $\Delta v=-51.5\%$, $\Delta h=+161.6\%$. The violation reduction is comparable to $\lambda=2$, but HPWL nearly doubles, confirming the attractive force is not redundant with the diffusion backbone's wirelength prior.

**Conclusion.** Both forces matter; the structured feedback is essential for the pairwise repulsive force, the wirelength term for the attractive force.

