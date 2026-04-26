# LOCO cross-validation for λ

## Selected λ per held-out circuit (in-sample objective: strict-Pareto count)

| held-out | λ̂ (objective a) | test Δv% | test Δh% | strict-P? |
|---|---|---|---|---|
| adaptec1 | 8.00 | -39.4 | -34.4 | ✓ |
| adaptec2 | 8.00 | -48.9 | -32.5 | ✓ |
| adaptec3 | 8.00 | -54.7 | -13.6 | ✓ |
| adaptec4 | 8.00 | -54.9 | -41.6 | ✓ |
| bigblue1 | 8.00 | -23.5 | -72.5 | ✓ |
| bigblue3 | 8.00 | -70.1 | -32.7 | ✓ |

## Strict-Pareto rate on held-out circuits

- Objective (a) strict-Pareto count: **6/6**
- Objective (b) Pareto improvement: **6/6**
- Fixed λ=2 (paper default): **3/6**
- Oracle (per-circuit best λ subject to Δv<0): **6/6**

## Headlines

- LOCO test Δv median: -51.8%
- LOCO test Δh median: -33.6%
