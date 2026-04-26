# Round-2 review GPU experiment summary

Trials: 24

## Decisive downstream pipeline

- raw → cd-sched          median: Δv=-28.5%  Δh_m=+215.7%  Δh_f=+8.4%  t=18s
- raw → VSR(λ=8) → cd-sched median: Δv=-35.9%  Δh_m=+226.2%  Δh_f=+7.1%  t=22s

## λ=8 main on 4 seeds

- median Δv=-52.2%  Δh_m=-34.1%  Δh_f=+8.9%  residual_v median=17819

## Classical FD on 4 seeds

- FD-pure        median: Δv=-54.2%  Δh=+143.6%
- FD+spring      median: Δv=-51.8%  Δh=+33.8%
- VSR-post(λ=8)  median: Δv=-52.2%  Δh=-34.1%
