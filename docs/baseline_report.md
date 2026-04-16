# ChipDiffusion Baseline Reproduction Report — ISPD2005 Macro Placement

**Date**: April 16, 2026 | **Team**: VSR-Place

---

## Experiment Setup

| Item | Value |
|------|-------|
| Model | ChipDiffusion Large+v2 (6.29M params) |
| Checkpoint | `large-v2.ckpt` (73MB, author-provided) |
| Benchmark | ISPD2005, 8 circuits, Bookshelf format |
| Eval mode | `eval_macro_only`, guided sampling + Adam legalization |
| Hardware | Google Colab H100 80GB |
| Total runtime | ~2.5 hours |

## Benchmark Availability

| Benchmark | Format | Circuits | Status |
|-----------|--------|----------|--------|
| IBM ICCAD04 | DEF/LEF | 18 | **Unavailable** — official source (vlsicad.eecs.umich.edu) offline, TILOS protobuf converter incompatible |
| ISPD2005 | Bookshelf | 8 | **Available** — UTexas CERC mirror, parsed with `parse_bookshelf()` |

## Results

| Circuit | HPWL | HPWL (orig) | Ratio | Legality | Viol.% | Macros | Time |
|---------|------|-------------|-------|----------|--------|--------|------|
| adaptec1 | 186.8 | 266.4 | 0.701 | 0.992 | 0.8% | 543 | 4.3m |
| adaptec2 | 512.8 | 526.5 | 0.974 | 0.917 | **8.3%** | 566 | 13.3m |
| adaptec3 | 506.8 | 669.5 | 0.757 | 0.994 | 0.6% | 723 | 12.3m |
| adaptec4 | 533.3 | 766.6 | 0.696 | 0.997 | 0.3% | 1329 | 4.8m |
| bigblue1 | 56.8 | 61.3 | 0.926 | 0.996 | 0.4% | 560 | 7.6m |
| bigblue2 | 621.8 | 899.7 | 0.691 | 0.999 | 0.1% | 23084 | 80.0m |
| bigblue3 | 288.7 | 413.1 | 0.699 | 0.998 | 0.2% | 1298 | 2.7m |
| bigblue4 | 897.7 | 1664.9 | 0.539 | 0.992 | 0.8% | 8170 | 11.8m |
| **Average** | **450.6** | **658.5** | **0.748** | **0.986** | **1.4%** | **4534** | **17.1m** |

## Observations

- **HPWL**: All 8 circuits achieve ratio < 1, average 0.748 (25.2% wirelength reduction).
- **Legality**: 7 of 8 circuits have violation < 1%. Average legality 0.986.
- **Outlier**: adaptec2 has 8.3% violation rate (legality 0.917).
- **Runtime**: bigblue2 took 80 min (23K macros); all others ≤ 13 min.
