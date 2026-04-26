# Round-2 downstream pipeline statistics

## Per-circuit residual + full legality

- adaptec1: raw $v_\text{post}=16604$, pipe $v_\text{post}=16663$, legal raw=0/4, pipe=0/4
- adaptec2: raw $v_\text{post}=16017$, pipe $v_\text{post}=18532$, legal raw=0/4, pipe=0/4
- adaptec3: raw $v_\text{post}=31044$, pipe $v_\text{post}=25647$, legal raw=0/4, pipe=0/4
- adaptec4: raw $v_\text{post}=70714$, pipe $v_\text{post}=69133$, legal raw=0/4, pipe=0/4
- bigblue1: raw $v_\text{post}=9216$, pipe $v_\text{post}=9393$, legal raw=0/4, pipe=0/4
- bigblue3: raw $v_\text{post}=66244$, pipe $v_\text{post}=37671$, legal raw=0/4, pipe=0/4

## Cross-circuit medians

- raw → cd-sched residual_v median: 23824
- vsr8 → cd-sched residual_v median: 22089
- raw → cd-sched full Δh median: +8.4%
- vsr8 → cd-sched full Δh median: +7.1%
- full-legality (v=0): raw 0/24, pipe 0/24

## Circuit-level paired Wilcoxon (n=6)

- residual $v_\text{post}$: $p = 0.562$
- full-design $\Delta h_f$: $p = 1.000$
