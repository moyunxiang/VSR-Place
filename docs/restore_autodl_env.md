# Restore AutoDL Environment

> When you open a new AutoDL instance, follow this to restore the working environment.
> All critical training artifacts are already in git — you just need to re-download external data.

## What's in git (auto-restored via `git clone`)

| Artifact | Path | Size |
|----------|------|------|
| All code | `src/`, `scripts/`, `tests/` | ~300KB |
| Trained NeuralVSR models | `checkpoints/neural_vsr/*.pt` | 2.5MB |
| 3 ISPD2005 guided placements | `data/ispd_placements.pkl` | 574KB |
| All experiment results JSONs | `results/neural_vsr/` | ~100KB |
| All training logs | `results/neural_vsr/training_logs/` | ~30KB |
| Hand-crafted baseline results | `results/ispd2005/final/all_results.json` | |

## What you need to re-acquire

| Artifact | Command | Time | Size |
|----------|---------|------|------|
| ChipDiffusion Large+v2 checkpoint | `bash scripts/download_checkpoints.sh` (needs gdown + Google Drive) | ~3 min | 73MB |
| ISPD2005 Bookshelf benchmarks | `python scripts/parse_ispd2005.py --download` | ~5 min | 1.7GB parsed |
| Synthetic v1.61 (for unguided experiments) | `cd third_party/chipdiffusion && python data-gen/generate.py versions@_global_=v1 num_train_samples=0 num_val_samples=20` | ~10 min | ~6MB |

## Full restore sequence (new AutoDL instance)

```bash
# 1. SSH config (see docs/autodl_ssh_setup.md)

# 2. Clone repo
cd /root/autodl-tmp
git clone --recursive https://github.com/moyunxiang/VSR-Place.git
cd VSR-Place

# 3. Install deps
pip install torch-geometric hydra-core omegaconf wandb shapely \
    matplotlib plotly tqdm pandas seaborn pyyaml performer-pytorch \
    pyunpack patool gdown
pip install -e .

# 4. Restore external data
bash scripts/download_checkpoints.sh   # 73MB ChipDiffusion model
python3 scripts/parse_ispd2005.py --download   # ISPD2005 benchmarks

# 5. Verify
pytest tests/  # should pass 57/57

# 6. If you're on A100 80GB, generate more real placements:
python3 scripts/gen_ispd_placements.py \
    --checkpoint checkpoints/large-v2/large-v2.ckpt \
    --circuits 0 1 2 3 4 5 6 7 \
    --seeds 0 1 2 3 4 5 6 7 \
    --output data/ispd_placements_full.pkl

# 7. Train NeuralVSR on larger dataset:
python3 scripts/train_neural_vsr_real.py \
    --data data/ispd_placements_full.pkl \
    --output checkpoints/neural_vsr/real_traj_v2.pt \
    --epochs 100 --batch-size 8 --trajectory-steps 30

# 8. Eval:
python3 scripts/eval_neural_vsr.py \
    --checkpoint checkpoints/large-v2/large-v2.ckpt \
    --neural checkpoints/neural_vsr/real_traj_v2.pt \
    --circuits 0 1 2 3 4 5 6 7 \
    --seeds 42 123 300
```

## Current results summary (as of 2026-04-20 shutdown)

**Hand-crafted VSR** (proven, reproducible):
- adaptec1: -57.9% violations
- adaptec3: -61.2%
- bigblue1: -37.3%
- 5 circuits OOM on 24GB

**NeuralVSR** (all failing due to 3-sample training set):
- v2 (synthetic): -23% worse
- v5 (scale-invariant synthetic): -12% worse
- real_v1 (3 real + augment): -81% worse
- real_traj_v1 (3 real × 30 trajectory steps): **-9.6%** worse (best, K=10)

Conclusion: need A100 80GB to generate ≥50 real placements across all 8 circuits. Then retrain.
