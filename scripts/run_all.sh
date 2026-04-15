#!/bin/bash
# Run ALL VSR-Place experiments end-to-end.
# This is the one-command script for AutoDL.
#
# Usage:
#   bash scripts/run_all.sh <checkpoint_path> [task] [seed]
#
# Example:
#   bash scripts/run_all.sh checkpoints/large-v2/step_250000.ckpt v1.61 42

set -e

CKPT="${1:?Usage: bash scripts/run_all.sh <checkpoint_path> [task] [seed]}"
TASK="${2:-v1.61}"
SEED="${3:-42}"

echo "=============================================="
echo " VSR-Place Full Experiment Suite"
echo "=============================================="
echo " Checkpoint: $CKPT"
echo " Task:       $TASK"
echo " Seed:       $SEED"
echo "=============================================="
echo ""

# Verify environment
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# ── 1. Baselines via ChipDiffusion ──────────────────────────────

echo ">>> [1/7] Baseline: ChipDiffusion unguided"
python scripts/run_vsr.py --checkpoint "$CKPT" --task "$TASK" --seed "$SEED" --no-vsr

echo ">>> [2/7] Baseline: ChipDiffusion unguided + legalizer"
python scripts/run_vsr.py --checkpoint "$CKPT" --task "$TASK" --seed "$SEED" --no-vsr --legalize

echo ">>> [3/7] Baseline: ChipDiffusion guided (via eval.py)"
python scripts/run_baseline.py --checkpoint "$CKPT" --task "$TASK" --mode guided --seed "$SEED"

echo ">>> [4/7] Baseline: ChipDiffusion guided + legalizer (via eval.py)"
python scripts/run_baseline.py --checkpoint "$CKPT" --task "$TASK" --mode guided_legalized --seed "$SEED"

# ── 2. VSR-Place ────────────────────────────────────────────────

echo ">>> [5/7] VSR-Place (Variant A)"
python scripts/run_vsr.py --checkpoint "$CKPT" --task "$TASK" --seed "$SEED"

echo ">>> [6/7] VSR-Place + legalizer"
python scripts/run_vsr.py --checkpoint "$CKPT" --task "$TASK" --seed "$SEED" --legalize

# ── 3. Ablations ────────────────────────────────────────────────

echo ">>> [7/7] Ablation suite"
python scripts/run_ablations.py --checkpoint "$CKPT" --task "$TASK" --seed "$SEED"

# ── 4. Aggregate ────────────────────────────────────────────────

echo ""
echo ">>> Aggregating results..."
python scripts/aggregate_results.py

echo ""
echo "=============================================="
echo " ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo " Results: results/"
echo "=============================================="
