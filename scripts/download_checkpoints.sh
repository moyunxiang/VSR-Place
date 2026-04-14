#!/bin/bash
# Download pretrained ChipDiffusion checkpoints
# Source: https://github.com/vint-1/chipdiffusion

set -e

CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "=== Downloading ChipDiffusion Checkpoints ==="
echo ""
echo "ChipDiffusion provides pretrained models via the GitHub repository."
echo "Check https://github.com/vint-1/chipdiffusion for latest checkpoint links."
echo ""

# The exact download method depends on how the authors host checkpoints.
# Common patterns: Google Drive, Hugging Face, direct GitHub release.
# Update the URLs below once confirmed.

# Placeholder - update with actual URLs from the repo
echo "TODO: Update with actual checkpoint download URLs from ChipDiffusion repo."
echo ""
echo "Manual steps:"
echo "  1. Visit https://github.com/vint-1/chipdiffusion"
echo "  2. Follow their checkpoint download instructions"
echo "  3. Place checkpoint files in: $CHECKPOINT_DIR/"
echo ""
echo "Expected files:"
echo "  $CHECKPOINT_DIR/chipdiffusion_large.pt"
echo "  $CHECKPOINT_DIR/chipdiffusion_v2.pt"

# Check if checkpoints exist
if ls "$CHECKPOINT_DIR"/*.pt 1>/dev/null 2>&1; then
    echo ""
    echo "Found checkpoints:"
    ls -lh "$CHECKPOINT_DIR"/*.pt
else
    echo ""
    echo "No checkpoints found yet. Please download manually."
fi
