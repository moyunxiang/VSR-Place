#!/bin/bash
# Download pretrained ChipDiffusion checkpoints
# Source: https://github.com/vint-1/chipdiffusion
# Checkpoint: https://drive.google.com/drive/folders/16b8RkVwMqcrlV_55JKwgprv-DevZOX8v

set -e

CHECKPOINT_DIR="checkpoints"
LOGS_DIR="third_party/chipdiffusion/logs"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOGS_DIR"

echo "=== Downloading ChipDiffusion Checkpoints (Large+v2) ==="
echo ""

# Check if gdown is available
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown for Google Drive downloads..."
    pip install gdown -q
fi

# Download from Google Drive
GDRIVE_FOLDER="https://drive.google.com/drive/folders/16b8RkVwMqcrlV_55JKwgprv-DevZOX8v"
echo "Downloading from: $GDRIVE_FOLDER"
echo "Target: $CHECKPOINT_DIR/"
echo ""

gdown --folder "$GDRIVE_FOLDER" -O "$CHECKPOINT_DIR/" --remaining-ok

echo ""
echo "=== Download complete ==="

# Also symlink into ChipDiffusion's expected logs/ directory
# ChipDiffusion expects: logs/<task>.<method>.<seed>/step_XXXXX.ckpt
# The from_checkpoint path is relative to logs/
if [ -d "$CHECKPOINT_DIR/large-v2" ]; then
    ln -sfn "$(realpath $CHECKPOINT_DIR/large-v2)" "$LOGS_DIR/large-v2"
    echo "Symlinked: $LOGS_DIR/large-v2 -> $CHECKPOINT_DIR/large-v2"
fi

# List what we got
echo ""
echo "Downloaded files:"
find "$CHECKPOINT_DIR" -type f | head -20

echo ""
echo "Usage:"
echo "  # VSR-Place"
echo "  python scripts/run_vsr.py --checkpoint $CHECKPOINT_DIR/large-v2/<ckpt_file> --task v1.61"
echo ""
echo "  # ChipDiffusion baseline"
echo "  python scripts/run_baseline.py --checkpoint large-v2/<ckpt_file> --task v1.61"
