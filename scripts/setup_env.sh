#!/bin/bash
# VSR-Place environment setup script
# Run this on a machine with GPU to set up the full environment

set -e

echo "=== VSR-Place Environment Setup ==="

# 1. Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# 2. Create conda environment
echo "Creating conda environment 'vsr_place'..."
conda create -n vsr_place python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate vsr_place

# 3. Install PyTorch with CUDA
echo "Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install torch-geometric (required by ChipDiffusion)
echo "Installing torch-geometric..."
pip install torch-geometric

# 5. Install ChipDiffusion dependencies
echo "Installing ChipDiffusion dependencies..."
pip install hydra-core omegaconf wandb shapely matplotlib plotly tqdm pandas

# 6. Install VSR-Place
echo "Installing VSR-Place..."
pip install -e ".[dev]"

# 7. Initialize git submodules
echo "Initializing git submodules..."
git submodule update --init --recursive

# 8. Verify installation
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
from vsr_place.verifier.verifier import Verifier
from vsr_place.loop.vsr_loop import VSRLoop
print('VSR-Place imports OK')
"

# 9. Run tests
echo "Running tests..."
pytest tests/ -v

echo ""
echo "=== Setup complete! ==="
echo "Activate with: conda activate vsr_place"
echo "Next steps:"
echo "  1. Download ChipDiffusion checkpoints: bash scripts/download_checkpoints.sh"
echo "  2. Download benchmark data: bash scripts/download_benchmarks.sh"
echo "  3. Run experiments: python scripts/run_vsr.py --config configs/defaults.yaml"
