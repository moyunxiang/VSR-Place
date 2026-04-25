#!/usr/bin/env bash
# Bootstrap a fresh AutoDL instance for VSR-Place.
#
# Repo: https://github.com/moyunxiang/VSR-Place
#
# Run on AutoDL AFTER:
#   1. git clone https://github.com/moyunxiang/VSR-Place.git /root/autodl-tmp/VSR-Place
#      (or: git clone https://ghfast.top/https://github.com/moyunxiang/VSR-Place.git
#       if direct GitHub is blocked from the AutoDL region)
#   2. scp local_artifacts/{large-v2.ckpt, config.yaml, ispd2005dp.tar.xz} from local mac
#      → /root/autodl-tmp/VSR-Place/local_artifacts/   (use scripts/push_to_autodl.sh)
#   3. cd /root/autodl-tmp/VSR-Place
#   4. bash scripts/bootstrap_autodl.sh
#
# Idempotent: re-running is safe.

set -euo pipefail

REPO=/root/autodl-tmp/VSR-Place
cd "$REPO"

echo "[1/6] Initializing submodule (ChipDiffusion)..."
if [ ! -f third_party/chipdiffusion/diffusion/models.py ]; then
    git submodule update --init --recursive
fi

echo "[2/6] Installing Python deps..."
pip install -r requirements.txt --quiet
pip install -e . --quiet
# torch_geometric is finicky; pin protobuf >= 6.32.1 to avoid wandb conflict
pip install "protobuf>=6.32.1" --quiet || true

echo "[3/6] Placing ChipDiffusion Large+v2 checkpoint..."
mkdir -p checkpoints/large-v2
if [ -f local_artifacts/large-v2.ckpt ] && [ ! -f checkpoints/large-v2/large-v2.ckpt ]; then
    cp local_artifacts/large-v2.ckpt   checkpoints/large-v2/large-v2.ckpt
    cp local_artifacts/config.yaml     checkpoints/large-v2/config.yaml
fi
ls -lh checkpoints/large-v2/

echo "[4/6] Extracting ISPD2005 archive..."
mkdir -p third_party/chipdiffusion/benchmarks/ispd2005
if [ -f local_artifacts/ispd2005dp.tar.xz ] && [ ! -d third_party/chipdiffusion/benchmarks/ispd2005/adaptec1 ]; then
    tar xJf local_artifacts/ispd2005dp.tar.xz \
        -C third_party/chipdiffusion/benchmarks/ispd2005/
fi
ls third_party/chipdiffusion/benchmarks/ispd2005/ | head

echo "[5/6] Parsing ISPD2005 to PyG pickles..."
if [ ! -f third_party/chipdiffusion/datasets/graph/ispd2005/00000005.pickle ]; then
    python3 scripts/parse_ispd2005.py
fi

echo "[6/6] Running preflight on GPU..."
python3 scripts/preflight_all.py

cat <<EOF

==================================================
Bootstrap done. Available experiment commands:

  # Main NeurIPS unified driver (~30-40 min)
  python3 scripts/run_main_neurips.py

  # Full timestep sweep (~1.5-2 h)
  python3 scripts/run_timestep_sweep.py

  # Memory profile for bigblue2/4 (~30 min)
  python3 scripts/run_mem_profile.py

Outputs land in results/ispd2005/*.json and can be rsync'd back to local.
==================================================
EOF
