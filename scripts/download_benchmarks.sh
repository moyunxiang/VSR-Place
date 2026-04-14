#!/bin/bash
# Download benchmark datasets for evaluation
# Supports: ICCAD04, ISPD2005

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "=== Downloading Benchmark Datasets ==="
echo ""
echo "ChipDiffusion uses preprocessed benchmark data stored as pickle files."
echo "The data is typically generated using the parsing/ scripts in ChipDiffusion."
echo ""

# Check if ChipDiffusion has datasets included
CD_DATASETS="third_party/chipdiffusion/datasets"
if [ -d "$CD_DATASETS" ]; then
    echo "Found ChipDiffusion datasets directory:"
    ls "$CD_DATASETS"/
    echo ""
    echo "These can be used directly by setting the task config to match"
    echo "ChipDiffusion's dataset naming convention."
else
    echo "ChipDiffusion datasets directory not found."
    echo "Run: git submodule update --init --recursive"
fi

echo ""
echo "For custom benchmarks, use ChipDiffusion's parsing pipeline:"
echo "  cd third_party/chipdiffusion"
echo "  python parsing/parse_bookshelf.py --input <benchmark_dir> --output datasets/<name>"
echo ""
echo "Standard benchmarks:"
echo "  - ICCAD04: https://vlsicad.ucsd.edu/GSRC/bookshelf/Slots/Mixed-Size/"
echo "  - ISPD2005: http://www.ispd.cc/contests/05/ispd2005_contest.html"
