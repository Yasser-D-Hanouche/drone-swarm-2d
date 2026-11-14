#!/usr/bin/env bash
# Build the GPU virtual environment (.venv-gpu) next to the CPU one (.venv).
#
# Run once, from the project folder, on a login node or any wICE node:
#     bash cluster/setup_gpu_venv.sh
# A GPU is NOT needed to install; verify afterwards on a GPU node (see README, section 10).
#
# Why a separate venv: the CPU venv holds torch 2.15.0+cpu, a wheel built without CUDA,
# which can never see a GPU. The default PyPI torch 2.15.0 wheel bundles its own CUDA 13.2
# runtime (nvidia-* packages), so no CUDA module needs to be loaded; only the node's NVIDIA
# driver is used (615.x on wICE, which supports CUDA 13).
# CUDA 13 does not support P100/V100 GPUs: use the A100 or H100 partitions.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_DIR/.venv-gpu"

# Same interpreter as the CPU venv, so both environments behave identically.
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module load Python/3.13.5-GCCcore-14.3.0

# $VSC_HOME is only 3 GB; the CUDA wheels are several GB. Keep pip's cache on scratch.
export PIP_CACHE_DIR="${VSC_SCRATCH:?VSC_SCRATCH not set}/pip-cache"

if [[ -e "$VENV" ]]; then
    echo "$VENV already exists. Remove it first if you want a clean rebuild." >&2
    exit 1
fi

python -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip

# Pinned to the versions verified in the CPU venv; only torch differs (CUDA build).
"$VENV/bin/python" -m pip install \
    "torch==2.15.0" \
    "stable-baselines3==2.9.2" \
    "gymnasium==1.4.0" \
    "numpy==2.5.4" \
    "matplotlib==3.11.2" \
    "pillow==13.0.0" \
    "tensorboard"

"$VENV/bin/python" -m pip check
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, '| built for CUDA', torch.version.cuda)"
echo "Done. Pip cache in $PIP_CACHE_DIR can be removed with: rm -rf $PIP_CACHE_DIR"
