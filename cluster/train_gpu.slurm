#!/bin/bash -l
# Batch PPO training on one wICE A100 GPU. Everything after the script name goes to run_experiment.py:
#
#   cd /data/leuven/394/vsc39468/drone-swarm
#   sbatch cluster/train_gpu.slurm --stage 3_search --name search --seed 42 --no-gif
#   sbatch --time=04:00:00 cluster/train_gpu.slurm --stage 3_search --seed 43 --set ppo.n_envs=8
#
# Check the queue with `squeue -M wice -u $USER`; output goes to logs/slurm-<jobid>.out (submit from the project folder; logs/ must exist, it is kept in git with logs/.gitkeep).
# For an H100 instead: sbatch --partition=gpu_h100 cluster/train_gpu.slurm ...
# Do NOT use the P100/V100 partitions: the CUDA 13 torch build in .venv-gpu does not support them.
#
#SBATCH --job-name=drone-ppo-gpu
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Python/3.13.5-GCCcore-14.3.0
source .venv-gpu/bin/activate
# GPU nodes may lack a writable $HOME/.cache; without this, matplotlib and fontconfig rebuild caches in every process.
export XDG_CACHE_HOME="$VSC_SCRATCH/.cache" MPLCONFIGDIR="$VSC_SCRATCH/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo "Host: $(hostname)   Job: $SLURM_JOB_ID   CPUs: $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('torch', torch.__version__, '->', torch.cuda.get_device_name(0))"

# Each parallel environment is a separate CPU process (ppo.n_envs); keep n_envs below --cpus-per-task.
python -u run_experiment.py --set ppo.device=cuda "$@"
