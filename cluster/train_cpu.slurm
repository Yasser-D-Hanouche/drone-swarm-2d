#!/bin/bash -l
# Batch PPO training on CPU cores of a wICE batch node (not limited to the 2 cores of an interactive session).
# Everything after the script name goes to run_experiment.py:
#
#   cd /data/leuven/394/vsc39468/drone-swarm
#   sbatch cluster/train_cpu.slurm --stage 3_search --name search --seed 42 --no-gif
#
# Several seeds in parallel, one job each (compare_runs.py groups them by --name):
#   for s in 42 43 44; do sbatch cluster/train_cpu.slurm --stage 3_search --name search --seed $s --no-gif; done
#
# More cores: sbatch --cpus-per-task=17 cluster/train_cpu.slurm ...  (n_envs follows automatically)
# Check the queue with `squeue -M wice -u $USER`; output goes to slurm-<jobid>.out in the submit folder.
#
#SBATCH --job-name=drone-ppo-cpu
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Python/3.13.5-GCCcore-14.3.0
source .venv/bin/activate
export XDG_CACHE_HOME="$VSC_SCRATCH/.cache" MPLCONFIGDIR="$VSC_SCRATCH/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# One core for the PPO process, one per environment process.
# n_steps is scaled so n_envs * n_steps stays 1024 environment steps per update (as with the defaults 4 x 256),
# so more cores give speed without changing the learning dynamics. A later --set in "$@" overrides either value.
N_ENVS=$(( SLURM_CPUS_PER_TASK - 1 ))
N_STEPS=$(( 1024 / N_ENVS ))
echo "Host: $(hostname)   Job: $SLURM_JOB_ID   CPUs: $SLURM_CPUS_PER_TASK   n_envs: $N_ENVS   n_steps: $N_STEPS"

python -u run_experiment.py --set ppo.device=cpu ppo.n_envs=$N_ENVS ppo.n_steps=$N_STEPS "$@"
