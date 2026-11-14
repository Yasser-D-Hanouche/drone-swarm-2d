# Project Log

Chronological record of implementation steps, problems, solutions, and results. For architecture and usage, see [README.md](README.md).

Commit history: this workspace has no `.git` folder, so per-commit hashes are not recorded here. Branch names are as reported by the user.

## Index

- [2026-10-31: Handoff and Stage 1 experiments](#2026-10-31-handoff-and-stage-1-experiments)
- [2026-11-05: Stage 3 review, fixes, and documentation restructure](#2026-11-05-stage-3-review-fixes-and-documentation-restructure)
- [2026-11-14: GPU environment](#2026-11-14-gpu-environment)
- [2026-11-14 (evening): Stage 3 diagnosis and fix](#2026-11-14-evening-stage-3-diagnosis-and-fix)

---

## 2026-10-31: Handoff and Stage 1 experiments

Working branch: `experimental/wall-proximity`. Target integration branch: `advanced`.

### 2026-10-31.1 Implemented before this log

- Gymnasium environment `DroneSwarm2D`: kinematics, vision rays, searched map, target detection, shaped rewards, collision and success detection.
- SB3 PPO training with a monitor callback, reward-term logging, and periodic evaluation.
- Diagnostics: Random and Greedy baselines, deterministic PPO evaluation, metrics, plots, GIFs, automatic diagnosis, and `report.md`.
- `run_experiment.py` entry point with timestamped run folders; `compare_runs.py` for grouping seeds.
- Reward terms: progress, cohesion, formation, separation, obstacle and wall proximity, step, collision, and success.
- Action residual option (`env.action_residual_scale`), vectorised raycast, and relative-position scaling.
- Wall-proximity calculation, which was later found missing and restored (see problem 3).

### 2026-10-31.2 Cluster environment

- Rocky Linux 9.8, Slurm, Python 3.13.5 (module `Python/3.13.5-GCCcore-14.3.0`), project path `/data/leuven/394/vsc39468/drone-swarm`.
- Verified packages: PyTorch 2.14.1, Stable-Baselines3 2.9.2, Gymnasium 1.4.0, TensorBoard 2.21.0. Matplotlib was missing and had to be installed.
- Activate the project venv before every session:

```bash
cd /data/leuven/394/vsc39468/drone-swarm
source .venv/bin/activate
which python
python -m pip show torch stable-baselines3 gymnasium tensorboard matplotlib pillow
python -m pip check
```

### 2026-10-31.3 Problems faced and solutions

1. **Packages installed outside the venv, exceeding disk quota.** `pip install -r requirements.txt` ran with system Python 3.9, which wrote about 3.0 GB under `~/.local` and failed with `Errno 122`. The project `.venv` (about 5.6 GB) was already usable.
   - Solution: activate `.venv` first; the pip cache was purged. Nothing in `~/.local` was removed, since it may belong to other work.
   - Lesson: check `which python` and `$VIRTUAL_ENV` before any install.

2. **Wall reward silently inactive.** `wall_proximity` stayed in `REWARD_TERMS` and `wall_weight=3.0` stayed in the config, but the assignment `terms["wall_proximity"] = ...` was missing from `environment.py`. Every episode logged `r_wall_proximity = 0.0`.
   - Solution: restore the calculation. Found by checking the CSV and grepping `environment.py`.
   - Lesson: a config value or term label does not prove a reward is active. Check per-episode `r_*` values and compare their sum with the return.

3. **"Changes vs defaults" misleading.** The report compares against current `config.py` defaults, so values that became defaults do not appear as overrides.
   - Lesson: treat each run's `config.json` as the source of truth.

4. **Generic automatic diagnosis.** Some reports suggested obstacle avoidance for Stage 1, which has zero obstacles.
   - Lesson: use outcome codes, collision positions, and the exact config instead of the generic text.

5. **Separation tuning counterproductive.** Margin 0.5 gave 17% success with 24 drone collisions. Weight 2.0 gave 0% success with 30 drone collisions.
   - Lesson: more repulsion is not automatically safer. Diagnose action and observation dynamics and reward alignment first.

6. **Strong action coupling counterproductive.** Residual scale 0.25 gave 13% success and 20 boundary collisions.
   - Lesson: a more constrained action space is not better by default. Keep `action_residual_scale=1.0` unless a multi-seed experiment shows otherwise.

7. **Single metric misleads.** Margin 0.5 had the lowest formation error (0.107) and the worst mission outcome.
   - Lesson: judge success, outcomes, minimum pair distance, worst-drone distance, and reward terms together.

8. **Single-seed evidence is weak.** All runs used training seed 42. A 60% point estimate with CI [42, 75] is preliminary.

9. **Branch deletion blocked by a worktree.** `git branch -d testing` failed because the branch was checked out in another worktree.
   - Solution: check `git worktree list` and switch away first. Do not force-delete.

### 2026-10-31.4 Experiment results

All full Stage 1 runs: seed 42, 300,000 timesteps, 30 final evaluation episodes, 95% Wilson intervals. Random scored 0% in these evaluations; Greedy scored 100% (95% CI [89, 100]).

| Experiment | Main configuration/change | PPO success | Boundary | Drone collision | Mean return | Mean steps | Mean final distance | Worst final distance | Formation error |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 2,000 steps; no obstacles; known target | 0% (3 episodes) | 1 | 2 | -33.1 | 94 | 8.06 | 8.87 | Not recorded |
| stage1_seed42 | Original Stage 1 reward | 30% [17–48] | 11 | 10 | 14.8 | 88 | 3.17 | 3.60 | Not recorded |
| stage1_wallfix_seed42 | Wall weight 1.0 | 0% [0–11] | 25 | 5 | -2.5 | 88 | 3.74 | 4.16 | Not recorded |
| stage1_wall_weight3_seed42 | Wall weight 3.0; no formation reward | 27% [14–44] | 17 | 5 | 7.3 | 99 | 2.42 | 2.68 | Not recorded |
| stage1_formation_seed42 | Formation weight 0.5; wall calculation accidentally missing | 47% [30–64] | 8 | 8 | 21.2 | 85 | 3.03 | 3.30 | 0.189 |
| stage1_formation_wallfix_seed42 | Formation 0.5 and wall reward active at 3.0 | 60% [42–75] | 5 | 7 | 23.0 | 87 | 2.48 | 2.75 | 0.209 |
| stage1_separation_margin05_seed42 | Separation margin 0.5 | 17% [7–34] | 1 | 24 | -2.6 | 36 | 6.91 | 7.18 | 0.107 |
| stage1_sepweight2_seed42 | Separation weight 2.0; margin 0.3 | 0% [0–11] | 0 | 30 | -13.2 | 18 | 8.86 | 9.16 | 0.111 |
| stage1_action_coupling_seed42 | Action residual scale 0.25 | 13% [5–30] | 20 | 6 | -11.2 | 106 | 2.30 | 2.92 | 0.227 |

Notes on each run:

- **smoke:** validated the pipeline only, not learning.
- **stage1_seed42:** success was rising during training, but many boundary and drone collisions remained.
- **stage1_wallfix_seed42:** wall weight 1.0 was worse than the baseline and was not kept.
- **stage1_wall_weight3_seed42:** a meaningful wall penalty, but overall success was not clearly better than baseline.
- **stage1_formation_seed42:** an invalid test of formation plus wall reward, because the wall term was inactive (`r_wall_proximity = 0` in every episode).
- **stage1_formation_wallfix_seed42:** best result. Wall term non-zero: successful episodes averaged about -1.22, boundary episodes about -9.89.
- **stage1_separation_margin05_seed42:** formation error improved but missions collapsed. Do not retain margin 0.5.
- **stage1_sepweight2_seed42:** every episode ended in a drone collision, typically after 8 to 14 steps. Reject.
- **stage1_action_coupling_seed42:** reject 0.25 as a control setting.

Results from before the vectorised raycast, blended-progress work, relative-position scaling, per-phase change, and reward normalisation are not directly comparable with new runs.

### 2026-10-31.5 What the results establish and do not

Supported:

- The experiment pipeline runs end to end and produces reports, models, CSVs, and plots.
- Greedy reaches the target with all three drones inside the success radius in every Stage 1 evaluation.
- PPO learned some navigation, with a best single-seed result of 60% in Stage 1.
- Pairwise formation reward together with correctly active wall proximity coincided with the best result.
- Larger separation margin or weight, and action coupling at 0.25, all performed poorly in the tested seed-42 runs.

Not established:

- That the 60% result reproduces across training seeds.
- That formation and wall terms alone caused the improvement.
- That PPO beats Greedy.
- That the policy generalizes to random starts, hidden targets, or obstacles.
- That the best configuration reaches the proposed approximately 80% curriculum gate across at least three seeds.

### 2026-10-31.6 Reference configuration (best-known Stage 1)

Frozen as the comparison baseline. Command used to reproduce:

```bash
python run_experiment.py \
  --stage 1_known_fixed \
  --name stage1_best \
  --seed 42 \
  --episodes 30 \
  --no-gif \
  --set env.start_jitter=0.0 \
       env.action_residual_scale=1.0 \
       reward.wall_weight=3.0 \
       reward.wall_margin=1.0 \
       reward.formation_weight=0.5 \
       reward.separation_weight=1.0 \
       reward.separation_margin=0.3
```

Settings: `env.action_residual_scale=1.0`, `reward.wall_weight=3.0`, `reward.wall_margin=1.0`, `reward.formation_weight=0.5`, `reward.separation_weight=1.0`, `reward.separation_margin=0.3`, `reward.progress_weight=2.0`, `reward.cohesion_weight=0.5`.

Note: the Stage 1 preset does not set `env.start_jitter`, so the default 0.3 applies unless overridden as above. The best-known runs used `n_envs=1` and `n_steps=1024`; the current defaults are `n_envs=4` and `n_steps=256`.

### 2026-10-31.7 Recommended next steps at handoff

1. Repeat the reference configuration across at least three training seeds (for example 42, 43, 44), with the same `--name` so `compare_runs.py` groups them. Use 100 evaluation episodes if runtime allows.
2. Compare with `python compare_runs.py --last 3 --md`.
3. Do not advance the curriculum on one seed reaching 60%. The proposed gate is about 80% across at least three seeds. If that gate is unrealistic, document a revised one.
4. Change one hypothesis per experiment. Do not change separation, learning rate, and duration together.
5. Check `eval_curve.csv` and `training_curves.png` before blaming the reward. Periodic evaluation uses 10 episodes and is noisy.
6. Record for each run: hypothesis, exact `config.json`, seed and timesteps, evaluation seed range, success with CI, outcome counts, mean and worst distances, formation error, minimum pair distance, and per-outcome reward breakdown.

Do not delete failed runs; they are part of the record.

### 2026-10-31.8 Bottleneck-aware progress (proposed, not implemented)

Current progress uses the mean reduction in target distance, which can be positive while one drone moves away. The proposal blends mean and worst-drone progress:

```python
mean_progress = float(np.mean(old_dist - target_dist))
worst_progress = float(np.max(old_dist) - np.max(target_dist))
progress = 0.5 * mean_progress + 0.5 * worst_progress
```

Before running it, add tests for: (a) all drones approaching equally, where the blended reward should equal the common progress; (b) two drones approaching while the farthest moves away, where the blend should be lower than mean-only progress. Change only the aggregation, keep the reference settings, and compare against the multi-seed baseline.

Status at the 2026-11-05 review: the code comment near the progress calculation implied this blend existed. It did not. The comment was corrected (see 2026-11-05.2).

### 2026-10-31.9 Git workflow (handoff instructions, to run on the cluster)

1. Inspect state: `git branch --show-current`, `git status --short`, `git worktree list`, `git branch -vv`. Confirm the current branch is `experimental/wall-proximity`.
2. Stage only intended source and documentation files. Do not add `.venv/` or `runs/`.
3. Commit with a descriptive message, for example "Add wall and formation rewards for swarm coordination".
4. Update the target: `git fetch origin`, then `git switch advanced` and `git pull --ff-only origin advanced`. If `advanced` is checked out in another worktree, merge from that worktree.
5. Merge with `git merge --no-ff experimental/wall-proximity`. Resolve conflicts deliberately; do not use `-X ours` or `-X theirs` without understanding them.
6. Validate after the merge: `python -m py_compile ...`, `python run_experiment.py --list-params`, and the 2,000-step smoke run. Push only after review.

### 2026-10-31.10 Limitations open at handoff

- No automated unit or integration test suite.
- No periodic checkpoints or resume after interruption; only the final model is saved.
- No frequency or duration metric for cohesion or formation violations; only mean errors are recorded.
- The policy has no temporal memory beyond the searched map.
- Vectorised raycast was not benchmarked against the older loop version on the cluster.
- `ppo.n_envs > 1` (multi-process) was not validated on the cluster.
- Thirty evaluation episodes give wide confidence intervals; use more for final reporting.
- `min_obstacle_clearance = inf` is expected in obstacle-free runs.

---

## 2026-11-05: Stage 3 review, fixes, and documentation restructure

### 2026-11-05.1 Stage 3 review

Scope: the workspace contains no Stage 3 run folders, so these findings come from code review and rough arithmetic. They need confirmation against a real Stage 3 run (`training_episodes.csv`, `report.md`).

Problems identified:

1. **Shaping punishes the spread search needs.** Formation (pairwise distance kept near 0.87 m) and cohesion (1.0 m from centroid) were tuned for Stage 1 and apply during search. Rough estimate: pairs 2 m apart cost about $0.75 \times 1.1 \approx 0.85$ per step, while exploring earns roughly 0.3 to 0.5 per step. The policy learns to stay clustered.
2. **Crashing can be cheaper than timing out.** The collision penalty is a one-time -10, while per-step penalties continue until the episode ends. Over 800 steps, a steady cost $c$ exceeds 10 once $c > 0.0125$. After the map is covered, remaining steps only cost reward.
3. **Finding the target stops exploration, and the bonus is small.** Full-map exploration is worth about 67 in mean-return units; the detection bonus was 10. `target_found_bonus` was also a plain class attribute, so it was missing from `config.json` and `--list-params`.
4. **"Searched" is a weaker test than "found".** A cell centre can be up to 0.71 m from the target. A fully searched map can therefore leave the target undetected, and then nothing guides the policy.
5. **Frame flips at detection.** The observation and action frame change in one step when the target is found. Lower priority; not changed.
6. **Config, code, and docs disagree.** `device="cuda"` fails on CPU-only machines; defaults differ from the reference configuration; collision penalty applies to all drones; the progress comment was wrong; the README gave 124 observation values where the code gives 126; there is no warm-start option.
7. **Greedy is a weak search benchmark.** It keeps the formation offsets and so searches as a cluster.

### 2026-11-05.2 Changes implemented

| File | Location | Change | Reason |
|---|---|---|---|
| `training.py` | `device` | `"cuda"` changed to `"auto"` | Problem 6: fails on CPU-only machines |
| `config.py` | `EnvConfig` | Added `search_cohesion_radius: float = 2.0` | Problem 1: looser cohesion limit before detection |
| `config.py` | `RewardConfig` | `target_found_bonus` made an annotated field, default 10 changed to 70 | Problem 3: recorded in config; above the roughly 67 a full map can yield |
| `config.py` | `RewardConfig` | Comment added to `formation_weight` noting it applies after detection | Problem 1 |
| `config.py` | `validate` | Rejects `target_sense_range <= explore_cell * sqrt(2) / 2` | Problem 4: keeps the searched-cell reach positive |
| `environment.py` | `_cohesion_excess` | Uses `search_cohesion_radius` until the target is found | Problem 1 |
| `environment.py` | `_update_search` | Cells marked searched within `target_sense_range - explore_cell * sqrt(2) / 2` | Problem 4: a fully searched map implies detection |
| `environment.py` | `step` | Formation term added only when `target_found` | Problem 1 |
| `environment.py` | progress comment | Replaced the incorrect worst-drone blend comment | Problem 6 |

### 2026-11-05.3 Behaviour impact

- **Stage 1 and Stage 2:** unchanged in principle. The target is known at reset (`target_sense_range = 20`), so the formation, cohesion, and detection-bonus paths behave as before. The searched-map threshold also changes, but the map is zeroed once the target is found.
- **Stages 3 to 5:** formation and cohesion behave differently before detection; the detection bonus is 70 instead of 10; the searched map is stricter. These changes are untested.
- **Logging:** `cohesion_error` in hidden-target runs now uses the search radius before detection.
- **Device:** training now uses GPU when available and CPU otherwise.

### 2026-11-05.4 Verification

- Passed: syntax check (`py_compile`) of all Python modules in the project.
- Not run: behavioural checks. A test script was prepared and the run was cancelled, so no training or evaluation has been executed on these changes.
- Temporary environment: `%TEMP%\dsv_test` (outside the project) holds numpy, gymnasium, and matplotlib. It can be deleted.

### 2026-11-05.5 Not changed (open items)

- **Collision penalty:** applies to every drone (`environment.py`, collision block). `culprits` is computed but unused. The README says only the crashing drone should be penalised. Decide before the next run.
- **Defaults vs reference:** default `formation_weight` is 0.75 versus 0.5 in the reference configuration; default `n_envs`/`n_steps` are 4/256 versus 2/512 in the README.
- **Stage 1 jitter:** the Stage 1 preset does not set `start_jitter`, so the default 0.3 applies.
- **Warm start:** no mechanism to load a previous stage's model. The observation size changed from 124 to 126, so older models cannot be loaded.
- **Crash versus timeout balance:** problem 2 is conditional. Check the "PPO return by outcome" table in the next Stage 3 report.
- **Frame flip at detection:** problem 5 needs an experiment before any change.
- **Greedy baseline:** problem 7 is left as a lower bound.

### 2026-11-05.6 Next steps

1. Run a Stage 3 smoke test, then three seeds, with the changed code.
2. Check `r_formation`, `r_exploration`, `r_target_found`, `r_collision`, and the outcome table in `report.md`.
3. Decide the collision-penalty scope and the defaults questions above.
4. Record each Stage 3 run as a new entry in this file.

---

## 2026-11-14: GPU environment

Working branch: `martin-branch`.

### 2026-11-14.1 Findings

- `.venv` holds `torch 2.15.0+cpu`, a build without CUDA, so it can never use a GPU. A separate venv is required.
- `.venv` was created as `.venv-cpu` and later renamed. Its `bin/pip` launcher still points to `.venv-cpu/bin/python`, which no longer exists, so a bare `pip` fails. Use `python -m pip` in that venv. Not changed.
- The default PyPI `torch 2.15.0` wheel is built for CUDA 13.2 and bundles the CUDA runtime as `nvidia-*` packages. wICE nodes run NVIDIA driver 615.71, which supports CUDA 13.
- CUDA 13 dropped support for Pascal (P100) and Volta (V100). Usable: wICE A100 (`gpu_a100`, `gpu_a100_debug`, `interactive`) and H100 (`gpu_h100`); mindwell B200 (`gpu_b200`) and RTX 5000 Ada (`interactive`).
- `$VSC_HOME` is 3 GB. The CUDA wheels are several GB, so pip's cache goes to `$VSC_SCRATCH/pip-cache`.
- `metadata.json` was overwritten at the end of each run with only status fields, which dropped `python` and `command`.

### 2026-11-14.2 Changes

| File | Change | Reason |
|---|---|---|
| `cluster/setup_gpu_venv.sh` | New. Builds `.venv-gpu` with `torch==2.15.0` (CUDA 13.2) and the same pinned versions as `.venv` | Reproducible GPU environment |
| `cluster/train_gpu.slurm` | New. One-GPU A100 batch job that passes its arguments to `run_experiment.py` and sets `ppo.device=cuda` | Batch GPU training |
| `config.py` | `PPOConfig.device: str = "auto"`, validated as `auto`, `cpu` or `cuda[:i]` | Device is now an explicit, recorded choice; replaces the hard-coded `"auto"` in `training.py` |
| `training.py` | Passes `ppo.device` to PPO and prints the device in use | Shows where the network runs |
| `run_experiment.py` | `metadata.json` gains a `hardware` block (host, Slurm job, venv, torch build, GPU name); end-of-run updates merge instead of overwrite | CPU and GPU runs can be told apart; fixes lost metadata |
| `.gitignore` | Adds `.venv*/` and `slurm-*.out` | Keep venvs and job logs out of git |
| `README.md` | Device row, new files, section 10.1 on both venvs | Documentation |

Training behaviour is unchanged: the default device is still `auto`.

### 2026-11-14.3 Verification

- Passed on CPU node k28i14 with `.venv-gpu`: `pip check`, `py_compile`, `--list-params`, and a 2,000-step Stage 1 smoke run (`runs/20261114_183348_gpu_env_smoke`). Without a GPU, `auto` falls back to CPU as intended. `ppo.device=gpu` is rejected before a run folder is created.
- Not yet verified at the time: `torch.cuda.is_available()` on a GPU node, and a training run with `ppo.device=cuda`. Both were verified later the same day (2026-11-14.5).

### 2026-11-14.4 Expected performance (to measure)

The policy is a [64, 64] MLP, and most of the wall-clock time goes to stepping the environment in Python on CPU. Stable-Baselines3 documents that PPO with an MLP policy usually runs as fast or faster on CPU, because moving small batches to the GPU has overhead. The main gain from a GPU node may be its CPU cores (18 per A100), which allow a larger `ppo.n_envs`. Before moving all training to GPU nodes, time the same short run with `ppo.device=cpu` and `ppo.device=cuda` on the GPU node, and record both here.

### 2026-11-14.5 Benchmark: CPU versus GPU device

Stage 3, 50,000 timesteps, 3 evaluation episodes, default PPO settings (`n_envs=4`, `n_steps=256`, `[64, 64]`).

| Run | Node | Venv | `ppo.device` | Wall time | User CPU time |
|---|---|---|---|---|---|
| `bench_cpu` | k28i14 (CPU) | `.venv` | `cpu` | 1 min 18 s | 38 s |
| `bench_cuda` | l11i31 (mindwell `interactive`, RTX 5000 Ada) | `.venv-gpu` | `cuda` | 1 min 40 s | 46 s |

- The GPU run was about 28% slower, as SB3's warning predicts for MLP policies. Each rollout step sends 12 observation rows to the GPU and the actions back to the CPU. Each gradient step processes 192 samples through a 64-wide network. Kernel-launch and transfer latency outweighs the computation.
- Part of the gap is fixed startup cost: CUDA initialisation, loading the CUDA libraries from `/data`, and matplotlib/fontconfig rebuilding caches because the GPU session had no writable `$HOME/.cache`. `cluster/train_gpu.slurm` now sets `XDG_CACHE_HOME` and `MPLCONFIGDIR` on scratch.
- Confounded: different clusters, CPU generations (icelake versus graniterapids) and torch builds. The GPU session was opened with `--cpus-per-task=1`, so 4 environment processes plus the PPO process shared one core. A same-node check (`ppo.device=cpu` on the GPU node) has not been run.
- Correction: this table first listed l11i31 as an A100 node. `metadata.json` (`hardware.gpu_name`) shows an NVIDIA RTX 5000 Ada on the mindwell cluster.
- Decision: train with `ppo.device=cpu`. Speed comes from more environment processes (`ppo.n_envs`) on more cores, not from the GPU. Keep `.venv-gpu` for larger networks or a batched environment.

### 2026-11-14.6 Interactive core limit and CPU batch jobs

- The interactive session (partition `interactive`) gives 2 cores, which the default `ppo.n_envs=4` already oversubscribes. The account `lp_deeplearning_gt_2026b` has no per-user core limit; the `batch` partition offers 72-core icelake nodes for up to 3 days.
- Added `cluster/train_cpu.slurm`: 9 cores, `.venv`, `ppo.device=cpu`, `ppo.n_envs = cores - 1`, and `ppo.n_steps = 1024 / n_envs`, which keeps 1,024 environment steps per update as with the 4 × 256 defaults.
- `sbatch --test-only` on 2026-11-14 estimated a start within about 20 minutes for the CPU job and the next afternoon for an A100 job.
- Speed-up measured in 2026-11-14.7. Still open: shorter rollouts (`n_steps=128`) change GAE bootstrapping; check that learning curves match the 4 × 256 setting on one seed.

### 2026-11-14.7 Batch-job throughput, progress tracking, GPU session details

Training throughput, from the last row of each run's `sb3_log/progress.csv` (`time/total_timesteps / num_drones` divided by `time/time_elapsed`). This covers the training phase only; evaluation, plots and report come after.

| Run | Where | Cores | `n_envs` × `n_steps` | Device | Env steps/s |
|---|---|---|---|---|---|
| `20261114_184159_bench_cpu` | wICE `interactive`, k28i14 (icelake) | 2 | 4 × 256 | cpu | 796 |
| `20261114_183914_bench_cuda` | mindwell `interactive`, l11i31 (graniterapids, RTX 5000 Ada) | 1 requested | 4 × 256 | cuda | 627 |
| `20261114_190757_search` | wICE `batch`, l33c22n3 (icelake), job 62693806 | 9 | 8 × 128 | cpu | 1,556 |

- The 9-core batch job trains about 1.95 times faster than the 2-core interactive session. Scaling is sub-linear (8 environment processes against 4): the single PPO process does the gradient updates serially while the workers wait.
- First batch job 62693337 (same command): `COMPLETED`, 11 min 33 s elapsed, 22 min 05 s total CPU time over 9 allocated cores, so about 21% average core utilisation over the whole job. The evaluation, diagnosis and plotting phases run in one process. Its run folder, `20261114_185431_search`, was removed and the job resubmitted as 62693806.
- The account `lp_deeplearning_gt_2026b` (balance 729,668 credits on 2026-11-14) has no per-user limits in its QOS. The wICE `interactive` QOS caps a session at 8 CPUs and 1 GPU; the session in use was opened with 2. On mindwell, `interactive` has billing weight 0 for CPUs and RTX 5000 Ada GPUs, and a 16 h time limit.
- The GPU session is opened from inside the wICE interactive job, so the shell holds `SLURM_JOB_ID` and related variables. `srun` would use them and try to start a step inside the current wICE job; they are removed with `env -u ...` before `srun --clusters=mindwell`. The exact command is in README 10.5.
- `.venv-gpu` was built on an icelake node with the icelake build of the Python module and runs unchanged on the graniterapids mindwell node. `torch.cuda.is_available()` returned `True` there, with driver support for CUDA 13.2.
- On the mindwell node, `$HOME/.cache` was not writable (matplotlib reported `/.cache`), so matplotlib and fontconfig rebuilt their caches in every process. Both Slurm scripts now set `XDG_CACHE_HOME` and `MPLCONFIGDIR` under `$VSC_SCRATCH`.
- In the first batch job, `slurm-62693337.out` showed only the header lines during training: Python block-buffers stdout when it is not a terminal. Both Slurm scripts now run `python -u`.
- Added `cluster/progress.py`, which reads `sb3_log/progress.csv` (written by SB3 after every update) and `config.json`, and reports environment steps (SB3 timesteps divided by `num_drones`), elapsed time, a linear estimate of the remaining time, last-rollout success and the latest periodic evaluation. Standard library only.
- `run_experiment.py` prints run folders via `Path.resolve()`, which shows `/vsc-hard-mounts/leuven-data/...`. `/data/leuven` is a symbolic link to `/vsc-hard-mounts/leuven-data` (the same inode was confirmed), so there is one folder. Not changed.
- README section 10 was restructured into 10.1 (venvs), 10.2 (interactive versus batch), 10.3 (batch jobs), 10.4 (progress tracking) and 10.5 (GPU).

---

## 2026-11-14 (evening): Stage 3 diagnosis and fix

Working branch: `martin-branch`. Code state before this entry: commit 306ce9a.

Conventions used in this entry:

- All training ran as CPU batch jobs with `cluster/train_cpu.slurm`: 9 cores, `ppo.n_envs=8`, `--no-gif`.
- Final evaluation: 100 episodes with the deterministic policy, 95% Wilson intervals. Evaluation seeds start at `seed + 1000`.
- **Found:** share of episodes in which the target is detected. **Conversion:** share of those episodes that end in success.
- Outcome counts are given as boundary / drone collision / timeout out of 100 episodes.
- "Before" is run `20261114_190757_search` (seed 42, 1M steps).

### 2026-11-14.8 Summary

- Stage 3 went from 36% to 43% success to 94%, 93% and 95% on three training seeds at 2M steps. It passes the 80% gate.
- Four causes were found. The evidence for each is in 2026-11-14.9 to 2026-11-14.13.

| # | Cause | Fix |
|---|---|---|
| 1 | The 36 border cells of the searched map could only be searched from inside the wall zone, and can never contain the target | Cells that cannot contain the target start as searched |
| 2 | Search let the swarm spread out; at detection a formation penalty about five times the progress reward switched on | Formation and cohesion are shaped the same way in both phases |
| 3 | In the world frame the policy ignored the waypoint and flew an open-loop sweep indexed by the map | During search, one shared frame whose +x points from the swarm centroid to the waypoint, updated every step; waypoint reward per drone |
| 4 | 1,024 environment steps per PPO update and a constant learning rate left the outcome to the training seed | 2,048 steps per update; linear learning-rate decay for stages 3 to 5; Stage 3 budget 2M steps |

- Side finding: Stage 2 does not reproduce its 100% of 2026-11-05 on the current code (2026-11-14.15). Open.
- Cost: 35 batch jobs, 9,858 credits, 67.3 core-hours (2026-11-14.17).

### 2026-11-14.9 Round 0: diagnosis before any change

No training. The saved model of the "before" run and the Greedy baseline were rolled out on the same 300 episodes (seeds 1042 to 1341), and every failure was classified by phase.

| Policy | Success | Found | Conversion | Failures |
|---|---|---|---|---|
| Greedy | 63.0% (189/300) | 63.0% | 100% | 111 boundary, all during search |
| PPO | 36.3% (109/300) | 65.0% | 55.9% | Search: 64 boundary, 40 drone collision, 1 timeout. Approach: 40 boundary, 35 drone collision, 11 timeout |

Findings:

1. **Border cells (cause 1).** Greedy is scripted and scored 99% in the runs of 2026-11-05. It scored 73% (22/30) in the "before" run. The only environment changes in between were those of 2026-11-05.2.
   - With `cell_reach = 1.5 - 0.71 = 0.79 m`, a border cell (centre 0.5 m from a wall) can only be searched from inside the wall-penalty zone. For a corner cell the best position has 0.91 m clearance, against `wall_margin = 1.0`.
   - In 108 of Greedy's 111 crashes its goal was a border cell.
   - The target is sampled in [1, 9]², so the 36 border cells cannot contain it.
2. **Hand-over at detection (cause 2).** PPO conversion against the swarm's largest pairwise distance at the moment of detection:

   | Spread at detection | Episodes | Conversion | Main failure |
   |---|---|---|---|
   | below 1.5 m | 57 | 74% | drone collision |
   | 1.5 to 2.5 m | 55 | 71% | drone collision |
   | 2.5 to 3.5 m | 36 | 53% | mixed |
   | above 3.5 m | 47 | 19% | boundary (29 of 47) |

   - Mean formation error at detection was 1.29 m: a formation penalty of 0.97 per step against a maximum progress reward of 0.20 per step.
   - Failed approaches accumulated -102.4 in formation and cohesion penalties, successful ones -16.7. A collision costs -10, so ending the episode was cheaper than regrouping.
3. **Waypoint not followed.** The `search_waypoint` reward was negative in 90% of episodes (mean -3.27): the centroid ended farther from its waypoints.
4. **Late degradation (cause 4).** Training success per 100k steps peaked at 48% (600k to 700k) and ended at 36%. The found rate fell from 73% to 59%. Each PPO update held about 3 to 9 finished episodes.
5. Stage 2 (known target) had scored 100% on two seeds on 2026-11-05, so the approach itself was learnable.

### 2026-11-14.10 Round 1: map and reward fixes

Hypothesis: removing causes 1 and 2, with a stronger waypoint reward, turns search into a formation flight to the waypoint.

Changes:

| File | Change |
|---|---|
| `config.py`, `environment.py` | New `env.skip_targetless_cells = True`: cells outside the target sampling square start as searched (64 of 100 remain) |
| `config.py`, `environment.py` | New `reward.search_formation_weight = 0.75`: formation is shaped before detection too. `env.search_cohesion_radius` 2.0 to 1.0 |
| `config.py`, `environment.py` | `search_waypoint` reward: each drone's own distance to the waypoint instead of the centroid's. `reward.search_waypoint_weight` 1.0 to 2.0 |
| `config.py`, `training.py` | New `ppo.lr_schedule` (`constant` by default, or `linear`) |
| `environment.py` | `info["searched_fraction"]` counts only the cells that can contain the target |

Scripted checks before training (300 episodes): Greedy 63% to 100%; a scripted policy that flies in formation to the waypoint the network observes, 100%.

Results (jobs 62694480 to 62694486, 1M steps):

| Run folder | Seed | Change against this round's defaults | Success [95% CI] | Found | Conversion | Outcomes |
|---|---|---|---|---|---|---|
| `20261114_195052_s3fix` | 42 | none | 50% [40–60] | 69% | 72% | 34 / 16 / 0 |
| `20261114_195155_s3fix` | 43 | none | 50% [40–60] | 55% | 91% | 39 / 11 / 0 |
| `20261114_195254_s3fix` | 44 | none | 32% [24–42] | 55% | 58% | 44 / 23 / 1 |
| `20261114_195055_s3fix_lr_linear` | 42 | `ppo.lr_schedule=linear` | 35% [26–45] | 41% | 85% | 7 / 50 / 8 |
| `20261114_195101_s3fix_abl_border_cells` | 42 | `env.skip_targetless_cells=false` | 28% [20–37] | 38% | 74% | 20 / 50 / 2 |
| `20261114_195055_s3fix_abl_free_search` | 42 | `reward.search_formation_weight=0`, `env.search_cohesion_radius=2.0` | 48% [38–58] | 68% | 71% | 13 / 39 / 0 |
| `20261114_195055_s3fix_abl_no_waypoint_reward` | 42 | `reward.search_waypoint_weight=0` | 38% [29–48] | 60% | 63% | 45 / 17 / 0 |

Findings:

- Not solved. Conversion rose from 56% to between 58% and 91%. The found rate stayed between 55% and 69%.
- The waypoint reward was still negative (-5 to -6 per episode), although it was now per drone and twice as strong.
- Input sensitivity of the trained policies during search (mean action magnitude about 0.6), for the "before" model and two `s3fix` models:

  | Input changed | Mean change in action |
  |---|---|
  | Waypoint vector reversed | 0.07 to 0.09 |
  | Own position shuffled between states | 0.21 to 0.23 |
  | Searched map shuffled between states | 0.80 to 0.88 |

- The mean cosine between a drone's velocity and the direction to its waypoint was -0.35 to -0.55. `trajectories.png` of `20261114_195155_s3fix` shows the same arc in every episode.
- Inside the last 0.6 m before a wall the policy's mean acceleration relative to the wall was about zero (+0.04 at 0 to 0.3 m, -0.02 at 0.3 to 0.6 m).
- Interpretation: about 50 active map bits and the absolute position dominate a two-number waypoint vector of magnitude about 0.2. The policy learns a sweep indexed by map and position. Stage 2 never has to read a direction, because its frame puts the target on +x.
- The ablations of this round ran while the policy ignored the waypoint. They do not show what the formation and waypoint-weight changes contribute. The border-cell fix is supported independently by the Greedy result.

### 2026-11-14.11 Round 2: search frame pointing at the waypoint

Hypothesis: with +x pointing at the waypoint, as it points at the target in Stage 2, the policy follows the waypoint.

Changes:

| File | Change |
|---|---|
| `config.py`, `environment.py` | New `env.waypoint_frame = True`: during search all drones share one frame whose +x points from the swarm centroid to the waypoint. In this round it was fixed when the waypoint was chosen |
| `config.py`, `environment.py` | New `env.observe_search_map = True`: `false` replaces own position and the searched map with zeros |

- A first version with one frame per drone was rejected by the scripted checks before any job ran: Greedy 49%, waypoint follower 19%, and a constant "+x" action collided in 150 of 150 episodes. Drones 0.87 m apart that look at a waypoint 1 to 1.5 m away have frames 40° to 60° apart, so "+x" makes them converge on every waypoint.
- Scripted checks with the shared frame: Greedy 100%, waypoint follower 100%. A constant action (+1, 0) found the target in 39% of episodes, against 7% in the world frame.

Results (jobs 62694613 to 62694617, 1M steps):

| Run folder | Seed | Change against this round's defaults | Success [95% CI] | Found | Conversion | Outcomes |
|---|---|---|---|---|---|---|
| `20261114_202052_s3frame` | 42 | none | 65% [55–74] | 66% | 98% | 30 / 5 / 0 |
| `20261114_202153_s3frame` | 43 | none | 65% [55–74] | 68% | 96% | 31 / 3 / 1 |
| `20261114_202055_s3frame_nomap` | 42 | `env.observe_search_map=false` | 44% [35–54] | 88% | 50% | 7 / 49 / 0 |
| `20261114_202153_s3frame_nomap` | 43 | `env.observe_search_map=false` | 57% [47–66] | 98% | 58% | 27 / 16 / 0 |
| `20261114_202053_s3_worldframe_nomap` | 42 | `env.waypoint_frame=false`, `env.observe_search_map=false` | 0% [0–4] | 19% | 0% | 0 / 100 / 0 |

Findings:

- The policy now follows the waypoint: the waypoint reward rose to +12.3 and +12.7 per episode, conversion to 96% and 98%, and drone collisions fell to 3% and 5%.
- Hiding the map without the frame does nothing (0%). The frame is what matters.
- Hiding the map with the frame raised the found rate (88% and 98%) and lowered conversion (50% and 58%). The default stays `true`. Not explained; one reading is that the map bits tell the network which phase it is in.
- Remaining failure: about 30% boundary crashes during search. On 200 episodes of `20261114_202153_s3frame` there were 52 of them.
  - In 96% the crashing drone was flying away from its waypoint, at +0.92 m/s along the frame's +x.
  - The median time since the waypoint last changed was 42 steps.
  - The swarm had passed the waypoint without any drone coming within 0.79 m of it. The fixed frame kept pointing along the old line, into the wall.

### 2026-11-14.12 Round 3: frame tracks the waypoint

Change: `environment.py`, `_refresh_search_goal`: the search frame is recomputed every step, so +x always points from the centroid to the waypoint.

Scripted checks (300 episodes):

- Greedy 100% (mean 164 steps, maximum 406). Waypoint follower 100% (mean 179 steps, maximum 444; the limit is 800).
- A constant action (+1, 0) found the target in 79% of episodes.
- Stages 1 and 2: observations and rewards identical step for step to the code before this entry (1,530 and 1,362 steps compared).

Results (jobs 62694807 to 62694810, 1M steps):

| Run folder | Seed | Change against this round's defaults | Success [95% CI] | Found | Conversion | Outcomes |
|---|---|---|---|---|---|---|
| `20261114_204344_s3track` | 42 | none | 88% [80–93] | 96% | 92% | 0 / 11 / 1 |
| `20261114_204441_s3track` | 43 | none | 70% [60–78] | 73% | 96% | 13 / 16 / 1 |
| `20261114_204542_s3track` | 44 | none | 33% [25–43] | 62% | 53% | 44 / 21 / 2 |
| `20261114_204344_s3track_nomap` | 42 | `env.observe_search_map=false` | 65% [55–74] | 97% | 67% | 21 / 14 / 0 |

Findings:

- Seed 42 reached 95% training success in the last 20% of the run with no boundary crashes. The environment is learnable.
- Seed 43 reached 75% (500k to 600k) and fell back to between 53% and 61%. Its wall penalty per episode rose from -11 to -39.
- Seed 44 stayed near 30% from 400k onwards. At its boundary crashes the swarm was 2.39 m wide on average (the formation is 0.87 m).
- In all seeds `approx_kl` rose from 0.010 to 0.021 and the clip fraction from 0.10 to 0.21, while the action standard deviation fell from about 0.35 to between 0.12 and 0.15.
- Flight quality against Stage 2, final evaluation:

  | | Formation error | Separation penalty | Wall penalty | Rollout |
  |---|---|---|---|---|
  | Stage 2 PPO, 2026-11-05 (100%) | 0.015 and 0.055 m | -0.1 and -0.2 | -0.9 and -0.8 | 4 × 512 = 2,048 steps per update |
  | Stage 3 `s3track` | 0.15 to 0.24 m | -5.1 to -7.0 | -11.2 to -14.8 | 8 × 128 = 1,024 steps per update |

- The Stage 1 and Stage 2 runs that scored 100% on 2026-11-05 all used 2,048 steps per update. The batch script used 1,024, which is the open item of 2026-11-14.6. With episodes of about 200 steps, an 8 × 128 rollout holds about five episodes.

### 2026-11-14.13 Round 4: PPO update size and learning-rate schedule

No code change. Settings were passed with `--set`.

Results (jobs 62695036 to 62695041, 1M steps):

| Run folder | Seed | Setting | Success [95% CI] | Found | Conversion | Outcomes |
|---|---|---|---|---|---|---|
| `20261114_210448_s3track_n256` | 43 | `ppo.n_steps=256` | 43% [34–53] | 48% | 90% | 34 / 23 / 0 |
| `20261114_210551_s3track_n256` | 44 | `ppo.n_steps=256` | 95% [89–98] | 95% | 100% | 1 / 4 / 0 |
| `20261114_210451_s3track_lin` | 43 | `ppo.lr_schedule=linear` | 77% [68–84] | 93% | 83% | 1 / 20 / 2 |
| `20261114_210553_s3track_lin` | 44 | `ppo.lr_schedule=linear` | 75% [66–82] | 92% | 82% | 16 / 8 / 1 |
| `20261114_210451_s3track_n256_lin` | 43 | both | 83% [74–89] | 93% | 89% | 0 / 13 / 4 |
| `20261114_210553_s3track_n256_lin` | 44 | both | 97% [92–99] | 98% | 99% | 0 / 3 / 0 |

Findings:

- 2,048 steps per update alone: 43% and 95%. Still a lottery.
- Linear decay alone: 77% and 75%. No collapse and no stuck run, but a lower ceiling.
- Both: 83% and 97%, with training success rising through the whole run (88% and 94% in the last tenth).
- Two seeds per setting is thin. The combination was confirmed on more seeds in the next round.

### 2026-11-14.14 Rounds 5 and 6: shipped configuration and budget

Changes:

| File | Change |
|---|---|
| `config.py` | `ppo.n_steps` default 256 to 512, so the defaults give 4 × 512 = 2,048 steps per update |
| `cluster/train_cpu.slurm` | `N_STEPS = 2048 / N_ENVS` instead of `1024 / N_ENVS` (256 with 9 cores) |
| `config.py` | Stage presets 3, 4 and 5 set `ppo.lr_schedule=linear` |
| `config.py` | After round 6: `3_search` budget 1M to 2M steps |

- The configuration that the batch script builds for `--stage 3_search` was compared field by field with `20261114_210553_s3track_n256_lin`: no difference outside the `run` section. The two `s3track_n256_lin` runs therefore count as seeds 43 and 44 of the shipped configuration at 1M steps.
- The `s3_final` runs used the plain command, with no `--set`.

Stage 3 at 1M steps (jobs 62695145 to 62695147, and round 4 for seeds 43 and 44):

| Run folder | Seed | Success [95% CI] | Found | Conversion | Outcomes |
|---|---|---|---|---|---|
| `20261114_212336_s3_final` | 42 | 92% [85–96] | 97% | 95% | 1 / 7 / 0 |
| `20261114_210451_s3track_n256_lin` | 43 | 83% [74–89] | 93% | 89% | 0 / 13 / 4 |
| `20261114_210553_s3track_n256_lin` | 44 | 97% [92–99] | 98% | 99% | 0 / 3 / 0 |
| `20261114_212451_s3_final` | 45 | 88% [80–93] | 90% | 98% | 0 / 11 / 1 |
| `20261114_212553_s3_final` | 46 | 62% [52–71] | 68% | 91% | 21 / 16 / 1 |

Mean 84%. Seed 46 was still improving when the learning rate reached zero (training success 51%, 54%, 56% in the last three tenths).

Stage 3 at 2M steps (`--timesteps 2000000`; jobs 62695148, 62695484, 62695485):

| Run folder | Seed | Success [95% CI] | Found | Conversion | Outcomes |
|---|---|---|---|---|---|
| `20261114_212339_s3_final_2M` | 43 | 94% [88–97] | 98% | 96% | 1 / 5 / 0 |
| `20261114_215003_s3_final_2M` | 45 | 95% [89–98] | 99% | 96% | 0 / 5 / 0 |
| `20261114_214903_s3_final_2M` | 46 | 93% [86–97] | 96% | 97% | 0 / 6 / 1 |

- Pooled: 282 of 300 episodes, 94%. Seed 46 went from 62% to 93%.
- Independent check of the weakest seed (`20261114_214903_s3_final_2M`) on 300 other episodes (seeds 1042 to 1341): 89.7% (269/300), 95% CI [85.7–92.6]; found 95.3%; conversion 94.1%.
  - No boundary crash. Failures: 24 drone collisions (14 in the approach, 10 during search) and 7 timeouts.
  - Success by distance of the target from the nearest wall: 87% (1.0 to 1.5 m), 88% (1.5 to 2.5 m), 97% (above 2.5 m).
  - Mean cosine between velocity and waypoint direction during search: +0.64 (before: -0.55).
- The preset was changed to 2M after these runs. `--timesteps 2000000` and the preset give the same configuration; it was not run again.

To reproduce:

```bash
sbatch cluster/train_cpu.slurm --stage 3_search --name search --seed 43 --episodes 100 --no-gif
```

### 2026-11-14.15 Stage 2 regression (rounds 5 to 7)

One Stage 2 run was added to round 5 to check that the changed PPO defaults did not break a stage that worked. It scored 50%. Rounds 6 and 7 looked for the reason.

Results (jobs 62695149, 62695486, 62695487, 62696072, 62696074, 62696076, 62696078; 500k steps):

| Run folder | Seed | Setting | Success [95% CI] | Outcomes |
|---|---|---|---|---|
| `20261114_212344_s2_check` | 42 | current batch setting, 8 × 256 | 50% [40–60] | 0 / 0 / 50 |
| `20261114_221852_s2_check` | 43 | 8 × 256 | 37% [28–47] | 0 / 0 / 63 |
| `20261114_221941_s2_check` | 44 | 8 × 256 | 95% [89–98] | 2 / 2 / 1 |
| `20261114_214904_s2_check_n128` | 42 | batch setting before this entry, 8 × 128 | 73% [64–81] | 0 / 0 / 27 |
| `20261114_221847_s2_check_n128` | 43 | 8 × 128 | 47% [38–57] | 0 / 4 / 49 |
| `20261114_221941_s2_check_n128` | 44 | 8 × 128 | 98% [93–99] | 0 / 2 / 0 |
| `20261114_214904_s2_check_nov5_settings` | 42 | settings of the 2026-11-05 run: 4 × 512, `ppo.ent_coef=0.0`, `reward.formation_weight=0.5` | 84% [76–90] | 0 / 1 / 15 |

Findings:

- Stage 2 fails on two of three seeds under both batch settings. The regression is not caused by the changes of this entry: the Stage 2 environment is identical step for step, and the setting used before this entry fails the same seeds.
- Failure mode, the same in every failing run: the swarm arrives and hovers just outside the success radius. In `20261114_212344_s2_check` the timeouts ended with the swarm centre 0.75 m from the target, the worst drone at 1.06 m (success needs below 0.8 m), and a formation error of 0.08 m.
- Learning starts late. The runs above have 0% training success for the first 150k to 400k steps. In `20261114_212344_s2_check`, 93% to 99% of the episodes in the first 150k steps ended in a drone collision. The 2026-11-05 run had 25% success in its first 100k steps and 95% in the second.
- Changes between the 2026-11-05 run (code snapshot in `runs/20261105_110151_stage2_random_seed42`) and the code before this entry that affect Stage 2:
  - The collision penalty changed from the drones that caused the crash to every drone.
  - `reward.formation_weight` 0.5 to 0.75, `ppo.ent_coef` 0.0 to 0.005, default `ppo.n_steps` 512 to 256.
  - The observation gained two entries for the waypoint, which are zero in Stage 2.
- Restoring the formation weight, the entropy coefficient and the rollout gave 84%, not 100%. That leaves the collision-penalty scope as the main untested suspect.
- 8 × 256 scored lower than 8 × 128 on each of the three seeds (mean 61% against 73%). Three seeds with swings this large do not establish the difference. A possible reason: the failing runs are slow starters, and 2,048 steps per update gives half as many updates in 500k steps.

### 2026-11-14.16 Code changes (final state)

| File | Location | Change | Reason |
|---|---|---|---|
| `config.py` | `EnvConfig` | New `skip_targetless_cells: bool = True` | Cause 1 |
| `config.py` | `EnvConfig` | New `waypoint_frame: bool = True` | Cause 3 |
| `config.py` | `EnvConfig` | New `observe_search_map: bool = True` | Ablation switch (2026-11-14.11) |
| `config.py` | `EnvConfig` | `search_cohesion_radius` 2.0 to 1.0 | Cause 2 |
| `config.py` | `RewardConfig` | New `search_formation_weight: float = 0.75` | Cause 2 |
| `config.py` | `RewardConfig` | `search_waypoint_weight` 1.0 to 2.0 | Same form and scale as `progress_weight` |
| `config.py` | `PPOConfig` | New `lr_schedule: str = "constant"`; `n_steps` 256 to 512 | Cause 4 |
| `config.py` | `STAGES` | Stages 3 to 5 set `ppo.lr_schedule=linear`; `3_search` budget 1M to 2M | Cause 4 |
| `config.py` | `validate` | Rejects unknown `lr_schedule` values and a negative `search_formation_weight` | Input checking |
| `environment.py` | `__init__` | `self.searchable`: cells that intersect the target sampling square | Cause 1 |
| `environment.py` | `reset` | `self.searched` starts as `~self.searchable` | Cause 1 |
| `environment.py` | `_refresh_search_goal` | Restructured. Sets the shared search frame every step when `waypoint_frame` is on | Cause 3 |
| `environment.py` | `_observation` | Own position and map are zeros when `observe_search_map` is off | Ablation switch |
| `environment.py` | `step` | Waypoint progress per drone, measured against the waypoint held before the move | Cause 3 |
| `environment.py` | `step` | Formation term in both phases, with `search_formation_weight` before detection | Cause 2 |
| `environment.py` | `step` | `info["searched_fraction"]` over searchable cells only | Follows from cause 1 |
| `environment.py` | class docstring, comments | Describe the two frames and the cohesion limit | Documentation |
| `training.py` | `train` | Passes a `LinearSchedule` to PPO when `lr_schedule` is `linear` | Cause 4 |
| `cluster/train_cpu.slurm` | body | `N_STEPS = 2048 / N_ENVS` | Cause 4 |
| `cluster/train_cpu.slurm` | header | Note on same-name jobs that start in the same second | See 2026-11-14.17 |
| `README.md` | 4.2, 4.3, 4.6, 5, 6, 7, 10.2, 10.3, 11, 12 | Frames, searchable cells, reward table, PPO settings, budgets and status, chained seed loop, limitations, glossary | Documentation |

Behaviour impact:

- **Stages 1 and 2:** the environment is unchanged. The PPO rollout changes: 4 × 512 by default and 8 × 256 in batch jobs, instead of 4 × 256 and 8 × 128.
- **Stages 3 to 5:** the map, the frame, the search rewards, the learning-rate schedule and the Stage 3 budget all change.
- **Saved models:** the observation size is still 126, but its meaning changed. Models trained before this entry cannot be evaluated in the current environment.

### 2026-11-14.17 Verification, cost and housekeeping

- Passed: `py_compile` of the changed modules; `run_experiment.py --list-params`; rejection of `ppo.lr_schedule=cosine`; two 2,000-step smoke runs (`20261114_194843_smoke_s3fix_constant`, `20261114_194854_smoke_s3fix_linear`).
- Passed: reward bookkeeping on Stage 3. The reward terms sum to the reward, and the per-drone waypoint reward matches an independent calculation.
- Passed: in 300 scripted episodes there was no step with the map fully searched and the target not found.
- No automated test suite was added. The analysis scripts for this entry were run from a temporary folder and are not in the repository.
- Cost, from `sam-list-usagerecords`: 35 batch jobs, 9,858 credits, 67.3 core-hours.

  | Job | Duration | Credits |
  |---|---|---|
  | Stage 3, 1M steps (25 jobs) | 12 to 20 min | 254 to 439 |
  | Stage 3, 2M steps (3 jobs) | 22 to 28 min | 493 to 604 |
  | Stage 2, 500k steps (7 jobs) | 6 to 9 min | 134 to 200 |

- Balance of `lp_deeplearning_gt_2026b` on 2026-11-14 at 22:29: 602,747. This user's batch jobs that day total 10,360 credits. The rest of the drop from the 729,668 noted in 2026-11-14.7 does not come from this user's jobs.
- All jobs that were not chained started within a minute of submission, with `--time=00:40:00`. `sbatch --test-only` had estimated a 45-minute wait.
- Run folders are named by start time to the second. Same-name seeds were chained with `--dependency=after:<jobid>+1` so that two jobs could not create the same folder (README 10.3). `make_run_dir` is unchanged.
- `runs/` grew from 25 MB to 128 MB. No run was deleted.

### 2026-11-14.18 What the results establish and do not

Supported:

- Stage 3 passes the 80% gate at 2M steps: 94%, 93% and 95% on three training seeds, and 89.7% on 300 further episodes for the weakest of them.
- The border-cell fix: a scripted policy goes from 63% to 100% with it, and PPO fell to 28% without it in round 1.
- The search frame: the waypoint reward changed sign, and the same setup without the frame scored 0% (round 2).
- The frame has to track the waypoint: the boundary crashes of round 2 were flights past a missed waypoint, and they disappeared in round 3.
- The 2,048-step update with linear decay is more reliable than either alone or neither (round 4, two seeds each; then five seeds).

Not established:

- What the formation shaping during search and the waypoint weight of 2.0 contribute on the final configuration. Their ablations ran in round 1, before the frame existed.
- Why hiding the map raises the found rate and lowers conversion.
- That 1M steps is enough: four of five seeds passed, one reached 62%.
- That the settings carry over to Stages 4 and 5.
- The cause of the Stage 2 regression, and whether 2,048 steps per update makes it worse.
- Stage 1 on the current code. It has not been run since 2026-11-05.

### 2026-11-14.19 Next steps

1. **Stage 2.** Test the collision-penalty scope. `culprits` is already computed in `step`, so the change is one line. Three seeds for each scope at 500k steps is six jobs, about 1,100 credits. Until then, `--set ppo.n_steps=128` scored higher than the default on each seed.
2. **Stage 3 approach collisions (about 5%).** After detection each drone still has its own frame, so "+x" makes the drones converge. A shared frame from the centroid to the target is the natural test: the formation radius (0.5 m) is smaller than the success radius (0.8 m), so the swarm does not need to converge. It changes Stages 1 and 2 as well and needs their runs repeated.
3. **Ablations on the final configuration**, if the contribution of each change is needed: `env.skip_targetless_cells=false`; `reward.search_formation_weight=0` with `env.search_cohesion_radius=2.0`; `reward.search_waypoint_weight=0`. Three 2M-step jobs.
4. **Before Stage 4.** The waypoint ignores obstacles. A cell whose centre lies deep inside an obstacle cannot be searched, because no drone can come within 0.79 m of it, so the waypoint would stay there for the rest of the episode. Such cells cannot contain the target either (`obstacle_safe_margin`) and should start as searched.
5. **Tooling.** Make `make_run_dir` safe for same-name jobs that start in the same second. The periodic evaluation uses 10 fixed episodes and is noisy; windows of `training_episodes.csv` are a steadier measure.
