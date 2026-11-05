# Project Log

Chronological record of implementation steps, problems, solutions, and results. For architecture and usage, see [README.md](README.md).

Commit history: this workspace has no `.git` folder, so per-commit hashes are not recorded here. Branch names are as reported by the user.

## Index

- [2026-10-31: Handoff and Stage 1 experiments](#2026-10-31-handoff-and-stage-1-experiments)
- [2026-11-05: Stage 3 review, fixes, and documentation restructure](#2026-11-05-stage-3-review-fixes-and-documentation-restructure)

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
