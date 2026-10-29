# Drone Swarm 2D: PPO navigation, search and cohesion (handoff / README)

Three drones in a 10 x 10 arena must find a hidden target, fly to it together, and avoid obstacles, walls and each other.
Gymnasium environment + Stable-Baselines3 PPO. This file is both the README and the handoff of all changes made so far.

> **Status: no training has been run on this code.** Every change below was made by static editing only (no packages
> installed, nothing executed). Expect to fix a bug or two on the first smoke test (section 4).
> The one earlier run (old code) failed completely and was deleted.

---

## 1. Current setup at a glance

| Area | Current default (`config.py`) |
|---|---|
| World | 10 x 10, 3 drones of radius 0.15, `dt` 0.1, max speed 1.0, max accel 1.0, 800 steps per episode |
| Target | hidden, random each episode; detected within 1.5 of any drone, then shared by the whole swarm |
| Start | random swarm centre each episode, drones on a circle of radius 0.5, at least 5.0 from the target |
| Obstacles | 3-5 random circles per episode, radius 0.5-0.9 |
| Success | every drone strictly within `env.success_radius` = 0.8 of the target at the same time |
| Cohesion | distance to the swarm centroid beyond `env.cohesion_radius` = 1.0 is penalised (no fixed formation) |
| Observation | 152 numbers (section 3) |
| Vision | 7 rays, 120 degrees, range 2.0, along the velocity direction |
| Reward | progress 2.0, exploration 0.2, cohesion 0.5, separation 1.0, obstacle proximity 1.0, step 0.01, collision 10, success 50 (section 5) |

Everything is in `config.py` and can be overridden per run. If you want a different success radius, a nearest-neighbour
cohesion rule instead of the centroid one, or different weights, those are config values (the nearest-neighbour rule is a
small code change in `_cohesion_excess` in `environment.py`).

---

## 2. Project files

| File | Purpose |
|---|---|
| `config.py` | **Every tunable value**, grouped and commented, plus curriculum presets (`STAGES`), CLI override logic and `validate()` |
| `environment.py` | The Gymnasium environment `DroneSwarm2D` (dynamics, sensing, rewards, termination) |
| `training.py` | PPO setup (parallel envs, policy settings), a callback that records outcomes, reward terms and periodic evaluation |
| `diagnostics.py` | Baseline policies, evaluation episodes, metrics, plots, GIFs, automatic diagnosis text, `report.md` writer |
| `run_experiment.py` | **The entry point**: train, evaluate against baselines, diagnose, write a run folder |
| `compare_runs.py` | Table comparing runs in `runs/` (runs with the same name are treated as seeds, shown as mean +- std) |
| `requirements.txt` | `stable-baselines3`, `gymnasium`, `numpy`, `matplotlib`, `pillow`, `tensorboard` |

Deleted as legacy or superseded: `train.py`, `animate.py`, `animate_trained.py`, `check_env.py`, `compare_policies.py`,
`evaluate.py`, `organize_files.py`, `__pycache__/`, and the one old run folder. `run_experiment.py` replaces training,
evaluation and the GIFs (first episode and first failure).

Suggested `.gitignore` (not created): `.venv/`, `runs/`, `__pycache__/`, `*.gif`, `*.zip`.

---

## 3. How the task works

- **World:** square, walls at 0 and 10. A drone touching a wall is a crash.
- **Drones:** 3, radius 0.15 (two crash when centres are < 0.3 apart). One policy controls all three
  (centralised); the action is a (3, 2) array in [-1, 1], the world-frame acceleration of each drone.
  Speed is capped at 1.0, `dt` = 0.1, so a drone crosses the world in about 100 steps.
- **Obstacles:** circles, 3-5 new random ones each episode, never placed on the start drones or the target.
  They can overlap and occasionally form a wall, so a few layouts are nearly impossible.
- **Hidden target:** the policy is **not** told where the target is. A drone detects it only within
  `env.target_sense_range` (1.5). Once any drone has, all drones know it and receive the vector to it. Until then the
  swarm gets a coarse map of which 1 x 1 cells it has already searched, and a small bonus for each newly searched cell.
  A cell counts as searched when its centre has been within sensing range of a drone.
- **Success:** every drone strictly within `env.success_radius` of the target at the same time, with no collision on
  that step. The swarm centre and the average distance are never used.
- **Episode ends** (`terminated`) on success or any collision (obstacle, wall, drone-drone), or is **truncated**
  after `env.max_steps`. SB3's vectorised wrapper handles the time-limit truncation correctly.
- **Vision:** 7 rays over 120 degrees, range 2.0, pointing along each drone's velocity. A ray returns the distance to
  the nearest obstacle, wall or other drone (1.0 = nothing within range). Rays do not say what was hit, and nothing is
  seen to the side or behind.

### Observation (152 numbers for 3 drones, roughly in [-1, 1])
Per drone (17 values x 3 = 51):
1. position / world_size (2)
2. velocity / max_speed (2)
3. vector to target / world_size (2), **zeros until the target is found**
4. relative position of each other drone, clipped to +-3 (4)
5. the 7 ray readings (7)

Then global: `target_found` flag (1) and the searched map (100 cells, row-major, 1 = searched). Total 51 + 1 + 100 = 152.

### Memory
The policy is an MLP with **no memory**. It forgets obstacles as soon as they leave the rays. Between episodes it
learns behaviour (turn away when a ray is short), not a map, because every layout is new. Real within-episode memory
would need an obstacle map in the observation or an LSTM policy (`RecurrentPPO` in `sb3-contrib`, not installed).

---

## 4. How to run

Run everything from the project folder (`runs/` is created relative to the current directory).

```
python run_experiment.py --list-params                       # all parameters, defaults and stages
python run_experiment.py --stage 1_known_fixed --timesteps 2000 --episodes 3 --no-gif --name smoke   # smoke test
python run_experiment.py --stage 1_known_fixed --seed 1      # real run
python run_experiment.py --stage 3_search --name search_lr --set ppo.learning_rate=1e-4 --hypothesis "slower lr is steadier"
python compare_runs.py --last 5 --md                         # compare experiments
```

Precedence of settings: defaults in `config.py` < `--config file.json` < `--stage` < `--timesteps/--seed/--name/...` < `--set`.
`--set` takes `section.name=value`, values are JSON (`true`, `0.5`, `[64,64]`).

**Smoke test checklist:** it should finish without error and create `runs/<time>_smoke/` with `report.md`, `results.json`,
`config.json`, `metadata.json`, `model.zip`, CSVs and plots. Then check by hand that: the observation length is 152, success
fires only when all three drones are inside the radius, collisions end the episode, and the reward terms in
`episode_metrics_ppo.csv` add up to the episode reward.

### Slurm template (untested; adjust to your cluster)
```bash
#!/bin/bash
#SBATCH --job-name=drones
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=1-3
#SBATCH --output=slurm_%A_%a.log
# add your account/partition lines here
cd <project folder on the cluster>
source .venv/bin/activate
python run_experiment.py --stage 1_known_fixed --seed $SLURM_ARRAY_TASK_ID --set ppo.n_envs=8 ppo.batch_size=256 --no-gif
```
Training is CPU-only (`device="cpu"`); a small MLP does not benefit from a GPU. With `ppo.n_envs` above 1 each
environment runs in its own process and `ppo.n_steps` is per environment. **This multi-process path is untested**;
verify it with a short job first.

### Outputs of a run (`runs/<timestamp>_<name>/`)
`report.md` (read this first), `results.json`, `config.json` (full settings), `metadata.json` (status, time, git commit,
command), copies of `environment.py` and `config.py` as used, `model.zip`, `training_episodes.csv`, `eval_curve.csv`,
`episode_metrics_ppo.csv`, `episode_metrics_baselines.csv`, `trajectories.png`, `reward_breakdown.png`,
`training_curves.png`, `episode_first.gif`, `episode_failure.gif` (first failed episode), and `sb3_log/` (CSV and
TensorBoard events if `tensorboard` is installed: `tensorboard --logdir runs`).

Not implemented: periodic checkpoints and resume after an interrupted job (only the final model is saved), a
"frequency/duration of cohesion violations" metric (only the mean cohesion error), and an automated test suite.

---

## 5. Reward (current values, in `config.py` under `RewardConfig`)

Per step, summed (each is logged separately as a column `r_<term>` in the CSVs and in `reward_breakdown.png`):

| Term | Formula | Default weight |
|---|---|---|
| `progress` | `+ w * mean over drones (old_dist - new_dist)`, **only once the target is found** | 2.0 |
| `exploration` | `+ w * (cells searched for the first time this step)`, only until found | 0.2 |
| `cohesion` | `- w * mean over drones max(0, dist to swarm centroid - env.cohesion_radius)` | 0.5 (radius 1.0) |
| `separation` | `- w * mean over drones max(0, margin - gap to nearest drone) / margin` | 1.0 (margin 0.3) |
| `obstacle_proximity` | `- w * mean over drones max(0, margin - clearance) / margin` | 1.0 (margin 1.0) |
| `step` | `- constant` | 0.01 |
| `collision` | `- once`, episode ends | 10.0 |
| `success` | `+ once`, episode ends | 50.0 |

Rough scale: full speed toward the target gives about `2.0 * 0.1 = 0.2` per step; a trip of about 100 steps gives about
20 from progress, plus up to 50 for success. A collision costs 10 plus the future progress you lose.

**Why a "total reward" can mislead.** A policy can score a decent return from exploration and progress terms without
ever succeeding. Always read the success rate and the per-term breakdown, not the return alone.

**If you use a nearest-neighbour cohesion rule** (each drone needs a neighbour within some distance), note that the
`separation` term (centres closer than 0.6 are penalised) can fight a tight threshold such as 0.5. Lower
`reward.separation_margin` or set `reward.separation_weight=0` in that case.

---

## 6. Evaluation, baselines and the report

After training, `run_experiment.py` evaluates three policies on the **same seeded episodes** (seeds `run.seed + 1000 ...`,
different from the periodic-eval seeds used during training):

- **Random:** random actions. The floor.
- **Greedy:** a hand-written, non-learning rule (in `diagnostics.py`). Flies the swarm to the nearest unsearched map
  cell until the target is found, then to the target, keeping each drone's offset from the swarm centre. It is blind
  to obstacles and does not know the hidden target before finding it.
- **PPO:** your trained policy, run deterministically (no exploration noise).

If PPO does not beat Greedy, learning is not adding value yet. Fix the setup or reward before tuning.

Report columns: success rate with a 95% Wilson confidence interval (wide with 30 episodes), collision rate, timeout
rate, **target found rate**, mean return, mean steps, mean and worst-drone final distance to the target, outcome counts
(success / obstacle / boundary / drone_collision / timeout), a per-outcome reward-term breakdown, and an automatic
**Diagnosis** section (for example "100% of episodes end in drone_collision", "target found in only X%", "PPO does
not beat the naive searcher", "crashing pays").

Per episode (`episode_metrics_ppo.csv`) also: drones at target, closest approach, path length, minimum obstacle
clearance, minimum drone-pair distance, mean cohesion error, collision position, target found, searched fraction.

---

## 7. Changelog

### Why the first run failed (old code, 100k steps)
100% of PPO episodes ended in a drone-drone collision after about 15 steps. The return stayed near -10 (just the
collision penalty); training never improved (`train/std` fell only from 1.0 to 0.91). Causes found:
- **Nothing discouraged drones from closing in on each other.** The old formation term only penalised a nearest
  neighbour that was *too far*, never too close, and did not track the shape.
- **The policy could hardly see.** The observation was raw positions (0-10) plus velocities and 7 rays (80 degrees,
  range 1.0) aimed along the velocity. Teammates and obstacles were almost never visible; relative positions had to be
  inferred from raw absolute coordinates.
- **The obstacle penalty started at 2.0 m** but vision reached only 1.0 m, so the penalty looked like random noise.
- **Action noise was too large:** SB3's default initial std is 1.0 on a [-1, 1] range.
- **Success was nearly impossible by design:** all drones had to be within 0.5 of one point while the formation was
  about 1.0 wide.

### Changes made
- `environment.py`
  - Observation rebuilt (normalised, relative drone positions, target vector, searched map). See section 3.
  - Success = every drone inside `success_radius` of the target (per-drone, not the centre).
  - Fixed formation removed; replaced by centroid cohesion.
  - New reward terms `separation` and `exploration`; `REWARD_TERMS` order is progress, exploration, cohesion,
    separation, step, obstacle_proximity, collision, success.
  - Random start and target each episode with a minimum start-target distance; hidden target with sensing and
    searched map; random obstacles avoid the start drones and the target.
  - Removed old module-level constants (`WORLD_SIZE`, `TARGET`, ...) and `save_plot`; `draw_scene` takes the
    episode's target.
  - Smaller fixes: heading uses the post-clipping speed; `ray_Offsets` renamed `ray_offsets`; duplicate import removed.
- `config.py`
  - Rewritten: every parameter grouped and commented; `STAGES` curriculum presets; `--stage` support; one complete
    `validate()` (there were two copies, the second silently overriding the first).
  - New parameters: `num_drones`, `start_spread`, `cohesion_radius`, `random_start`, `random_target`, `edge_margin`,
    `min_start_target_distance`, `target_sense_range`, `explore_cell`, `reward.cohesion_weight`,
    `reward.separation_weight/margin`, `reward.explore_weight`, `ppo.log_std_init` (-1.0), `ppo.n_envs`.
  - Removed: `formation`, `cohesion_distance`.
  - Changed defaults: `vision_range` 1.0 -> 2.0, `vision_angle` 80 -> 120, `obstacle_margin` 2.0 -> 1.0,
    `success_radius` 0.5 -> 0.8, `max_steps` 500 -> 800.
- `training.py`: passes `log_std_init`; builds a vectorised environment with `ppo.n_envs`.
- `diagnostics.py`: baselines updated for the hidden target; per-episode target stored and drawn; new metrics
  (`found_rate`, `mean_searched_fraction`, `mean_cohesion_error`); updated advice text; "Target found" column.
- `run_experiment.py`: `--stage`, updated examples.
- Legacy scripts and the old run deleted (section 2).
- A mistake of mine, fixed: one edit had turned the `num_rays` field in `config.py` into part of a comment. The editor's
  error check did not catch it; it was found on re-reading and `config.py` was recreated.

---

## 8. Recommended training plan (staged)

Training everything at once gives the policy almost no success signal, and when it fails you cannot tell why.
Move to the next stage only at about 80% success on the final evaluation across at least 3 seeds.

| Stage (`--stage`) | What it tests | Key settings |
|---|---|---|
| `1_known_fixed` | setup works at all: fixed layout, target known, no obstacles | `target_sense_range=20`, 0 obstacles, fixed start/target, 300k steps |
| `2_known_random` | random start and target, still known | 500k steps |
| `3_search` | hidden target, search | default sensing, 1M steps |
| `4_search_few_obstacles` | search plus 1-2 obstacles | 1M steps |
| `5_full` | everything random, 3-5 obstacles | 3M steps |

The `run.timesteps` numbers are guesses; read `eval_curve.csv` and `training_curves.png`. If success is still rising at the
end, train longer.

**Tuning order:** `run.timesteps`, then `reward.collision_penalty` and `reward.separation_*`, then
`reward.cohesion_weight` against `reward.progress_weight`, then `ppo.learning_rate` and `ppo.ent_coef`. Change one thing
per run, use 3 seeds, and note the reason in `--hypothesis`.

---

## 9. Known limitations and risks

- **Untested code** (see the status note at the top).
- **Hard task as a whole:** search, avoidance, cohesion, random everything and a memoryless policy. Expect success to
  stay below 100% even when it works (some random layouts are blocked).
- **Speed:** the raycast is a Python loop (about 650 steps/s with one environment before the changes). It is the
  bottleneck; vectorise it before scaling up on the cluster.
- **Observation size:** 152 inputs with a [64, 64] network may be small; try `ppo.net_arch=[128,128]`.
- **Evaluation CI:** 30 episodes give wide intervals; use more `--episodes` for final numbers.
- **`inf` values:** with zero obstacles, `min_obstacle_clearance` is `inf` in the logs (harmless).
- **Searching needs time:** `max_steps` is 800; the swarm may need more if cohesion keeps it tightly bunched.

---

## 10. Beginner glossary

- **Episode:** one attempt from reset until success, crash or time limit.
- **Return:** total reward over an episode. Misleading on its own; read the per-term breakdown.
- **Policy:** the neural network mapping the observation to actions.
- **PPO:** the learning algorithm. Collects `n_steps` of experience per environment, then updates the policy a few
  times (`n_epochs`) with small clipped steps (`clip_range`) so it does not change too abruptly.
- **Rollout / `n_steps`:** how much experience is collected between updates.
- **Learning rate:** step size of the updates. Too high is unstable, too low is slow.
- **Discount `gamma` (0.99):** how much future reward matters; about 100 steps (10 s) of lookahead.
- **Entropy (`ent_coef`):** a bonus that keeps the policy exploring. Raise it if the policy stops exploring too early.
- **`log_std_init`:** initial random noise on the actions (std = exp(value)). -1.0 means std about 0.37.
- **Deterministic evaluation:** at test time the policy uses its mean action with no noise.
- **Terminated vs truncated:** terminated = the task ended (success or crash); truncated = the time limit hit.
- **Seed:** the random-number start. Always compare several seeds; one run can look good or bad by luck.
- **Curriculum:** training on easier versions first (the stages above).
