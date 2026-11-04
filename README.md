Drone Swarm 2D — PPO navigation, search and formation coordination
Project handoff / working README
Updated: 2026-10-31
Working branch reported by the user: experimental/wall-proximity
Target integration branch: advanced

This document consolidates the original project README, the earlier handoff, the implementation changes described during development, and the experiment results reported through 2026-10-31. It is intended to let another student or AI assistant continue the work without relying on the chat history.
Important status distinction: the environment, experiment runner and several reward/control changes have been executed and tested. The best reported Stage 1 evaluation is 60% PPO success over 30 episodes. This is not yet a validated final result: it is one training seed, with a wide confidence interval. The latest proposed bottleneck-aware progress reward has not been implemented or tested as of this handoff.

The most recent poor experiments are retained in the results history. Do not treat every experimental parameter as a recommended default. The current recommended Stage 1 comparison configuration is listed in section 8.

1. Project goal and research question
Three allied drones operate in a 10 × 10 two-dimensional arena. The swarm must find a target, travel to it together, and avoid obstacles, walls and collisions between drones.

The project uses:
- Python and Gymnasium for the custom simulation environment.
- Stable-Baselines3 (SB3) PPO for reinforcement learning.
- A centralized policy controlling all three drones.
- A curriculum that gradually introduces randomization, hidden-target search and obstacles.
- Reproducible experiment folders with evaluation against Random and Greedy baselines.
Research question: How does reward design and coordinated control affect the balance between mission completion, formation maintenance and hazard avoidance in a small drone swarm?
Main evaluation metrics
- Mission success rate, including a 95% Wilson confidence interval.
- Collision rate, separated into obstacle, boundary and drone-drone collisions.
- Timeout rate.
- Target-found rate.
- Mean and worst-drone final distance to the target.
- Mean formation error.
- Minimum pairwise drone distance.
- Episode length and path length.
- Per-term reward breakdown.
- Generalization across training seeds and, later, randomized target/obstacle scenarios.
2. Current project architecture
The current workflow is:
config.py
  ├── environment settings
  ├── reward settings
  ├── PPO settings
  └── curriculum stages
        |
        v
 environment.py  <--- Gymnasium DroneSwarm2D
        |
        v
 training.py     <--- SB3 PPO, callbacks, training metrics and periodic evaluation
        |
        v
 diagnostics.py <--- Random/Greedy/PPO evaluation, episode metrics, plots, GIFs, report
        |
        v
 run_experiment.py <--- main experiment entry point; creates a timestamped run directory
        |
        v
 compare_runs.py   <--- compares runs / training seeds
Source files
File	Responsibility
config.py	Tunable environment, reward, PPO and run parameters; curriculum presets; CLI overrides; configuration validation.
environment.py	DroneSwarm2D: dynamics, observation, raycasting, target search, reward calculation, collision detection, success and termination.
training.py	PPO construction, vectorized environment setup, training callback, reward-term accumulation and periodic evaluation.
diagnostics.py	Random and Greedy policies, deterministic PPO evaluation, per-episode metrics, summaries, plots, GIFs, diagnosis text and report.md.
run_experiment.py	Main entry point: load configuration, apply stage/CLI overrides, train, evaluate, save model and write artifacts.
compare_runs.py	Summarizes and compares saved runs. Runs sharing the same experiment name can be treated as different seeds.
requirements.txt	Python dependencies.
README.md	Project documentation and experimental handoff.


Legacy scripts from the earlier implementation were removed/superseded: train.py, animate.py, animate_trained.py, check_env.py, compare_policies.py, evaluate.py and organize_files.py. run_experiment.py is the current entry point.

Suggested .gitignore entries:
.venv/
runs/
__pycache__/
*.gif
*.zip
Keep source code and the README in Git. Do not normally commit the virtual environment or generated experiment folders; preserve run folders on the cluster for analysis and back them up separately when necessary.
3. Environment and task mechanics
3.1 World and drones
Property	Value / behaviour
World	Square arena, 10 × 10 m; walls at 0 and 10.
Number of drones	3.
Drone radius	0.15 m.
Drone-drone collision	Centre-to-centre distance strictly below 0.30 m.
Time step	0.1 s.
Maximum speed	1.0 m/s.
Maximum acceleration	1.0 m/s².
Episode step limit	800.
Initial layout	Three drones evenly distributed on a circle around the swarm centre; default start_spread = 0.5 m.


The action is a (3, 2) array in [-1, 1]. It represents world-frame acceleration commands, two components per drone. One PPO policy outputs all six values.

The simulation updates velocity from acceleration, clips velocity to the maximum speed, then updates position. This is a simple kinematic model; it is not a high-fidelity aerodynamic or motor model.
3.2 Target and search
- In the full task, the target is hidden and randomized each episode.
- A drone detects the target when it is within target_sense_range (default 1.5 m).
- Once any drone detects it, the target becomes known to the whole swarm.
- Before detection, the observation includes a 10 × 10 searched-cell map (100 binary values, row-major).
- A cell is marked searched when its centre comes within the sensing range of at least one drone.
- The exploration reward is paid for newly searched cells, and only until the target is found.
For Stage 1, the target is effectively known from the beginning by setting target_sense_range = 20.0, larger than the world.
3.3 Obstacles and walls
- The full task uses 3–5 random circular obstacles, with radii 0.5–0.9 m.
- Random obstacles are placed away from the start drones and the target, but can overlap one another and may form difficult or occasionally near-blocked layouts.
- A drone touching an obstacle or wall causes an episode-ending collision.
- Stage 1 disables obstacles, but wall collisions remain active.
3.4 Success and termination
Success requires every drone to be strictly within env.success_radius = 0.8 m of the target simultaneously, and no collision on that step. The mean swarm distance or swarm centroid is not sufficient.

An episode is:
- terminated on success or any collision (obstacle, wall/boundary or drone-drone).
- truncated at the max_steps time limit.
The evaluation code distinguishes success, obstacle, boundary, drone collision and timeout.
3.5 Vision and observation
The observation has 152 values:
- Per drone: 17 values × 3 = 51.
  - Position normalized by world size (2).
  - Velocity normalized by maximum speed (2).
  - Target vector normalized by world size (2); zeros before target discovery.
  - Relative positions of the other two drones (4).
  - Seven ray readings (7).
- Global target-found flag (1).
- Searched map (100).
The seven vision rays span 120 degrees, extend 2.0 m and point along the drone's velocity/heading direction. A ray reports the nearest obstacle, wall or other drone. The rays do not identify the object type and do not see fully behind the drone or to the sides beyond the ray span.

The default policy is an MLP without recurrent memory. It does not retain a persistent obstacle map beyond information explicitly included in the observation. An LSTM/RecurrentPPO policy has not been implemented.
4. PPO and software environment
4.1 PPO configuration
The configuration described in the original project README is:
Parameter	Value
Learning rate	3e-4
Number of environments	1
Rollout steps per environment	1024
Batch size	64
Epochs	10
Discount factor (gamma)	0.99
GAE lambda	0.95
Clip range	0.2
Entropy coefficient	0.0
Initial action log standard deviation	-1.0
MLP architecture	[64, 64]
Device	CPU


The 300,000-timestep Stage 1 runs with one environment and n_steps=1024 represent roughly 293 PPO rollout/update cycles (the exact final rollout accounting depends on SB3's timestep handling). The 2,000-timestep smoke test only exercises about two updates and is not a meaningful learning evaluation.
4.2 Cluster and virtual environment
The reported compute environment is the university cluster:
- Rocky Linux 9.8.
- Slurm available.
- Python module previously used: Python/3.13.5-GCCcore-14.3.0.
- Project virtual environment: .venv/.
- Project path previously reported as /data/leuven/394/vsc39468/drone-swarm (also mounted under /vsc-hard-mounts/leuven-data/394/vsc39468/drone-swarm).
Always activate the project virtual environment in every new terminal/session:
cd /data/leuven/394/vsc39468/drone-swarm
source .venv/bin/activate

echo "$VIRTUAL_ENV"
which python
python --version
python -m pip --version
The confirmed active environment reported:
- Python 3.13.5.
- Python path inside drone-swarm/.venv/bin/python.
- pip 26.2.2.
- PyTorch 2.14.1.
- Stable-Baselines3 2.9.2.
- Gymnasium 1.4.0.
- TensorBoard 2.21.0.
Matplotlib was later found missing from the active environment and had to be installed there. Confirm imports rather than assuming an earlier package installation succeeded.

Useful checks:
python -m pip show torch stable-baselines3 gymnasium tensorboard matplotlib pillow
python -m pip check
python - <<'PY'
import torch, numpy, gymnasium, stable_baselines3, matplotlib, PIL, tensorboard
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Gymnasium:", gymnasium.__version__)
print("SB3:", stable_baselines3.__version__)
print("All imports successful")
PY
4.3 Disk quota / package installation incident
An initial pip install -r requirements.txt was run while the project .venv was not activated. The shell used /usr/bin/python (Python 3.9), and packages were installed under the user's home directory. The install then failed with OSError: [Errno 122] Disk quota exceeded after downloading substantial NVIDIA/CUDA-related packages.

At the time of inspection:
- ~/.local/lib/python3.9 occupied about 3.0 GB.
- ~/.local/lib/python3.13 occupied about 56 MB.
- The project .venv occupied about 5.6 GB.
- ~/.cache/pip was purged (101 cached files removed).
The user then activated the correct .venv; the main packages were already installed there. The active PyTorch package also declared CUDA/NVIDIA dependencies even though project training is CPU-based. No broad cleanup of ~/.local was performed, because those packages could belong to other work.

Do not repeat a full requirements installation without first confirming which python and VIRTUAL_ENV. Avoid unnecessary package reinstalls; the cluster quota is constrained.
5. Current configuration and reward design
The working code evolved beyond the original defaults. The following are the recommended Stage 1 settings based on the best reported experiment, not a claim that every value is the final optimal setting.
5.1 Recommended comparison configuration
Parameter	Recommended Stage 1 value	Notes
env.random_start	False	Stage 1.
env.random_target	False	Stage 1.
env.start_jitter	0.0	Fixed swarm-centre location for comparison.
env.target_sense_range	20.0	Target known from the beginning.
env.min_obstacles	0	Stage 1.
env.max_obstacles	0	Stage 1.
env.action_residual_scale	1.0	Original independent actions; coupling at 0.25 performed poorly.
reward.progress_weight	2.0	Blend of mean and worst-drone progress (0.5/0.5). Earlier results used mean-only.
reward.cohesion_weight	0.5	Centroid cohesion.
reward.formation_weight	0.5	Pairwise triangular-formation reward.
reward.separation_weight	1.0	Keep at original value.
reward.separation_margin	0.3	Keep at original value.
reward.wall_weight	3.0	Wall-proximity shaping; calculation must remain present and active.
reward.wall_margin	1.0	Wall shaping range.
reward.obstacle_weight	1.0	Obstacle proximity.
reward.obstacle_margin	1.0	Obstacle shaping range.
reward.step_penalty	0.01	Per-step cost.
reward.collision_penalty	10.0	One-time collision penalty.
reward.success_bonus	50.0	One-time success bonus.


The last verified config.json for stage1_formation_seed42 contained wall_weight=3.0, wall_margin=1.0, formation_weight=0.5, separation_weight=1.0, and separation_margin=0.3. That first formation run did not actually apply the wall reward because its calculation had accidentally gone missing from environment.py; the later stage1_formation_wallfix_seed42 run restored it and is the best result.

The report's “Changes vs defaults” table only reports values different from the current config.py defaults. If the default itself has been changed to wall_weight=3.0 or formation_weight=0.5, those values will not necessarily appear in that table. Always inspect the run's config.json when the exact settings matter.
5.2 Reward components
Every term is accumulated per step and logged separately as r_<term>.
Term	Meaning / calculation	Current useful setting
progress	Reward for reducing target distance, only after the target is found. Original implementation uses mean progress across drones.	Weight 2.0
exploration	Reward for newly searched cells, only before target discovery.	Weight 0.2
cohesion	Penalizes drones whose distance from the swarm centroid exceeds env.cohesion_radius (default 1.0 m).	Weight 0.5
formation	Penalizes mean absolute deviation of all pairwise distances from the initial circular/triangular formation distances.	Weight 0.5
separation	Penalizes a drone when its nearest-neighbour surface-to-surface gap is below separation_margin.	Weight 1.0; margin 0.3 m
obstacle_proximity	Penalizes proximity to circular obstacles.	Weight 1.0; margin 1.0 m
wall_proximity	Penalizes proximity to the nearest arena wall.	Weight 3.0; margin 1.0 m
step	Small fixed cost every step.	0.01
collision	One-time penalty; episode ends.	10.0
success	One-time bonus; episode ends.	50.0


Cohesion versus formation
These are different objectives:
- Cohesion keeps drones within a broad radius of the swarm centroid. It does not preserve a particular shape and does not prevent the formation from collapsing into a cluster.
- Formation preserves the pairwise geometry of the initial triangular arrangement while allowing the triangle to translate and rotate.
- Separation discourages dangerously close approaches. It is not a substitute for formation maintenance.
For three drones initially spaced equally on a circle of radius start_spread=0.5 m, the desired pairwise distance is:
d_desired = sqrt(3) * start_spread = approximately 0.866 m
The formation error implementation calculates the mean absolute difference between the actual distance of each unique drone pair and its desired distance. The formation reward is:
r_formation = - formation_weight * formation_error
The formation_error is also recorded in episode metrics and summaries. The reported Greedy baseline has approximately zero formation error because it maintains the initial offsets.
Separation margin: surface gap versus centre distance
The separation calculation uses:
surface_gap = nearest_centre_distance - 2 * drone_radius
With drone radius 0.15 m:
- separation_margin=0.3 begins penalizing when centre distance is below 0.60 m.
- separation_margin=0.5 begins penalizing when centre distance is below 0.80 m.
- Physical collision occurs below 0.30 m centre distance.
Increasing the margin to 0.5 m caused a severe regression in the tested run; do not use it as the current Stage 1 setting.
Wall proximity
wall_clear is the per-drone clearance from the nearest wall, accounting for drone radius. The intended reward calculation is:
if w.wall_margin > 0:
    wall_proximity = np.maximum(
        0.0, w.wall_margin - wall_clear
    ) / w.wall_margin
    terms["wall_proximity"] = (
        -w.wall_weight * float(np.mean(wall_proximity))
    )
This term must actually be assigned in environment.py; adding "wall_proximity" to REWARD_TERMS or setting wall_weight in config.py is not sufficient. A missing calculation was found during development and restored. Confirm the term is non-zero in episode_metrics_ppo.csv on boundary-approach episodes.
5.3 Experimental action coupling
An optional control transformation was added to test whether a shared swarm command plus smaller individual corrections would improve coordination.

For raw drone actions a_i, the shared action is their mean, and the transformed action is:
shared_action = mean(a_i)
a_i_new = shared_action + alpha * (a_i - shared_action)
env.action_residual_scale (alpha) behaves as follows:
- 1.0: original independent action output; no coupling.
- 0.25: strong coupling; individual deviations are reduced to 25% before clipping.
- 0.0: every drone receives the same acceleration command.
The 0.25 experiment performed poorly (13% success, 20 boundary collisions and 6 drone collisions). Keep action_residual_scale=1.0 for the current recommended baseline. The option may remain in the code for future controlled testing, but there is no evidence that 0.25 is beneficial.
6. Running experiments
Run commands from the project root with .venv activated.
source .venv/bin/activate
python run_experiment.py --list-params
6.1 Smoke test
Use only to confirm that the complete pipeline runs. It does not establish learning performance.
python run_experiment.py \
  --stage 1_known_fixed \
  --timesteps 2000 \
  --episodes 3 \
  --no-gif \
  --name smoke
6.2 Stage 1 best-known configuration
The Stage 1 preset supplies 300,000 training timesteps, fixed target, target known from the beginning and no obstacles. For an exact repeat of the best-known reward/control configuration:
python run_experiment.py \
  --stage 1_known_fixed \
  --name stage1_best \
  --seed 42 \
  --episodes 30 \
  --no-gif \
  --hypothesis "Repeat the best-known Stage 1 configuration with the wall and formation rewards active." \
  --set env.start_jitter=0.0 \
       env.action_residual_scale=1.0 \
       reward.wall_weight=3.0 \
       reward.wall_margin=1.0 \
       reward.formation_weight=0.5 \
       reward.separation_weight=1.0 \
       reward.separation_margin=0.3
The experiment runner creates a unique timestamped directory under runs/, so reusing the same --name does not overwrite previous runs.
6.3 Configuration precedence
The README/code design defines this precedence:
defaults in config.py
  < --config file.json
  < --stage
  < explicit CLI run arguments
  < --set section.parameter=value
--set values are JSON-like values. Use true/`false` for booleans and numeric values without quotes.
6.4 Slurm
A Slurm template exists in the original README but is untested for this exact cluster allocation. The default is now ppo.n_envs=2 with ppo.n_steps=512 (still 1024 samples per update), which uses separate processes and has not been validated yet: smoke-test it with a short job and request at least n_envs CPUs (--cpus-per-task) before expensive runs. Earlier results used n_envs=1, n_steps=1024.
7. Evaluation and experiment artifacts
run_experiment.py evaluates three policies on the same final evaluation seeds:
- Random: random actions; a floor/reference, not a meaningful navigation controller.
- Greedy: hand-written non-learning policy. It moves towards unsearched cells until target detection, then towards the target while preserving each drone's offset from the swarm centre. It is obstacle-blind.
- PPO: trained policy evaluated deterministically (without action sampling noise).
For the reported seed-42 experiments, the final evaluation episode seeds were 1042–1071. The same final evaluation seeds were used across the reported configurations, which makes the per-seed outcomes comparable, but does not remove training-seed variability.
Run directory
A typical run folder contains:
runs/<timestamp>_<name>/
├── report.md
├── results.json
├── config.json
├── metadata.json
├── model.zip
├── environment.py          # source snapshot, if written by the runner
├── config.py               # source snapshot, if written by the runner
├── training_episodes.csv
├── eval_curve.csv
├── episode_metrics_ppo.csv
├── episode_metrics_baselines.csv
├── trajectories.png
├── reward_breakdown.png
├── training_curves.png
├── episode_first.gif       # unless --no-gif
├── episode_failure.gif    # unless --no-gif / generated when supported
└── sb3_log/
Exact artifacts vary depending on flags and the point at which a run finishes.
How to read the files
- report.md: first summary to read; includes final evaluation, outcomes, reward terms and automated diagnosis.
- results.json: machine-readable policy summaries, confidence intervals, outcome counts and reward breakdowns.
- config.json: authoritative record of actual parameters used in that run. Prefer this over “Changes vs defaults” when the defaults themselves have changed.
- training_curves.png: training episode outcome/reward trends and periodic evaluation.
- eval_curve.csv: periodic evaluation measurements during training. Each periodic evaluation used 10 episodes in the reported configuration.
- trajectories.png: trajectory examples for a subset of final evaluation episodes; the plot showed the first 12, not all 30, in the reviewed version.
- episode_metrics_ppo.csv: one row per final PPO evaluation episode. Useful columns include outcome, steps, final_mean_dist, final_worst_dist, drones_at_target, min_pair_distance, mean_formation_error, collision coordinates and each r_* term.
- episode_metrics_baselines.csv: matching per-episode results for Random and Greedy.
- reward_breakdown.png: summed reward terms, useful for finding incentives that dominate the return.
Do not judge a policy by mean return alone. A policy can get positive progress reward while failing the all-drones success condition. Success rate and outcome counts are primary; return and per-term breakdown are diagnostic.
8. Experiment history (2026-10-31)
All full Stage 1 training runs below used seed 42, 300,000 timesteps and 30 final evaluation episodes unless otherwise stated. Reported final confidence intervals are 95% Wilson intervals. The values are the results reported from the experiment folders in the chat.
8.1 Results overview
Experiment	Main configuration/change	PPO success	Boundary	Drone collision	Mean return	Mean steps	Mean final distance	Worst final distance	Formation error
smoke	2,000 steps; no obstacles; known target	0% (3 episodes)	1	2	-33.1	94	8.06	8.87	Not recorded
stage1_seed42	Original Stage 1 reward	30% [17–48]	11	10	14.8	88	3.17	3.60	Not recorded
stage1_wallfix_seed42	Wall weight 1.0	0% [0–11]	25	5	-2.5	88	3.74	4.16	Not recorded
stage1_wall_weight3_seed42	Wall weight 3.0; no formation reward	27% [14–44]	17	5	7.3	99	2.42	2.68	Not recorded
stage1_formation_seed42	Formation weight 0.5; wall calculation accidentally missing	47% [30–64]	8	8	21.2	85	3.03	3.30	0.189
stage1_formation_wallfix_seed42	Formation 0.5 + wall reward active at weight 3.0	60% [42–75]	5	7	23.0	87	2.48	2.75	0.209
stage1_separation_margin05_seed42	Separation margin 0.5	17% [7–34]	1	24	-2.6	36	6.91	7.18	0.107
stage1_sepweight2_seed42	Separation weight 2.0; margin 0.3	0% [0–11]	0	30	-13.2	18	8.86	9.16	0.111
stage1_action_coupling_seed42	Action residual scale 0.25	13% [5–30]	20	6	-11.2	106	2.30	2.92	0.227


Random achieved 0% success in these Stage 1 evaluations. Greedy achieved 100% in all reported 30-episode Stage 1 evaluations (95% CI [89–100]). Greedy is a useful reachability/reference policy; it is not a learned baseline.
8.2 Experiment-by-experiment notes
A. Smoke test — smoke
- 2,000 timesteps, seed 42, 3 final evaluation episodes.
- Runner completed and created the expected artifacts. This validated the software pipeline, not PPO learning.
- PPO: 0% success; one boundary collision and two drone collisions.
- Greedy: 100% success (3/3); Random: 0%.
- PPO mean final distance: 8.06 m; mean length: 94 steps.
B. Original Stage 1 baseline — stage1_seed42
- 300,000 timesteps; known fixed target; no obstacles; no start jitter.
- PPO: 9/30 successes (30%, 95% CI [17–48]); 11 boundary collisions; 10 drone collisions; no timeouts.
- Mean return 14.8 ± 34.0; mean length 88; mean final distance 3.17 m; worst-drone final distance 3.60 m.
- Training diagnostics indicated success was rising during the run. The policy had learned some target progress but failed frequently.
C. Wall proximity, weight 1 — stage1_wallfix_seed42
- Added a wall-proximity reward with wall_weight=1.0, wall_margin=1.0.
- PPO: 0/30 successes; 25 boundary collisions; 5 drone collisions.
- Mean return -2.5 ± 5.1; final distance 3.74 m; worst-drone distance 4.16 m.
- This setting was worse than the original baseline. It was not retained as the recommended weight.
D. Wall proximity, weight 3 — stage1_wall_weight3_seed42
- Increased wall_weight to 3.0; other reward components remained at their then-current values.
- PPO: 8/30 successes (27%, CI [14–44]); 17 boundary collisions; 5 drone collisions.
- Mean return 7.3 ± 32.3; mean final distance 2.42 m; worst-drone distance 2.68 m.
- Better than wall weight 1, but not clearly better than the original baseline in overall success.
- Retained as a candidate because the wall term had a meaningful penalty when the drones approached the boundary.
E. Formation reward added, wall reward accidentally inactive — stage1_formation_seed42
- Added pairwise formation error and formation_weight=0.5.
- PPO: 14/30 successes (47%, CI [30–64]); 8 boundary collisions; 8 drone collisions; mean formation error 0.189.
- The CSV showed r_wall_proximity=0.0 for every episode. The configuration contained wall_weight=3.0, but a subsequent source inspection showed that the reward calculation was missing from environment.py (the term name existed in REWARD_TERMS, but the assignment was absent).
- This run is not a valid test of formation reward plus wall reward. It is evidence about the formation change with wall shaping inactive, not a like-for-like comparison with the wall-weight-3 run.
F. Formation plus restored wall reward — stage1_formation_wallfix_seed42
- Restored the missing wall-proximity reward calculation; used wall_weight=3.0, wall_margin=1.0, formation_weight=0.5, separation_weight=1.0, separation_margin=0.3.
- PPO: 18/30 successes (60%, CI [42–75]); 5 boundary collisions; 7 drone collisions; no timeouts.
- Mean return 23.0 ± 32.0; mean length 87; mean final distance 2.48 m; worst-drone final distance 2.75 m; mean formation error 0.209.
- Per-outcome wall reward was non-zero: successful episodes averaged approximately -1.22; boundary episodes averaged approximately -9.89.
- This is the best reported result and the recommended Stage 1 configuration to repeat and validate. It is still only one training seed.
G. Separation margin increased to 0.5 — stage1_separation_margin05_seed42
- Changed only the separation margin from 0.3 to 0.5 (surface gap), with the intent of penalizing close approaches earlier.
- PPO: 5/30 successes (17%, CI [7–34]); 1 boundary collision; 24 drone collisions; 0 timeouts.
- Mean return -2.6 ± 25.3; mean length 36; mean final distance 6.91 m; worst-drone distance 7.18 m.
- The formation error metric improved numerically to 0.107, but mission performance collapsed. This illustrates why a single metric cannot be optimized in isolation.
- Conclusion: do not retain separation_margin=0.5 as the baseline.
H. Separation weight increased to 2.0 — stage1_sepweight2_seed42
- Returned to margin 0.3 and increased separation weight from 1.0 to 2.0.
- PPO: 0/30 successes; all 30 episodes ended in drone-drone collision.
- Mean return -13.2 ± 1.2; mean length 18; mean final distance 8.86 m; worst-drone distance 9.16 m.
- Most episodes terminated after 8–14 steps, often near the starting area. The reward terms still penalized separation violations, but that did not prevent collision.
- Conclusion: reject this separation-weight increase. Do not continue tuning separation margin/weight blindly.
I. Action coupling, residual scale 0.25 — stage1_action_coupling_seed42
- Added a configurable action transformation: shared mean acceleration plus a scaled individual residual.
- Tested env.action_residual_scale=0.25; this substantially reduced the independent component of each drone's action.
- PPO: 4/30 successes (13%, CI [5–30]); 20 boundary collisions; 6 drone collisions.
- Mean return -11.2 ± 26.3; mean length 106; mean final distance 2.30 m; worst-drone distance 2.92 m; formation error 0.227.
- Drone collisions were not the only or dominant failure; boundary collisions increased sharply.
- Conclusion: reject 0.25 as a recommended control setting. Keep action_residual_scale=1.0 for the baseline. The optional transformation can remain available for later testing, but the current evidence does not support using strong coupling.
8.3 What the experiments do and do not establish
Supported by the reported runs:
- The full experiment pipeline executes and produces reports, models, CSVs and plots.
- In Stage 1, Greedy can reach the target with all three drones in the success radius in all 30 final evaluation episodes.
- PPO learned some navigation behaviour, with its best reported single-seed Stage 1 result at 60% success.
- Adding pairwise formation reward and correctly active wall proximity coincided with the best result.
- Increasing separation margin to 0.5, increasing separation weight to 2.0, and applying action coupling at 0.25 all produced poor results in the tested seed-42 runs.
Not established:
- That the 60% result is reproducible across training seeds.
- That formation/wall terms alone caused the increase; all results are single training seeds and one formation run had a wall reward bug.
- That PPO is better than Greedy. Greedy scored 100% in the tested fixed task.
- That the current policy generalizes to random starts/targets, hidden-target search or obstacles.
- That the best reported configuration reaches the proposed ~80% curriculum advancement threshold across at least three seeds.
- That the proposed bottleneck-aware progress reward improves performance; it has not yet been implemented or tested.
9. Mistakes, debugging findings and lessons
This section is intentionally explicit so the same problems are not repeated.
9.1 Python environment was not activated during package installation
The first full pip install -r requirements.txt was run with system Python 3.9 rather than the project's Python 3.13 virtual environment. Packages went into ~/.local, consumed quota and installation failed. Always check which python and $VIRTUAL_ENV first.
9.2 Wall reward was present in the term list/config but its calculation was absent
During the formation edits, wall_proximity remained in REWARD_TERMS and wall_weight=3.0 remained in the configuration, but the terms["wall_proximity"] = ... calculation had gone missing. Therefore, r_wall_proximity was zero for every episode in stage1_formation_seed42.

The mistake was found by inspecting the CSV and running grep on environment.py. The calculation was restored before stage1_formation_wallfix_seed42, whose CSV showed non-zero wall penalties.

Lesson: a config value and a reward-term label do not prove that the reward is active. Verify actual per-episode r_* values and compare their sum with episode return.
9.3 The report's “Changes vs defaults” can be misleading after defaults are changed
The experiment report compares the run config with the current config.py defaults. If wall_weight=3.0 and formation_weight=0.5 have been made defaults, they will not appear as overrides even though they differ from the original upstream README. config.json is the authoritative record of values used.
9.4 The automatic diagnosis can be generic or misleading
Some reports suggested obstacle avoidance even though Stage 1 explicitly had zero obstacles. In an obstacle-free Stage 1 run, a boundary collision is not an obstacle collision. Use the outcome code, collision position and exact run config rather than blindly following generic diagnosis text.
9.5 Separation-margin and separation-weight tuning was counterproductive
The margin 0.5 run produced 80% drone collisions. The weight 2.0 run produced 100% drone collisions. The numbers show that increasing repulsion is not automatically safer; it can interact badly with formation and progress rewards and can make PPO learning worse.

Lesson: do not keep increasing separation penalties as a default response to drone collisions. Diagnose action/observation dynamics and reward alignment first.
9.6 Strong action coupling was counterproductive
The residual-scale 0.25 run achieved only 13% success and had 20 boundary collisions. It restricted independent control but did not solve the coordination problem.

Lesson: do not assume a more constrained action space is better. Keep the normal action output (action_residual_scale=1.0) unless a multi-seed experiment demonstrates otherwise.
9.7 Do not interpret one metric in isolation
The margin 0.5 run had a lower mean formation error (0.107), but only 17% success and 24 drone collisions. Formation error alone is not mission performance. Always examine success, outcomes, minimum pair distance, final worst-drone distance and reward components together.
9.8 A single 30-episode, single-training-seed evaluation is not robust evidence
All these experiments used training seed 42. Although the final evaluation seeds were consistent, the policy weights themselves depend on the training seed. A 30-episode confidence interval is wide. A 60% point estimate with CI [42–75] should be described as preliminary, not a stable success rate.
9.9 A branch can be blocked by a Git worktree
An attempt to delete local branch testing with git branch -d testing failed because the branch was checked out in a worktree at /vsc-hard-mounts/leuven-data/394/vsc39468/drone-swarm. Git will not delete a branch that is checked out in any worktree. Check git worktree list; switch away from the branch in the relevant worktree before deleting it. Do not force-delete a branch merely to bypass this protection.
10. Current recommended experimental strategy
The recent sequence of ad hoc weight changes was not productive. The next phase should be more controlled and should not add reward terms or change several parameters at once.
10.1 Freeze the best-known Stage 1 configuration
Use the configuration from stage1_formation_wallfix_seed42:
env.action_residual_scale = 1.0
reward.wall_weight = 3.0
reward.wall_margin = 1.0
reward.formation_weight = 0.5
reward.separation_weight = 1.0
reward.separation_margin = 0.3
reward.progress_weight = 2.0
reward.cohesion_weight = 0.5
Do not use separation_margin=0.5, separation_weight=2.0 or action_residual_scale=0.25 as the baseline.
10.2 Repeat the best configuration across training seeds
Before making another change, establish whether the 60% result is reproducible. Train the exact same configuration with at least three training seeds (for example 42, 43 and 44). Keep the same experiment name so compare_runs.py can group the runs. Use a larger final evaluation sample if runtime allows (for example 100 episodes per trained model).

Example, run separately or in a shell loop:
for seed in 42 43 44; do
  python run_experiment.py \
    --stage 1_known_fixed \
    --name stage1_best_config \
    --seed "$seed" \
    --episodes 100 \
    --no-gif \
    --hypothesis "Repeat the best-known Stage 1 configuration across training seeds." \
    --set env.start_jitter=0.0 \
         env.action_residual_scale=1.0 \
         reward.wall_weight=3.0 \
         reward.wall_margin=1.0 \
         reward.formation_weight=0.5 \
         reward.separation_weight=1.0 \
         reward.separation_margin=0.3
done
The Stage 1 preset supplies 300,000 timesteps. Make sure the shell loop is run inside the activated .venv and from the repository root. If cluster wall-time limits make three runs in one allocation unsuitable, submit separate Slurm jobs.

Compare the runs:
python compare_runs.py --last 3 --md
Do not advance the curriculum solely because one seed reaches 60%. The original README's proposed gate is approximately 80% success on final evaluation across at least three seeds. If that gate is unrealistic for the course timeline, agree and document a revised criterion rather than silently changing it.
10.3 Investigate reward alignment: bottleneck-aware progress (proposed, not yet implemented)
The current progress term rewards the mean reduction in target distance:
progress = float(np.mean(old_dist - target_dist)) if self.target_found else 0.0
But mission success requires all drones to be within the success radius. Mean progress can be positive even if one drone moves away or remains far behind.

A proposed candidate is a blended mean/worst-drone progress signal:
mean_progress = float(np.mean(old_dist - target_dist))
worst_progress = float(np.max(old_dist) - np.max(target_dist))
progress = 0.5 * mean_progress + 0.5 * worst_progress
Apply only when the target is found; otherwise progress remains zero, as in the current design.

Status: proposed in the chat but not reported as implemented or tested. Do not describe it as part of the current results. Before running it:
1. Add a small unit/sanity test for a case where all drones approach equally (the blended reward should match their common progress).
2. Add a case where two drones approach but the farthest drone moves away (the blended signal should be lower than mean-only progress).
3. Change only the progress aggregation; retain the frozen best-known settings.
4. Compare the result against the multi-seed baseline, not merely seed 42.
This is a hypothesis about reward alignment, not a guaranteed fix. The max-distance term can change which drone is the bottleneck from step to step, so inspect the reward scale and trajectories.
10.4 Separate training instability from reward problems
For each run:
- Inspect eval_curve.csv and training_curves.png before concluding that the reward is wrong.
- Check whether periodic evaluation success is trending upward, flat or unstable.
- Compare the last evaluation checkpoints with the final 30/100-episode evaluation. Periodic evaluation uses a smaller sample (10 in the reported setup), so its success rate is coarse and noisy.
- If learning is still clearly improving at the end, consider increasing timesteps without changing reward/configuration.
- If it is flat or collapsing, inspect trajectories and reward breakdown before choosing a code change.
Do not respond to every poor run by changing the separation margin, separation weight, PPO learning rate and training duration together. One experiment should test one primary hypothesis.
10.5 Recommended experiment record format
For every run, keep:
- One-sentence hypothesis (--hypothesis).
- Exact config.json.
- Training seed and total timesteps.
- Number of final evaluation episodes and evaluation seed range.
- Success rate and confidence interval.
- Outcome counts.
- Mean and worst-drone final distance.
- Mean formation error and minimum pair distance.
- Per-outcome reward breakdown.
- A comparison with the frozen baseline.
Do not delete failed runs. They are part of the experimental record.
11. Curriculum plan after Stage 1
Move to the next stage only after Stage 1 behaviour is reliable enough across multiple training seeds. The original target is approximately 80% success across at least three seeds.
Stage	What it tests	Suggested budget
1_known_fixed	Fixed start/target, target known, no obstacles. Validate navigation and coordination.	300k
2_known_random	Random start and target, target still known, no obstacles.	500k
3_search	Hidden target and exploration; no obstacles.	1M
4_search_few_obstacles	Search plus 1–2 random obstacles.	1M
5_full	Randomized start/target, hidden target and 3–5 random obstacles.	3M


These are starting budgets, not convergence guarantees. Read periodic evaluation curves. If the policy is still improving, train longer; if performance collapses, investigate before scaling the budget.

The fully randomized task is expected to be difficult for a memoryless MLP. Some obstacle layouts may be unusually difficult or effectively blocked. Generalization should be evaluated on held-out scenarios, not only training-like layouts.
12. Limitations and work not yet implemented
- No automated unit/integration test suite has been reported.
- No periodic model checkpoints or resume-after-interruption mechanism has been reported; the runner saves the final model.
- No frequency/duration metric for cohesion or formation violations has been reported; only mean errors are recorded.
- The policy is feed-forward and has no explicit temporal memory for obstacles.
- Raycasting is vectorised with NumPy; it was not benchmarked or compared against the old loop version on this cluster. Results in section 8 predate the vectorised raycast, blended progress reward, relative-position scaling (/3), per-episode-phase change and reward normalisation (ppo.normalize_reward), so they are not directly comparable to new runs.
- ppo.n_envs > 1 / multi-process training was not validated in the original README; smoke-test it before cluster-scale use.
- Thirty evaluation episodes give relatively wide confidence intervals; use more episodes for final reporting.
- min_obstacle_clearance=inf is expected in zero-obstacle Stage 1 runs; it is not a numerical error in that context.
- Stage 1 success has not been validated across multiple training seeds.
- Bottleneck-aware progress is only a proposal as of this README update.
- The current best Stage 1 PPO result is below Greedy's 100% and below the proposed 80% multi-seed curriculum gate.
13. Git workflow and merging experimental/wall-proximity into advanced
The user reported that the current working branch is experimental/wall-proximity, and the target branch is advanced. Before merging, verify that the current branch really contains the desired source changes and that all intended files are committed. Generated runs/ directories and .venv/ should normally remain untracked/ignored.
13.1 Inspect current state
From the project directory:
cd /data/leuven/394/vsc39468/drone-swarm
source .venv/bin/activate

git branch --show-current
git status --short
git worktree list
git branch -vv
Confirm that git branch --show-current prints:
experimental/wall-proximity
Review git diff and git diff --staged. Check that the latest code includes the wall-proximity reward assignment, formation reward/metric, and the intended action-residual option. Do not merge an uncommitted working tree.
13.2 Commit the experimental branch
Stage only intended source/documentation files. For example:
git add README.md config.py environment.py diagnostics.py training.py run_experiment.py compare_runs.py requirements.txt
Adjust the list to match the actual changed files. Do not add .venv/ or generated runs/ artifacts unless there is a deliberate reason.

Review and commit:
git status --short
git diff --cached
git commit -m "Add wall and formation rewards for swarm coordination"
If Git says there is nothing to commit, verify that the latest changes are already committed and inspect the log.
13.3 Fetch the remote and update advanced
git fetch origin
git switch advanced
git pull --ff-only origin advanced
If the local advanced branch does not exist but origin/advanced does:
git switch --track -c advanced origin/advanced
Worktree warning: Git may refuse git switch advanced if advanced is already checked out in another worktree. This is the same class of issue encountered when trying to delete testing. Run git worktree list; then perform the merge from the worktree where advanced is already checked out. Do not remove a worktree unless its status is clean and you know it is no longer needed.
13.4 Merge the experimental branch
Once on advanced:
git merge --no-ff experimental/wall-proximity \
  -m "Merge wall proximity and swarm coordination experiments"
If there are conflicts:
git status
Open each conflicted file, reconcile the changes deliberately, then:
git add <resolved-files>
git commit
Do not use git merge -X ours, git merge -X theirs or force-push as a shortcut without understanding the conflicts.
13.5 Validate and push advanced
After the merge:
python -m py_compile config.py environment.py training.py diagnostics.py run_experiment.py compare_runs.py
python run_experiment.py --list-params
python run_experiment.py \
  --stage 1_known_fixed \
  --timesteps 2000 \
  --episodes 3 \
  --no-gif \
  --name post_merge_smoke

git status --short
git log --oneline --decorate -5
git push origin advanced
Only push after the smoke test and source review pass. The 2,000-step run validates the merged pipeline, not learning quality.
14. Quick command reference
# Project and virtual environment
cd /data/leuven/394/vsc39468/drone-swarm
source .venv/bin/activate
which python
python --version

# Inspect parameters and dependencies
python run_experiment.py --list-params
python -m pip check

# Syntax check
python -m py_compile config.py environment.py training.py diagnostics.py run_experiment.py compare_runs.py

# Stage 1 smoke test
python run_experiment.py --stage 1_known_fixed --timesteps 2000 --episodes 3 --no-gif --name smoke

# Best-known Stage 1 configuration (300k steps from the stage preset)
python run_experiment.py --stage 1_known_fixed --name stage1_best --seed 42 --episodes 100 --no-gif \
  --set env.start_jitter=0.0 env.action_residual_scale=1.0 \
       reward.wall_weight=3.0 reward.wall_margin=1.0 \
       reward.formation_weight=0.5 \
       reward.separation_weight=1.0 reward.separation_margin=0.3

# Compare recent runs
python compare_runs.py --last 5 --md

# Git status and worktrees
git status --short
git branch --show-current
git worktree list
15. Glossary
- Episode: one attempt from reset to success, collision or time limit.
- Return: total reward over an episode. It is not a substitute for success rate.
- Policy: neural network mapping observations to actions.
- PPO: policy-gradient algorithm that collects rollouts and performs clipped policy updates.
- Rollout / n_steps: experience collected per environment between PPO updates.
- Learning rate: optimizer step size.
- Discount (gamma): controls how strongly future rewards contribute to current value estimates.
- Entropy coefficient: encourages action exploration.
- log_std_init: initial Gaussian action-noise scale in SB3's policy distribution; -1.0 corresponds to standard deviation exp(-1), approximately 0.37.
- Deterministic evaluation: policy uses its mean action rather than sampling action noise.
- Terminated vs truncated: terminated means success or collision; truncated means the episode hit its time limit.
- Seed: controls pseudo-randomness. Different training seeds can produce materially different policies.
- Curriculum: training on easier task variants before introducing the full task.
- Cohesion: keeping drones near the swarm centroid.
- Formation: maintaining the intended pairwise geometry.
- Separation: discouraging drones from getting too close.
- Bottleneck drone: the drone currently farthest from the target; proposed progress shaping would give its improvement more influence.
Handoff rule: treat config.json in each run as the source of truth for the parameters actually used; treat episode CSVs as the source of truth for outcomes and reward terms; and do not promote a reward/control change based on one training seed al