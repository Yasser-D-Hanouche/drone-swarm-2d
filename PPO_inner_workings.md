# PPO inner workings

How PPO training works in this project, where the training time goes, and how it could be shortened. Numbers refer to the Stage 4 setup: 3 drones, 2M environment steps, CPU batch job with 17 cores (16 environments × 128 steps per update). For the design, see [README.md](README.md); for experiment history, see [LOG.md](LOG.md).

## 1. The loop in one paragraph

PPO repeats two phases about 977 times in a 2M-step run:

1. **Collect data:** the current network flies the drones for a while.
2. **Learn from it:** adjust the network's weights so good actions become more likely, then discard the data and collect fresh data with the improved network.

## 2. What goes in and what comes out

**Inputs**

- **Observations:** each drone gets its own row of 126 numbers every step (velocity, waypoint direction, neighbour positions, rays, wall distances, searched map, and so on; README section 4.6).
- **Rewards:** after each step the environment gives each drone a reward (progress, exploration, penalties, bonuses; README section 5).

**Outputs**

- **An action per drone per step:** the network outputs an average acceleration (2 numbers) and a spread. During training the action is sampled around that average, which lets the policy try variations. In evaluation it uses the average.
- **A value estimate:** a second output head predicts how much reward is still to come from the current situation. PPO uses it to judge whether an action turned out better or worse than expected.
- **At the end, `model.zip`:** the trained network, which maps an observation to an action.

## 3. Samples, updates, epochs and gradient steps

### 3.1 Definitions

| Term | Meaning | Number in this project |
|---|---|---|
| Sample | One drone at one step: its observation, the action taken, the reward, the value estimate, and how likely that action was. All drones share one network, so one environment step gives 3 samples. | 3 per environment step |
| Rollout | The data collected before one update: 16 environments × 128 steps | 2,048 environment steps = 6,144 samples |
| Advantage | For each sample, how much better than expected things went. Computed with the discount (γ = 0.99) and the GAE factor (λ = 0.95). | One per sample |
| Update | One round of collect, score, optimise, then discard the data | 2M ÷ 2,048 ≈ 977 per run |
| Epoch | One pass over all 6,144 samples of a rollout, in shuffled order | 10 per update |
| Minibatch | A slice of the samples used for one gradient step | 192 samples; 32 per epoch |
| Gradient step | One small change of the network weights, computed from one minibatch | 32 × 10 = 320 per update |

What one gradient step does:

1. Run the 192 samples through the network.
2. Compute the loss:
   - a policy term: do more of what went better than expected, but clipped (`clip_range` 0.2) so one update cannot move the policy too far;
   - a value term: predict returns better;
   - a small entropy bonus (`ent_coef` 0.005): keep exploring.
3. Backpropagate.
4. Let the Adam optimiser nudge the weights by an amount set by the learning rate (3e-4, falling linearly to 0 over the run).

### 3.2 The same thing as a sports team learning from video

The three drones are a team. The network is the team's playbook: a set of habits saying "in this situation, do this."

| PPO term | In the analogy |
|---|---|
| Sample | One frame of the video: one drone at one moment, what it saw, what it did, what happened next |
| Rollout | Play with the current playbook while a camera records: 6,144 frames of footage |
| Update | One "play, then review" cycle. Play, study the footage, improve the playbook, go back out. The old footage is thrown away, because it shows habits the team no longer has. |
| Epoch | Watching all the footage once. The team watches it 10 times, but a rule (clipping) forbids overhauling the playbook based on one tape. |
| Gradient step | The footage is cut into clips of 192 frames. After each clip the team makes one small correction: "in situations like these, lean a little more toward what worked and away from what didn't." |

```
Training run (2M steps)
└── 977 updates ("play, then review" cycles)
    ├── Play: collect 6,144 frames of footage
    └── Review: watch it 10 times (10 epochs)
        └── each viewing: 32 clips of 192 frames
            └── after each clip: 1 small correction (gradient step)
```

## 4. Why Python overhead dominates

One gradient step involves a few dozen PyTorch operations: the network layers, log-probabilities, the probability ratio, clipping, the losses, the backward pass, gradient clipping and the Adam step. Each operation has a fixed launch cost in Python and PyTorch's dispatcher, a few microseconds to tens of microseconds, regardless of the size of the data. The arithmetic itself is tiny: 192 samples through a 64-wide network take microseconds.

So most of the measured 0.83 s per update (about 2.6 ms per gradient step × 320) is bookkeeping, not maths. In the analogy, every correction comes with a fixed amount of admin (rewind, pause, take notes, edit the playbook), and the clips are so short that the admin takes longer than the watching.

Consequences:

- **More torch threads do not help.** Measured on 2026-11-15: one update took 0.83 s with 1 thread and 0.89 s with 2. Threads speed up the arithmetic inside an operation, which is not the bottleneck. `training.py` therefore sets `torch.set_num_threads(1)`.
- **A GPU does not help** for the same reason, and adds a transfer to and from the card on every step (LOG 2026-11-14.5).
- **More cores help only the environment stepping,** which already runs in parallel processes.

## 5. Where the 25 to 35 minutes go

Stage 4, 2M steps, 17 cores:

| Part | Time | Notes |
|---|---|---|
| PPO updates (977 × about 1.0 s) | about 16 min | One process; cores cannot help |
| Environment stepping (977 × about 0.3 s with 16 environments) | about 5 min | Already parallel |
| Periodic evaluation during training (40 × 10 episodes, in the main process) | estimated 1 to 3 min | Not measured separately |
| Final evaluation (100 episodes × 3 policies), plots and GIFs | a few minutes | One process |

The updates are the floor: even with unlimited cores, training takes at least about 16 minutes. Reaching 10 to 15 minutes in total means making each update cheaper or doing fewer of them.

## 6. Options to bring a run down to 10 to 15 minutes

| # | Option | Expected effect | Changes learning? | Cost to adopt |
|---|---|---|---|---|
| 1 | Larger minibatches (`ppo.batch_size` 192 to 768) | 4× fewer gradient steps (80 instead of 320 per update); update about 0.25 s; training about 977 × (0.25 + 0.3) s ≈ 9 min | Yes: fewer, smoother steps; may need a higher learning rate | Validate on 2 to 3 Stage 4 seeds against 90% and 84% |
| 2 | Less periodic evaluation (`run.eval_freq` 50k to 250k, or 0) | Saves an estimated 1 to 3 min | No | None; progress is judged from `training_episodes.csv`, which is steadier anyway |
| 3 | Warm start from the previous stage's model | Stage 4 might need 0.5 to 1M steps instead of 2M (the Stage 3 model scored 48% on Stage 4 without obstacle training) | It changes the "every stage from scratch" setup | Small code addition (`--init-model`); fits the view that Stage 5 is the deliverable |
| 4 | Fewer epochs (`ppo.n_epochs` 10 to 5) | Halves the update time | Yes | Validation; do not combine with option 1 in the same test |
| 5 | Batched environment (all environments as one numpy array in one process) | Attacks only the 5-minute environment part | No | Large rewrite; not worth it now |

Recommendation: option 2 (free) together with option 1 (validated on 2 to 3 seeds). Expected total: about 9 to 10 minutes of training plus about 3 minutes of evaluation, so 12 to 13 minutes per run. If larger minibatches lower success, fall back to option 3. None of these has been applied yet.

## 7. Practical note: a new run folder does not appear in VS Code

A batch job creates its run folder within seconds of starting (for example, job 62704103 started at 11:32:00 and created `runs/20261115_113213_s4_mask` at 11:32:13). If it does not show up in the VS Code Explorer, the Explorer has not refreshed: changes made by a job on another node do not trigger VS Code's file watcher on the VSC data filesystem.

- Click the refresh icon at the top of the Explorer panel, or collapse and reopen `runs/`.
- `ls runs/` in the terminal always shows the real state.
- `squeue -M wice -u $USER` shows whether the job is running (`R`) or still queued (`PD`). A queued job has no folder yet.
- `python3 cluster/progress.py runs/<folder>` shows how far the training is.

---

*Written 2026-11-15.*

*Project status at the time of writing:*

- *Branch: `martin-branch`. Latest commit: `0dbcaa8` ("Stage 4 done. Gifs and trajectory generation need to be changed").*
- *Uncommitted changes: `LOG.md` and `README.md` (Stage 4 entry and one corrected sentence), `diagnostics.py` and `run_experiment.py` (per-episode trajectory panels with each episode's obstacles; GIFs with true-size drones, rays, searched map, waypoint and target), and this file.*
- *Curriculum: Stage 3 passes the gate (94%, 93%, 95% on three seeds). Stage 4 passes the gate: 90%, 84% and 93% on three seeds at 2M steps (the third, seed 44, is run `20261115_113213_s4_mask`, with GIFs). Stage 5 has not been run with the current settings.*
