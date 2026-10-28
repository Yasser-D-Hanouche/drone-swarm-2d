"""Rollouts, baselines, metrics, plots, failure diagnosis and the markdown report."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from config import Config
from environment import OUTCOMES, REWARD_TERMS, DroneSwarm2D, draw_scene

OUTCOME_COLORS = {
    "success": "tab:green",
    "obstacle": "tab:red",
    "boundary": "tab:orange",
    "drone_collision": "tab:purple",
    "timeout": "tab:gray",
}
DRONE_COLORS = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:cyan", "tab:pink"]


# ----------------------------------------------------------------------------- policies

def random_policy(env):
    return lambda obs: env.action_space.sample()


def greedy_policy(env):
    """Naive baseline: each drone flies straight to a compact slot around the target, ignoring obstacles."""
    offsets = 0.5 * env.formation

    def act(obs):
        to_goal = env.target + offsets - env.positions
        dist = np.linalg.norm(to_goal, axis=1, keepdims=True)
        desired = to_goal / np.maximum(dist, 1e-8) * np.minimum(env.cfg.max_speed, dist)
        return np.clip((desired - env.velocities) / (env.cfg.max_accel * env.cfg.dt), -1.0, 1.0)

    return act


def ppo_policy(model):
    return lambda env: (lambda obs: model.predict(obs, deterministic=True)[0])


# ----------------------------------------------------------------------------- rollouts

def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    traj = [env.positions.copy()]
    terms = dict.fromkeys(REWARD_TERMS, 0.0)
    form_errors, min_clear, min_pair, min_mean_dist, path = [], math.inf, math.inf, math.inf, 0.0

    while True:
        prev = env.positions.copy()
        obs, _, terminated, truncated, info = env.step(policy(obs))
        traj.append(env.positions.copy())
        path += float(np.linalg.norm(env.positions - prev, axis=1).mean())
        for k, v in info["reward_terms"].items():
            terms[k] += v
        form_errors.append(info["formation_error"])
        min_clear = min(min_clear, info["min_obstacle_clearance"])
        min_pair = min(min_pair, info["min_pair_distance"])
        min_mean_dist = min(min_mean_dist, info["mean_target_distance"])
        if terminated or truncated:
            break

    final = np.linalg.norm(env.positions - env.target, axis=1)
    cx, cy = info["collision_pos"] or (math.nan, math.nan)
    row = {
        "seed": seed,
        "outcome": info["termination"],
        "success": info["success"],
        "collision": info["collision"],
        "reward": sum(terms.values()),
        "steps": env.steps,
        "final_mean_dist": float(final.mean()),
        "final_worst_dist": float(final.max()),
        "drones_at_target": int((final < env.cfg.success_radius).sum()),
        "closest_mean_dist": min_mean_dist,
        "path_length": path,
        "min_obstacle_clearance": min_clear,
        "min_pair_distance": min_pair,
        "mean_formation_error": float(np.mean(form_errors)),
        "collision_x": cx,
        "collision_y": cy,
    }
    row.update({f"r_{k}": v for k, v in terms.items()})
    row["traj"] = np.asarray(traj)
    return row


def run_episodes(cfg: Config, policy_factory, n: int, seed0: int):
    env = DroneSwarm2D(cfg.env, cfg.reward)
    env.action_space.seed(seed0)
    policy = policy_factory(env)
    return [rollout(env, policy, seed0 + i) for i in range(n)]


def strip_traj(rows):
    return [{k: v for k, v in r.items() if k != "traj"} for r in rows]


def wilson(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(rows):
    n = len(rows)
    rewards = np.array([r["reward"] for r in rows])
    by_outcome = {o: [r for r in rows if r["outcome"] == o] for o in OUTCOMES}
    successes = len(by_outcome["success"])
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    return {
        "n": n,
        "success_rate": successes / n,
        "success_ci95": list(wilson(successes, n)),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "outcome_counts": {o: len(v) for o, v in by_outcome.items()},
        "outcome_rates": {o: len(v) / n for o, v in by_outcome.items()},
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_steps": mean("steps"),
        "mean_final_dist": mean("final_mean_dist"),
        "mean_worst_final_dist": mean("final_worst_dist"),
        "mean_drones_at_target": mean("drones_at_target"),
        "mean_path_length": mean("path_length"),
        "mean_min_obstacle_clearance": mean("min_obstacle_clearance"),
        "mean_formation_error": mean("mean_formation_error"),
        "mean_terms": {k: mean(f"r_{k}") for k in REWARD_TERMS},
        "reward_by_outcome": {o: float(np.mean([r["reward"] for r in v])) for o, v in by_outcome.items() if v},
        "terms_by_outcome": {
            o: {k: float(np.mean([r[f"r_{k}"] for r in v])) for k in REWARD_TERMS}
            for o, v in by_outcome.items() if v
        },
    }


# ----------------------------------------------------------------------------- plots

def plot_trajectories(results: dict, cfg: Config, path, max_episodes=30):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6.4), squeeze=False)
    for ax, (name, rows) in zip(axes[0], results.items()):
        draw_scene(ax, cfg.env)
        counts = dict.fromkeys(OUTCOMES, 0)
        for r in rows[:max_episodes]:
            counts[r["outcome"]] += 1
            for d in range(r["traj"].shape[1]):
                ax.plot(r["traj"][:, d, 0], r["traj"][:, d, 1], color=OUTCOME_COLORS[r["outcome"]], alpha=0.35, lw=1)
            if not math.isnan(r["collision_x"]):
                ax.scatter(r["collision_x"], r["collision_y"], marker="x", color="black", s=45, zorder=6)
        ax.set_title(name)
        handles = [
            plt.Line2D([], [], color=OUTCOME_COLORS[o], label=f"{o} ({c})") for o, c in counts.items() if c
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.suptitle("Evaluation trajectories (x = collision point)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_reward_breakdown(summary, path):
    terms_by_outcome = summary["terms_by_outcome"]
    outcomes = list(terms_by_outcome)
    x = np.arange(len(REWARD_TERMS))
    width = 0.8 / max(len(outcomes), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, o in enumerate(outcomes):
        ax.bar(x + i * width, [terms_by_outcome[o][k] for k in REWARD_TERMS], width,
               color=OUTCOME_COLORS[o], label=f"{o} (return {summary['reward_by_outcome'][o]:.1f})")
    ax.set_xticks(x + width * (len(outcomes) - 1) / 2, REWARD_TERMS)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("mean summed reward per episode")
    ax.set_title("Where the return comes from, by episode outcome (PPO, evaluation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _bin_training(rows, total, n_bins=25):
    edges = np.linspace(0, total, n_bins + 1)
    idx = np.clip(np.digitize([r["timesteps"] for r in rows], edges) - 1, 0, n_bins - 1)
    centers, groups = [], []
    for b in range(n_bins):
        group = [r for r, i in zip(rows, idx) if i == b]
        if group:
            centers.append((edges[b] + edges[b + 1]) / 2)
            groups.append(group)
    return np.array(centers), groups


def plot_training(training_rows, eval_curve, total, path):
    if not training_rows:
        return
    centers, groups = _bin_training(training_rows, total)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    fractions = [[np.mean([r["outcome"] == o for r in g]) for g in groups] for o in OUTCOMES]
    axes[0].stackplot(centers, fractions, labels=OUTCOMES, colors=[OUTCOME_COLORS[o] for o in OUTCOMES], alpha=0.8)
    if eval_curve:
        axes[0].plot([e["timesteps"] for e in eval_curve], [e["success_rate"] for e in eval_curve],
                     "ko-", ms=4, label="periodic eval success")
    axes[0].set_ylabel("fraction of episodes")
    axes[0].set_title("How training episodes end")
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)

    axes[1].plot(centers, [np.mean([r["reward"] for r in g]) for g in groups], label="training")
    if eval_curve:
        axes[1].plot([e["timesteps"] for e in eval_curve], [e["mean_reward"] for e in eval_curve], "ko-", ms=4, label="periodic eval")
    axes[1].set_ylabel("episode return")
    axes[1].set_title("Return")
    axes[1].legend(fontsize=8)

    for k in REWARD_TERMS:
        values = [np.mean([r[f"r_{k}"] for r in g]) for g in groups]
        if any(abs(v) > 1e-9 for v in values):
            axes[2].plot(centers, values, label=k)
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_ylabel("summed reward per episode")
    axes[2].set_xlabel("training timesteps")
    axes[2].set_title("Reward terms (which incentive is driving behaviour?)")
    axes[2].legend(fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_gif(cfg: Config, row, path, title, fps=15, stride=2, hold=12):
    traj = row["traj"]
    frames = list(range(0, len(traj), stride)) + [len(traj) - 1] * hold
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_scene(ax, cfg.env)
    n = traj.shape[1]
    dots = [ax.plot([], [], "o", color=DRONE_COLORS[d % len(DRONE_COLORS)], ms=8, zorder=5)[0] for d in range(n)]
    trails = [ax.plot([], [], color=DRONE_COLORS[d % len(DRONE_COLORS)], alpha=0.6, lw=1.3, zorder=4)[0] for d in range(n)]
    heading = ax.set_title("")

    def update(k):
        f = frames[k]
        for d in range(n):
            dots[d].set_data([traj[f, d, 0]], [traj[f, d, 1]])
            trails[d].set_data(traj[: f + 1, d, 0], traj[: f + 1, d, 1])
        end = f == len(traj) - 1
        heading.set_text(f"{title} | step {f}" + (f" | {row['outcome'].upper()}" if end else ""))
        return dots + trails

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)


# ----------------------------------------------------------------------------- diagnosis

_CRASH_ADVICE = {
    "obstacle": (
        "Most failures are obstacle collisions: the policy takes the direct route and clips an obstacle.",
        "Enable shaping (reward.obstacle_margin=0.8 reward.obstacle_weight=1.0), raise reward.collision_penalty, or train longer.",
    ),
    "boundary": (
        "Many episodes end by hitting the world boundary: the policy overshoots or swerves around obstacles into walls.",
        "Check momentum (env.max_accel), add start jitter for variety, or train longer.",
    ),
    "drone_collision": (
        "Drones collide with each other, typically while converging on the same target.",
        "Raise reward.formation_weight, or increase env.cohesion_distance so drones are rewarded for keeping spacing.",
    ),
}


def _success_by_third(training_rows):
    if len(training_rows) < 30:
        return None
    ts = np.array([r["timesteps"] for r in training_rows])
    edges = np.quantile(ts, [0, 1 / 3, 2 / 3, 1])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        group = [r for r, t in zip(training_rows, ts) if lo <= t <= hi]
        out.append(float(np.mean([r["outcome"] == "success" for r in group])) if group else 0.0)
    return out


def diagnose(cfg: Config, summaries: dict, training_rows: list, eval_curve: list):
    out = []
    add = lambda finding, suggestion: out.append({"finding": finding, "suggestion": suggestion})
    ppo = summaries["PPO"]
    rates = ppo["outcome_rates"]

    if cfg.env.start_jitter == 0:
        add(
            f"env.start_jitter=0: all {ppo['n']} evaluation episodes start identically, so the result is one episode repeated "
            "(0% or 100%), not an estimate of generalisation.",
            "Use --set env.start_jitter=0.3 for an honest evaluation.",
        )

    if ppo["success_rate"] >= 0.9:
        add("The policy solves the task reliably.", "Make the task harder (more jitter, obstacles) or reduce timesteps to find the minimum budget.")
    else:
        failures = {o: rates[o] for o in OUTCOMES if o != "success"}
        main, rate = max(failures.items(), key=lambda kv: kv[1])
        if main in _CRASH_ADVICE and rate >= 0.3:
            add(f"{rate:.0%} of episodes end in '{main}'. " + _CRASH_ADVICE[main][0], _CRASH_ADVICE[main][1])
        if rates["timeout"] >= 0.3:
            if ppo["mean_worst_final_dist"] < 3 * cfg.env.success_radius:
                add(
                    f"{rates['timeout']:.0%} of episodes time out near the target (worst drone ends "
                    f"{ppo['mean_worst_final_dist']:.2f} away, success needs < {cfg.env.success_radius}): "
                    "the swarm arrives but not every drone is inside the success radius.",
                    "Check that env.formation fits inside env.success_radius, raise reward.success_bonus, or loosen env.success_radius.",
                )
            else:
                add(
                    f"{rates['timeout']:.0%} of episodes time out far from the target "
                    f"(mean distance {ppo['mean_final_dist']:.1f}): the policy hovers or detours instead of progressing.",
                    "Raise reward.progress_weight, check reward.formation_weight is not larger than progress, or train longer.",
                )

    rbo = ppo["reward_by_outcome"]
    crash_returns = [rbo[o] for o in ("obstacle", "boundary", "drone_collision") if o in rbo]
    if "timeout" in rbo and crash_returns and max(crash_returns) > rbo["timeout"]:
        add(
            f"Crashing pays: best crash return {max(crash_returns):.1f} > timeout return {rbo['timeout']:.1f}. "
            "Per-step costs outweigh the collision penalty, so ending the episode early is attractive.",
            "Raise reward.collision_penalty above the cumulative step/formation cost, or lower reward.step_penalty / reward.formation_weight.",
        )

    terms = ppo["mean_terms"]
    if abs(terms["formation"]) > abs(terms["progress"]):
        add(
            f"Formation penalty ({terms['formation']:.1f}) outweighs progress reward ({terms['progress']:.1f}) per episode.",
            "Lower reward.formation_weight, or raise reward.progress_weight.",
        )

    greedy = summaries.get("Greedy")
    if greedy and ppo["success_rate"] <= greedy["success_rate"]:
        add(
            f"PPO ({ppo['success_rate']:.0%}) does not beat the naive goal-seeker ({greedy['success_rate']:.0%}).",
            "Learning is not adding value yet: train longer or fix the reward before tuning anything else.",
        )

    thirds = _success_by_third(training_rows)
    if thirds is not None:
        first, mid, last = thirds
        if last == 0:
            add(
                "No training episode ever succeeded in the final third, so the success bonus gave no learning signal.",
                "Rely on dense shaping (progress_weight), shrink the task (fewer obstacles / closer target) to bootstrap, or raise ent_coef for exploration.",
            )
        elif last > mid + 0.05:
            add(f"Training success is still rising (thirds: {first:.0%} -> {mid:.0%} -> {last:.0%}).", "Increase run.timesteps; the run is likely under-trained.")
        elif last < mid - 0.1:
            add(f"Training success dropped late (thirds: {first:.0%} -> {mid:.0%} -> {last:.0%}): unstable learning.", "Lower ppo.learning_rate or raise ppo.n_steps / ppo.batch_size.")

    if not out:
        add("No obvious problem detected.", "Compare against other runs with compare_runs.py.")
    return out


# ----------------------------------------------------------------------------- report

def _fmt_rate(summary):
    lo, hi = summary["success_ci95"]
    return f"{summary['success_rate']:.0%} [{lo:.0%}-{hi:.0%}]"


def write_report(path, cfg: Config, overrides: dict, summaries: dict, diagnosis: list, training_rows: list, files: list):
    lines = [f"# Experiment: {cfg.run.name}", "", f"- Created: {datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"- Seed: {cfg.run.seed} | timesteps: {cfg.run.timesteps:,} | eval episodes: {cfg.run.eval_episodes}"]
    if cfg.run.hypothesis:
        lines.append(f"- Hypothesis: {cfg.run.hypothesis}")
    lines += ["", "## Changes vs defaults", ""]
    changed = {k: v for k, v in overrides.items() if not k.startswith("run.")}
    lines += [f"- `{k} = {v}`" for k, v in changed.items()] or ["- none (defaults)"]

    lines += ["", "## Results (same evaluation seeds for every policy)", "",
              "| Policy | Success [95% CI] | Collision | Timeout | Return | Steps | Final dist | Worst drone dist |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {_fmt_rate(s)} | {s['collision_rate']:.0%} | {s['outcome_rates']['timeout']:.0%} | "
            f"{s['mean_reward']:.1f} ± {s['std_reward']:.1f} | {s['mean_steps']:.0f} | {s['mean_final_dist']:.2f} | {s['mean_worst_final_dist']:.2f} |"
        )

    lines += ["", "## How episodes end", "", "| Policy | " + " | ".join(OUTCOMES) + " |", "|---|" + "---|" * len(OUTCOMES)]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(str(s["outcome_counts"][o]) for o in OUTCOMES) + " |")

    ppo = summaries["PPO"]
    lines += ["", "## PPO return by outcome (mean summed reward terms)", "",
              "| Outcome | Return | " + " | ".join(REWARD_TERMS) + " |", "|---|---|" + "---|" * len(REWARD_TERMS)]
    for o, terms in ppo["terms_by_outcome"].items():
        lines.append(f"| {o} | {ppo['reward_by_outcome'][o]:.1f} | " + " | ".join(f"{terms[k]:.2f}" for k in REWARD_TERMS) + " |")

    lines += ["", "## Diagnosis", ""]
    for d in diagnosis:
        lines.append(f"- **{d['finding']}**  \n  Try: {d['suggestion']}")

    lines += ["", "## Files", ""] + [f"- `{f}`" for f in files]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
