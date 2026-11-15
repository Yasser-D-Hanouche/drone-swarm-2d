from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

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


def random_policy(env):
    return lambda obs: env.action_space.sample()


def greedy_policy(env):
    # Obstacle-blind searcher: fly to the nearest unsearched cell until the target is found, then to the target.
    def act(obs):
        centroid = env.positions.mean(axis=0)
        if env.target_found:
            goal = env.target
        elif not env.searched.all():
            cells = env.cell_centers[~env.searched]
            goal = cells[np.linalg.norm(cells - centroid, axis=1).argmin()]
        else:
            goal = centroid
        to_goal = goal + (env.positions - centroid) - env.positions
        dist = np.linalg.norm(to_goal, axis=1, keepdims=True)
        desired = to_goal / np.maximum(dist, 1e-8) * np.minimum(env.cfg.max_speed, dist)
        return np.clip(env.to_frame((desired - env.velocities) / (env.cfg.max_accel * env.cfg.dt)), -1.0, 1.0)

    return act


def ppo_policy(model):
    return lambda env: (lambda obs: model.predict(obs, deterministic=True)[0])


def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    traj = [env.positions.copy()]
    terms = dict.fromkeys(REWARD_TERMS, 0.0)
    cohesion_errors = []
    formation_errors = []

    min_clear = math.inf
    min_pair = math.inf
    min_mean_dist = math.inf
    path = 0.0

    while True:
        prev = env.positions.copy()
        obs, _, terminated, truncated, info = env.step(policy(obs))
        traj.append(env.positions.copy())
        path += float(np.linalg.norm(env.positions - prev, axis=1).mean())
        for k, v in info["reward_terms"].items():
            terms[k] += v
        cohesion_errors.append(info["cohesion_error"])
        formation_errors.append(info["formation_error"])
        min_clear = min(min_clear, info["min_obstacle_clearance"])
        min_pair = min(min_pair, info["min_pair_distance"])
        min_mean_dist = min(min_mean_dist, info["mean_target_distance"])
        found = info["target_found"]
        if terminated or truncated:
            break

    final = np.linalg.norm(env.positions - env.target, axis=1)
    cx, cy = info["collision_pos"] or (math.nan, math.nan)
    row = {
        "seed": seed,
        "outcome": info["termination"],
        "success": info["success"],
        "collision": info["collision"],
        "target_found": found,
        "searched_fraction": info["searched_fraction"],
        "reward": sum(terms.values()),
        "steps": env.steps,
        "final_mean_dist": float(final.mean()),
        "final_worst_dist": float(final.max()),
        "drones_at_target": int((final < env.cfg.success_radius).sum()),
        "closest_mean_dist": min_mean_dist,
        "path_length": path,
        "min_obstacle_clearance": min_clear,
        "min_pair_distance": min_pair,
        "mean_cohesion_error": float(np.mean(cohesion_errors)),
        "mean_formation_error": float(np.mean(formation_errors)),
        "collision_x": cx,
        "collision_y": cy,
    }
    row.update({f"r_{k}": v for k, v in terms.items()})
    row["traj"] = np.asarray(traj)
    row["obstacles"] = env.obstacles.copy()
    row["target"] = env.target.copy()
    return row


def run_episodes(cfg: Config, policy_factory, n: int, seed0: int):
    env = DroneSwarm2D(cfg.env, cfg.reward)
    env.action_space.seed(seed0)
    policy = policy_factory(env)
    return [rollout(env, policy, seed0 + i) for i in range(n)]


def strip_traj(rows):
    return [{k: v for k, v in r.items() if k not in ("traj", "obstacles", "target")} for r in rows]


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
        "found_rate": mean("target_found"),
        "mean_searched_fraction": mean("searched_fraction"),
        "outcome_counts": {o: len(v) for o, v in by_outcome.items()},
        "outcome_rates": {o: len(v) / n for o, v in by_outcome.items()},
        # Crashes split by phase: "search" = target not yet found, "approach" = after detection.
        "crash_phase_counts": {
            o: {"search": sum(not r["target_found"] for r in by_outcome[o]), "approach": sum(r["target_found"] for r in by_outcome[o])}
            for o in ("obstacle", "boundary", "drone_collision")
        },
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_steps": mean("steps"),
        "mean_final_dist": mean("final_mean_dist"),
        "mean_worst_final_dist": mean("final_worst_dist"),
        "mean_drones_at_target": mean("drones_at_target"),
        "mean_path_length": mean("path_length"),
        "mean_min_obstacle_clearance": mean("min_obstacle_clearance"),
        "mean_cohesion_error": mean("mean_cohesion_error"),
        "mean_formation_error": mean("mean_formation_error"),
        "mean_terms": {k: mean(f"r_{k}") for k in REWARD_TERMS},
        "reward_by_outcome": {o: float(np.mean([r["reward"] for r in v])) for o, v in by_outcome.items() if v},
        "terms_by_outcome": {
            o: {k: float(np.mean([r[f"r_{k}"] for r in v])) for k in REWARD_TERMS}
            for o, v in by_outcome.items() if v
        },
    }


def _plot_trajectories_per_episode(results: dict, cfg: Config, path, n_cols=4):
    # Random obstacles differ per episode, so episodes cannot share one panel. One panel per episode, drawn with that
    # episode's own obstacles and target. Every policy is evaluated on the same seeds, so a column is one layout.
    names = list(results)
    n_cols = max(1, min(n_cols, min(len(rows) for rows in results.values())))
    fig, axes = plt.subplots(len(names), n_cols, figsize=(4.2 * n_cols, 4.4 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        for j in range(n_cols):
            ax, r = axes[i][j], results[name][j]
            draw_scene(ax, cfg.env, r["obstacles"], r["target"])
            traj = r["traj"]
            for d in range(traj.shape[1]):
                color = DRONE_COLORS[d % len(DRONE_COLORS)]
                ax.plot(traj[:, d, 0], traj[:, d, 1], color=color, alpha=0.8, lw=1)
                ax.scatter(*traj[0, d], color=color, s=14, zorder=5)
            if not math.isnan(r["collision_x"]):
                ax.scatter(r["collision_x"], r["collision_y"], marker="x", color="black", s=60, zorder=6)
            ax.set_title(f"{name} | seed {r['seed']} | {r['outcome']}", fontsize=9, color=OUTCOME_COLORS[r["outcome"]])
            ax.set_xlabel("")
            ax.set_ylabel("")
    fig.suptitle("Evaluation trajectories, one episode per panel (red = obstacles, star = target, dot = start, x = collision)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_trajectories(results: dict, cfg: Config, path, max_episodes=12):
    if cfg.env.random_obstacles and cfg.env.max_obstacles > 0:
        _plot_trajectories_per_episode(results, cfg, path)
        return
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6.4), squeeze=False)
    for ax, (name, rows) in zip(axes[0], results.items()):
        first_obs = rows[0]["obstacles"] if (rows and not cfg.env.random_obstacles) else []
        draw_scene(ax, cfg.env, first_obs)
        counts = dict.fromkeys(OUTCOMES, 0)
        for r in rows[:max_episodes]:
            counts[r["outcome"]] += 1
            for d in range(r["traj"].shape[1]):
                ax.plot(r["traj"][:, d, 0], r["traj"][:, d, 1], color=OUTCOME_COLORS[r["outcome"]], alpha=0.35, lw=1)
            ax.scatter(*r["target"], marker="*", color="green", s=90, zorder=5)
            if not math.isnan(r["collision_x"]):
                ax.scatter(r["collision_x"], r["collision_y"], marker="x", color="black", s=45, zorder=6)
        ax.set_title(name)
        handles = [
            plt.Line2D([], [], color=OUTCOME_COLORS[o], label=f"{o} ({c})") for o, c in counts.items() if c
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.suptitle("Evaluation trajectories (x = collision point, star = target)")
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


def record_episode(cfg: Config, policy_factory, seed):
    """Replay one episode and record, per step, what the GIF draws: positions, rays, searched map, waypoint, found flag.

    The environment and the deterministic evaluation policies are reproducible from the episode seed, so the replay is
    the same episode as the evaluated one (save_gif checks the positions). Ray ends mark where the drone centre would
    touch the obstacle, wall or drone, so they stop drone_radius short of the surface.
    """
    env = DroneSwarm2D(cfg.env, cfg.reward)
    env.action_space.seed(seed)
    policy = policy_factory(env)
    obs, _ = env.reset(seed=seed)
    start_searched = env.searched.copy()
    frames = []

    def snapshot():
        dist = env._raycast() * cfg.env.vision_range  # (n, rays) in metres
        angles = env.headings[:, None] + env.ray_offsets[None, :]
        ends = env.positions[:, None, :] + dist[..., None] * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        frames.append({
            "pos": env.positions.copy(),
            "ray_start": np.repeat(env.positions[:, None, :], len(env.ray_offsets), axis=1).reshape(-1, 2),
            "ray_end": ends.reshape(-1, 2),
            "ray_hit": (dist < cfg.env.vision_range - 1e-6).ravel(),
            "searched": env.searched.copy(),
            "waypoint": None if env.search_goal_idx is None else env.cell_centers[env.search_goal_idx].copy(),
            "found": env.target_found,
        })

    snapshot()
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        snapshot()
        if terminated or truncated:
            break
    return {"frames": frames, "start_searched": start_searched, "obstacles": env.obstacles.copy(),
            "target": env.target.copy(), "cell_centers": env.cell_centers, "outcome": info["termination"]}


def save_gif(cfg: Config, row, path, title, policy_factory, fps=15, stride=2, hold=12):
    """Animate one evaluated episode: drones at true size, their rays, the searched map, the waypoint and the target."""
    ep = record_episode(cfg, policy_factory, row["seed"])
    traj = np.stack([f["pos"] for f in ep["frames"]])
    if traj.shape != row["traj"].shape or not np.allclose(traj, row["traj"], atol=1e-5):
        print(f"Warning: replay of seed {row['seed']} differs from the evaluated episode; the GIF shows the replay.")
    c = cfg.env
    frames = list(range(0, len(traj), stride)) + [len(traj) - 1] * hold
    fig, ax = plt.subplots(figsize=(6.4, 6.9))
    draw_scene(ax, c, ep["obstacles"], None)

    # Searched map: grey = needs no search from the start (along the walls, around obstacles), blue = searched by the drones.
    grid = int(np.ceil(c.world_size / c.explore_cell))
    extent = (0, grid * c.explore_cell, 0, grid * c.explore_cell)

    def map_image(searched):
        img = np.zeros((len(searched), 4), dtype=np.float32)
        img[ep["start_searched"]] = (0.6, 0.6, 0.6, 0.25)
        img[searched & ~ep["start_searched"]] = (0.35, 0.6, 0.95, 0.25)
        return img.reshape(grid, grid, 4).transpose(1, 0, 2)  # cells are indexed [x, y]; imshow wants [row = y, col = x]

    searched_img = ax.imshow(map_image(ep["frames"][0]["searched"]), origin="lower", extent=extent, zorder=0,
                             interpolation="nearest")
    waypoint = Rectangle((0, 0), c.explore_cell, c.explore_cell, fill=False, edgecolor="darkorange", lw=2, zorder=3,
                         visible=False)
    ax.add_patch(waypoint)
    target_zone = Circle(ep["target"], c.success_radius, color="green", alpha=0.08, zorder=1)
    ax.add_patch(target_zone)
    target_star = ax.scatter(*ep["target"], color="green", marker="*", s=150, alpha=0.3, zorder=2)
    rays = LineCollection([], linewidths=0.8, zorder=4)
    ax.add_collection(rays)
    n = traj.shape[1]
    colors = [DRONE_COLORS[d % len(DRONE_COLORS)] for d in range(n)]
    trails = [ax.plot([], [], color=colors[d], alpha=0.6, lw=1.2, zorder=4)[0] for d in range(n)]
    bodies = [Circle(traj[0, d], c.drone_radius, color=colors[d], zorder=6) for d in range(n)]
    for body in bodies:
        ax.add_patch(body)
    heading = ax.set_title("", fontsize=10)
    fig.text(0.5, 0.01, "rays: red = hits something within range, grey = clear | orange square = search waypoint\n"
             "grey cells = no search needed | blue cells = searched | star = target (faint until found)",
             ha="center", fontsize=7.5)

    def update(k):
        f = frames[k]
        fr = ep["frames"][f]
        searched_img.set_data(map_image(fr["searched"]))
        if fr["waypoint"] is not None:
            waypoint.set_xy(fr["waypoint"] - c.explore_cell / 2)
        waypoint.set_visible(fr["waypoint"] is not None)
        alpha = 1.0 if fr["found"] else 0.3
        target_star.set_alpha(alpha)
        target_zone.set_alpha(0.25 if fr["found"] else 0.08)
        rays.set_segments(np.stack([fr["ray_start"], fr["ray_end"]], axis=1))
        rays.set_colors(np.where(fr["ray_hit"][:, None], [[0.85, 0.1, 0.1, 0.9]], [[0.5, 0.5, 0.5, 0.35]]))
        for d in range(n):
            bodies[d].center = traj[f, d]
            trails[d].set_data(traj[: f + 1, d, 0], traj[: f + 1, d, 1])
        phase = "approaching target" if fr["found"] else "searching"
        end = f == len(traj) - 1
        heading.set_text(f"{title} | step {f} | {phase}" + (f" | {ep['outcome'].upper()}" if end else ""))
        return []

    fig.subplots_adjust(bottom=0.1)
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


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
        "Raise reward.separation_weight / reward.separation_margin so closing in on a neighbour is penalised before contact, or raise reward.collision_penalty.",
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

    env_c = cfg.env
    if env_c.start_jitter == 0 and not (env_c.random_obstacles or env_c.random_start or env_c.random_target):
        add(
            f"No randomness in the environment: all {ppo['n']} evaluation episodes are identical, so the result is one episode repeated "
            "(0% or 100%), not an estimate of generalisation.",
            "Enable env.random_obstacles, env.random_start or env.random_target for an honest evaluation.",
        )

    if ppo["found_rate"] < 0.7:
        add(
            f"The target is found in only {ppo['found_rate']:.0%} of episodes (mean searched area {ppo['mean_searched_fraction']:.0%}): "
            "the swarm is not searching systematically.",
            "Raise reward.explore_weight, raise env.max_steps, enlarge env.target_sense_range, or allow a wider swarm (env.cohesion_radius) so it covers more ground.",
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
                    f"{ppo['mean_worst_final_dist']:.2f} from the target, success needs < {cfg.env.success_radius}): "
                    "the swarm arrives but not every drone is inside the success radius.",
                    "Check reward.cohesion_weight is not too high to let the swarm settle, raise reward.success_bonus, or loosen env.success_radius.",
                )
            else:
                add(
                    f"{rates['timeout']:.0%} of episodes time out far from the target "
                    f"(mean distance {ppo['mean_final_dist']:.1f}): the policy hovers or detours instead of progressing.",
                    "Raise reward.progress_weight, check reward.cohesion_weight is not larger than progress, or train longer.",
                )

    rbo = ppo["reward_by_outcome"]
    crash_returns = [rbo[o] for o in ("obstacle", "boundary", "drone_collision") if o in rbo]
    if "timeout" in rbo and crash_returns and max(crash_returns) > rbo["timeout"]:
        add(
            f"Crashing pays: best crash return {max(crash_returns):.1f} > timeout return {rbo['timeout']:.1f}. "
            "Per-step costs outweigh the collision penalty, so ending the episode early is attractive.",
            "Raise reward.collision_penalty above the cumulative step/cohesion cost, or lower reward.step_penalty / reward.cohesion_weight.",
        )

    terms = ppo["mean_terms"]
    if abs(terms["cohesion"]) > abs(terms["progress"]):
        add(
            f"Cohesion penalty ({terms['cohesion']:.1f}) outweighs progress reward ({terms['progress']:.1f}) per episode.",
            "Lower reward.cohesion_weight, or raise reward.progress_weight.",
        )

    greedy = summaries.get("Greedy")
    if greedy and ppo["success_rate"] <= greedy["success_rate"]:
        add(
            f"PPO ({ppo['success_rate']:.0%}) does not beat the naive searcher ({greedy['success_rate']:.0%}).",
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
              "| Policy | Success [95% CI] | Collision | Timeout | Target found | Return | Steps | Final dist | Worst drone dist | Formation error |",
              "|---|---|---|---|---|---|---|---|---|---|",]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {_fmt_rate(s)} | {s['collision_rate']:.0%} | {s['outcome_rates']['timeout']:.0%} | {s['found_rate']:.0%} | "
            f"{s['mean_reward']:.1f} ± {s['std_reward']:.1f} | {s['mean_steps']:.0f} | {s['mean_final_dist']:.2f} | {s['mean_worst_final_dist']:.2f} | {s['mean_formation_error']:.3f} |"
        )

    lines += ["", "## How episodes end", "", "| Policy | " + " | ".join(OUTCOMES) + " |", "|---|" + "---|" * len(OUTCOMES)]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(str(s["outcome_counts"][o]) for o in OUTCOMES) + " |")

    crashes = ("obstacle", "boundary", "drone_collision")
    lines += ["", "## Crashes by phase (search = before detection / approach = after)", "",
              "| Policy | " + " | ".join(crashes) + " |", "|---|" + "---|" * len(crashes)]
    for name, s in summaries.items():
        phase = s["crash_phase_counts"]
        lines.append(f"| {name} | " + " | ".join(f"{phase[o]['search']} / {phase[o]['approach']}" for o in crashes) + " |")

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