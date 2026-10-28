from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from config import EnvConfig, RewardConfig

REWARD_TERMS = ["progress", "formation", "step", "obstacle_proximity", "collision", "success"]
OUTCOMES = ["success", "obstacle", "boundary", "drone_collision", "timeout"]

# Defaults kept as module constants so the older standalone scripts keep working.
_DEFAULT = EnvConfig()
WORLD_SIZE = _DEFAULT.world_size
DRONE_RADIUS = _DEFAULT.drone_radius
SUCCESS_RADIUS = _DEFAULT.success_radius
TARGET = np.array(_DEFAULT.target, dtype=np.float32)
FORMATION = np.array(_DEFAULT.formation, dtype=np.float32)
NUM_DRONES = len(_DEFAULT.formation)
OBSTACLES = [tuple(o) for o in _DEFAULT.obstacles]


def draw_scene(ax, cfg: EnvConfig):
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.2)
    ax.add_patch(Circle(cfg.target, cfg.success_radius, color="green", alpha=0.25, zorder=1))
    ax.scatter(*cfg.target, color="green", marker="*", s=150, zorder=2)
    for ox, oy, radius in cfg.obstacles:
        ax.add_patch(Circle((ox, oy), radius, color="red", alpha=0.4, zorder=2))


class DroneSwarm2D(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_cfg: EnvConfig | None = None, reward_cfg: RewardConfig | None = None):
        super().__init__()
        self.cfg = env_cfg or EnvConfig()
        self.rw = reward_cfg or RewardConfig()
        self.n = len(self.cfg.formation)
        self.target = np.asarray(self.cfg.target, dtype=np.float32)
        self.formation = np.asarray(self.cfg.formation, dtype=np.float32)
        self.obstacles = np.asarray(self.cfg.obstacles, dtype=np.float32).reshape(-1, 3)
        self.max_steps = self.cfg.max_steps

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n, 2), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n * 4,), dtype=np.float32)
        self.reset()

    def _observation(self):
        return np.concatenate([self.positions, self.velocities], axis=1).astype(np.float32).flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        center = np.asarray(self.cfg.start_center, dtype=np.float32)
        if self.cfg.start_jitter > 0:
            center = center + self.np_random.uniform(
                -self.cfg.start_jitter, self.cfg.start_jitter, size=2
            ).astype(np.float32)
        self.positions = center + self.formation
        self.velocities = np.zeros((self.n, 2), dtype=np.float32)
        self.steps = 0
        return self._observation(), {}

    def _geometry(self):
        """Distances/clearances for the current positions (negative clearance = overlap)."""
        c = self.cfg
        pos = self.positions
        target_dist = np.linalg.norm(pos - self.target, axis=1)

        pair = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(pair, np.inf)
        nearest = pair.min(axis=1)

        if len(self.obstacles):
            d = np.linalg.norm(pos[:, None, :] - self.obstacles[None, :, :2], axis=2)
            obs_clear = (d - self.obstacles[None, :, 2] - c.drone_radius).min(axis=1)
        else:
            obs_clear = np.full(self.n, np.inf)

        wall_clear = np.minimum(pos - c.drone_radius, c.world_size - c.drone_radius - pos).min(axis=1)
        return target_dist, pair, nearest, obs_clear, wall_clear

    def step(self, actions):
        c, w = self.cfg, self.rw
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)

        old_dist = np.linalg.norm(self.positions - self.target, axis=1)

        self.velocities += actions * c.max_accel * c.dt
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities *= np.minimum(1.0, c.max_speed / np.maximum(speeds, 1e-8))
        self.positions += self.velocities * c.dt
        self.steps += 1

        target_dist, pair, nearest, obs_clear, wall_clear = self._geometry()
        progress = float(np.mean(old_dist - target_dist))
        formation_error = float(np.mean(np.maximum(nearest - c.cohesion_distance, 0.0)))
        min_pair = float(pair.min())

        crash = None
        collision_pos = None
        if obs_clear.min() < 0:
            crash = "obstacle"
            collision_pos = self.positions[int(obs_clear.argmin())]
        elif wall_clear.min() < 0:
            crash = "boundary"
            collision_pos = self.positions[int(wall_clear.argmin())]
        elif min_pair < 2 * c.drone_radius:
            crash = "drone_collision"
            i, j = np.unravel_index(int(pair.argmin()), pair.shape)
            collision_pos = (self.positions[i] + self.positions[j]) / 2

        collision = crash is not None
        success = bool(np.all(target_dist < c.success_radius)) and not collision

        terms = dict.fromkeys(REWARD_TERMS, 0.0)
        terms["progress"] = w.progress_weight * progress
        terms["formation"] = -w.formation_weight * formation_error
        terms["step"] = -w.step_penalty
        if w.obstacle_margin > 0:
            proximity = np.maximum(0.0, w.obstacle_margin - obs_clear) / w.obstacle_margin
            terms["obstacle_proximity"] = -w.obstacle_weight * float(np.mean(proximity))
        if collision:
            terms["collision"] = -w.collision_penalty
        if success:
            terms["success"] = w.success_bonus
        reward = float(sum(terms.values()))

        terminated = collision or success
        truncated = self.steps >= self.max_steps and not terminated

        info = {
            "success": success,
            "collision": collision,
            "termination": crash or ("success" if success else ("timeout" if truncated else "running")),
            "collision_pos": None if collision_pos is None else [float(collision_pos[0]), float(collision_pos[1])],
            "formation_error": formation_error,
            "mean_target_distance": float(np.mean(target_dist)),
            "max_target_distance": float(np.max(target_dist)),
            "min_obstacle_clearance": float(obs_clear.min()),
            "min_pair_distance": min_pair,
            "reward_terms": terms,
        }
        return self._observation(), reward, terminated, truncated, info

    def save_plot(self, filename="swarm.png"):
        fig, ax = plt.subplots(figsize=(7, 7))
        draw_scene(ax, self.cfg)
        ax.set_title(f"2D Drone Swarm Step {self.steps}")
        for i, position in enumerate(self.positions):
            ax.add_patch(Circle(position, self.cfg.drone_radius, color="blue"))
            ax.text(position[0] + 0.1, position[1] + 0.1, str(i))
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    env = DroneSwarm2D()
    obs, _ = env.reset(seed=42)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    env.save_plot()
    print("Observation shape:", obs.shape, "| Action shape:", env.action_space.shape)
    print("Steps:", env.steps, "| Termination:", info["termination"])
    print("Reward terms:", info["reward_terms"])
