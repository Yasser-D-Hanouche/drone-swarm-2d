from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle

from config import EnvConfig, RewardConfig

REWARD_TERMS = ["progress", "exploration", "cohesion", "separation", "step", "obstacle_proximity", "collision", "success"]
OUTCOMES = ["success", "obstacle", "boundary", "drone_collision", "timeout"]


def draw_scene(ax, cfg: EnvConfig, obstacles=None, target=None):
    pad = 0.4
    ax.set_xlim(-pad, cfg.world_size + pad)
    ax.set_ylim(-pad, cfg.world_size + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.2)
    ax.add_patch(
        Rectangle(
            (0, 0),
            cfg.world_size,
            cfg.world_size,
            fill=False,
            edgecolor="black",
            linewidth=2.5,
            zorder=3,
        )
    )
    if target is not None:
        ax.add_patch(Circle(target, cfg.success_radius, color="green", alpha=0.25, zorder=1))
        ax.scatter(*target, color="green", marker="*", s=150, zorder=2)
    obs_list = cfg.obstacles if obstacles is None else obstacles
    for ox, oy, radius in obs_list:
        ax.add_patch(Circle((ox, oy), radius, color="red", alpha=0.4, zorder=2))


class DroneSwarm2D(gym.Env):
    """One centralised policy controls all drones.

    Action: (n, 2) world-frame acceleration per drone in [-1, 1] (scaled by max_accel).
    The target position is hidden. A drone senses it only within target_sense_range; once any drone has,
    the whole swarm knows it (shared). Until then the swarm must search, guided by a coarse map of the
    area already searched (cells whose centre has been within sensing range of a drone).
    Goal: find the target, then bring every drone within success_radius of it while staying together
    (within cohesion_radius of the swarm centroid) and clear of obstacles, walls and each other.
    Observation = per-drone blocks, then global items:
      per drone: pos/world_size (2) | vel/max_speed (2) | (target - pos)/world_size, zeros until found (2) |
        relative position of every other drone, world units, clipped +-3 (2*(n-1)) |
        rays (num_rays; 1.0 = nothing within vision_range). Rays point along the drone's velocity direction.
      global: target_found flag (1) | searched map, grid x grid cells of explore_cell, row-major, 1 = searched.
    The policy is memoryless: obstacles seen earlier are forgotten unless they are currently in the rays.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_cfg: EnvConfig | None = None, reward_cfg: RewardConfig | None = None):
        super().__init__()
        self.cfg = env_cfg or EnvConfig()
        self.rw = reward_cfg or RewardConfig()
        self.n = self.cfg.num_drones
        self.target = np.asarray(self.cfg.target, dtype=np.float32)
        self.obstacles = np.asarray(self.cfg.obstacles, dtype=np.float32).reshape(-1, 3)
        self.max_steps = self.cfg.max_steps

        half_fov = np.deg2rad(self.cfg.vision_angle / 2.0)
        if self.cfg.num_rays > 1:
            self.ray_offsets = np.linspace(-half_fov, half_fov, self.cfg.num_rays, dtype=np.float32)
        else:
            self.ray_offsets = np.array([0.0], dtype=np.float32)

        grid = int(np.ceil(self.cfg.world_size / self.cfg.explore_cell))
        axis = (np.arange(grid) + 0.5) * self.cfg.explore_cell
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        self.cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

        obs_dim = self.n * (6 + 2 * (self.n - 1) + self.cfg.num_rays) + 1 + len(self.cell_centers)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n, 2), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def _raycast(self):
        c = self.cfg
        readings = np.full((self.n, c.num_rays), c.vision_range, dtype=np.float32)

        for i in range(self.n):
            px, py = self.positions[i]
            angles = self.headings[i] + self.ray_offsets
            dx = np.cos(angles)
            dy = np.sin(angles)

            for r in range(c.num_rays):
                rx, ry = dx[r], dy[r]
                min_t = c.vision_range

                if rx > 1e-6:
                    t = (c.world_size - c.drone_radius - px) / rx
                    if 0.0 < t < min_t:
                        min_t = t
                elif rx < -1e-6:
                    t = (c.drone_radius - px) / rx
                    if 0.0 < t < min_t:
                        min_t = t

                if ry > 1e-6:
                    t = (c.world_size - c.drone_radius - py) / ry
                    if 0.0 < t < min_t:
                        min_t = t
                elif ry < -1e-6:
                    t = (c.drone_radius - py) / ry
                    if 0.0 < t < min_t:
                        min_t = t

                for ox, oy, rad in self.obstacles:
                    fx, fy = px - ox, py - oy
                    combined_r = rad + c.drone_radius
                    b = 2.0 * (fx * rx + fy * ry)
                    c_val = fx * fx + fy * fy - combined_r * combined_r
                    disc = b * b - 4.0 * c_val
                    if disc >= 0.0:
                        t = (-b - np.sqrt(disc)) / 2.0
                        if 0.0 < t < min_t:
                            min_t = t

                for j in range(self.n):
                    if i == j:
                        continue
                    ox, oy = self.positions[j]
                    fx, fy = px - ox, py - oy
                    combined_r = 2.0 * c.drone_radius
                    b = 2.0 * (fx * rx + fy * ry)
                    c_val = fx * fx + fy * fy - combined_r * combined_r
                    disc = b * b - 4.0 * c_val
                    if disc >= 0.0:
                        t = (-b - np.sqrt(disc)) / 2.0
                        if 0.0 < t < min_t:
                            min_t = t

                readings[i, r] = min_t

        return readings / c.vision_range

    def _cohesion_excess(self):
        # How far each drone is outside cohesion_radius from the swarm centroid.
        dist = np.linalg.norm(self.positions - self.positions.mean(axis=0), axis=1)
        return np.maximum(dist - self.cfg.cohesion_radius, 0.0)

    def _update_search(self):
        # Returns the number of cells searched for the first time this step; may flip target_found.
        c = self.cfg
        d = np.linalg.norm(self.positions[:, None, :] - self.cell_centers[None, :, :], axis=2)
        seen = (d <= c.target_sense_range).any(axis=0)
        newly = int((seen & ~self.searched).sum())
        self.searched |= seen
        if np.linalg.norm(self.positions - self.target, axis=1).min() <= c.target_sense_range:
            self.target_found = True
        return newly

    def _observation(self):
        c = self.cfg
        rays = self._raycast()
        if self.target_found:
            to_target = (self.target - self.positions) / c.world_size
        else:
            to_target = np.zeros((self.n, 2), dtype=np.float32)
        rel = self.positions[None, :, :] - self.positions[:, None, :]
        rel = rel[~np.eye(self.n, dtype=bool)].reshape(self.n, -1)
        per_drone = np.concatenate(
            [
                self.positions / c.world_size,
                self.velocities / c.max_speed,
                to_target,
                np.clip(rel, -3.0, 3.0),
                rays,
            ],
            axis=1,
        )
        global_part = np.concatenate([[float(self.target_found)], self.searched.astype(np.float32)])
        return np.concatenate([per_drone.flatten(), global_part]).astype(np.float32)

    def _generate_obstacles(self):
        c = self.cfg
        if not c.random_obstacles:
            return np.asarray(c.obstacles, dtype=np.float32).reshape(-1, 3)

        count = int(self.np_random.integers(c.min_obstacles, c.max_obstacles + 1))
        generated = []
        for _ in range(count):
            for _ in range(200):
                radius = float(self.np_random.uniform(c.min_obstacle_radius, c.max_obstacle_radius))
                pos = self.np_random.uniform(radius, c.world_size - radius, size=2).astype(np.float32)

                dist_drones = np.linalg.norm(self.positions - pos, axis=1)
                if np.any(dist_drones < radius + c.drone_radius + c.obstacle_safe_margin):
                    continue

                dist_target = float(np.linalg.norm(self.target - pos))
                if dist_target < radius + c.success_radius + c.obstacle_safe_margin:
                    continue

                generated.append([pos[0], pos[1], radius])
                break

        if not generated:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(generated, dtype=np.float32)

    def _sample_start_and_target(self):
        c = self.cfg
        lo, hi = c.edge_margin, c.world_size - c.edge_margin
        for _ in range(1000):
            if c.random_start:
                center = self.np_random.uniform(lo, hi, size=2)
            else:
                center = np.asarray(c.start_center, dtype=np.float64)
                if c.start_jitter > 0:
                    center = center + self.np_random.uniform(-c.start_jitter, c.start_jitter, size=2)
            target = self.np_random.uniform(lo, hi, size=2) if c.random_target else np.asarray(c.target, dtype=np.float64)
            if not (c.random_start or c.random_target) or np.linalg.norm(target - center) >= c.min_start_target_distance:
                return center.astype(np.float32), target.astype(np.float32)
        raise RuntimeError("Could not sample a start/target pair; lower env.min_start_target_distance")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        center, self.target = self._sample_start_and_target()
        # Drones start evenly spaced on a circle of radius start_spread, with a random rotation.
        phase = float(self.np_random.uniform(0.0, 2 * np.pi))
        angles = phase + 2 * np.pi * np.arange(self.n) / self.n
        c_spread = self.cfg.start_spread * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        self.positions = (center + c_spread).astype(np.float32)
        self.velocities = np.zeros((self.n, 2), dtype=np.float32)
        self.headings = np.full(self.n, self.np_random.uniform(-np.pi, np.pi), dtype=np.float32)
        self.obstacles = self._generate_obstacles()
        self.steps = 0
        self.searched = np.zeros(len(self.cell_centers), dtype=bool)
        self.target_found = False
        self._update_search()
        return self._observation(), {}

    def _geometry(self):
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

        moving = np.linalg.norm(self.velocities, axis=1) > 0.05
        if np.any(moving):
            self.headings[moving] = np.arctan2(self.velocities[moving, 1], self.velocities[moving, 0])

        self.steps += 1

        was_found = self.target_found
        newly_searched = self._update_search()
        target_dist, pair, nearest, obs_clear, wall_clear = self._geometry()
        progress = float(np.mean(old_dist - target_dist)) if self.target_found else 0.0
        cohesion_excess = self._cohesion_excess()
        cohesion_error = float(cohesion_excess.mean())
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
        terms["cohesion"] = -w.cohesion_weight * cohesion_error
        terms["step"] = -w.step_penalty
        if not was_found:
            terms["exploration"] = w.explore_weight * newly_searched
        if w.separation_margin > 0:
            gap = nearest - 2 * c.drone_radius
            close = np.maximum(0.0, w.separation_margin - gap) / w.separation_margin
            terms["separation"] = -w.separation_weight * float(np.mean(close))
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
            "cohesion_error": cohesion_error,
            "target_found": self.target_found,
            "searched_fraction": float(self.searched.mean()),
            "mean_target_distance": float(np.mean(target_dist)),
            "max_target_distance": float(np.max(target_dist)),
            "min_obstacle_clearance": float(obs_clear.min()),
            "min_pair_distance": min_pair,
            "reward_terms": terms,
        }
        return self._observation(), reward, terminated, truncated, info



