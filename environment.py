from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle

from config import EnvConfig, RewardConfig

REWARD_TERMS = [
    "progress",
    "exploration",
    "search_waypoint",
    "cohesion",
    "formation",
    "separation",
    "step",
    "target_found",
    "obstacle_proximity",
    "wall_proximity",
    "collision",
    "success"
]
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
    """One policy, with weights shared across drones, controls all drones.

    Action: (n, 2) acceleration per drone in [-1, 1] (scaled by max_accel), expressed in the drone's own frame (see below).
    The target position is hidden. A drone senses it only within target_sense_range; once any drone has,
    the whole swarm knows it (shared). Until then the swarm must search, guided by a coarse map of the
    area already searched (cells whose centre has been within sensing range of a drone).
    Goal: find the target, then bring every drone within success_radius of it while staying together
    (within cohesion_radius of the swarm centroid) and clear of obstacles, walls and each other.
    Drone frame: each drone's axes are rotated so +x points from that drone to the target, fixed at the moment the
    target is found (world axes until then). Observed vectors and actions use it, so one learned behaviour covers every direction.
    Observation = (n, obs_dim): one row per drone, the same layout for every drone:
      vel/max_speed (2) | (target - pos)/world_size, zeros until found (2) |
        relative position of every other drone, nearest first, clipped +-3 world units then divided by 3 (2*(n-1)) |
        heading relative to the frame as cos, sin (2) |
        rays (num_rays; 1.0 = nothing within vision_range), pointing along the drone's velocity direction |
        distance to the wall along +x, -x, +y, -y of the frame, divided by vision_range (4) |
        target_found flag (1) | pos/world_size*2-1, zeros once found (2) |
        searched map, grid x grid cells of explore_cell, row-major, 1 = searched, zeros once found.
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

        obs_dim = 15 + 2 * (self.n - 1) + self.cfg.num_rays + len(self.cell_centers)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n, 2), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n, obs_dim), dtype=np.float32)
        self.frame_angle = np.zeros(self.n, dtype=np.float32)
        self.reset()

    def _raycast(self):
        # Vectorised over drones, rays and circles; same geometry as the original per-ray loops.
        c = self.cfg
        pos = self.positions.astype(np.float64)
        angles = self.headings.astype(np.float64)[:, None] + self.ray_offsets[None, :]
        d = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (n, R, 2)
        min_t = np.full(angles.shape, c.vision_range, dtype=np.float64)

        for axis in range(2):
            da = d[..., axis]
            pa = pos[:, axis:axis + 1]
            safe = np.where(np.abs(da) > 1e-6, da, 1.0)
            wall = np.where(da > 1e-6, c.world_size - c.drone_radius, c.drone_radius)
            t = np.where(np.abs(da) > 1e-6, (wall - pa) / safe, np.inf)
            min_t = np.where((t > 0.0) & (t < min_t), t, min_t)

        def circle_hits(centers, combined_r, exclude_self=False):
            f = pos[:, None, :] - centers[None, :, :]  # (n, M, 2)
            b = 2.0 * np.einsum("nrk,nmk->nrm", d, f)
            c_val = (f * f).sum(-1) - combined_r * combined_r  # (n, M)
            disc = b * b - 4.0 * c_val[:, None, :]
            t = (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0
            valid = (disc >= 0.0) & (t > 0.0)
            if exclude_self:
                valid &= ~np.eye(self.n, dtype=bool)[:, None, :]
            return np.where(valid, t, np.inf).min(axis=2)

        if len(self.obstacles):
            obs = self.obstacles.astype(np.float64)
            min_t = np.minimum(min_t, circle_hits(obs[:, :2], obs[None, :, 2] + c.drone_radius))
        min_t = np.minimum(min_t, circle_hits(pos, 2.0 * c.drone_radius, exclude_self=True))

        return (min_t / c.vision_range).astype(np.float32)

    def _cohesion_excess(self):
        # How far each drone is outside cohesion_radius from the swarm centroid.
        dist = np.linalg.norm(self.positions - self.positions.mean(axis=0), axis=1)
        return np.maximum(dist - self.cfg.cohesion_radius, 0.0)

    def _formation_errors(self):
        """
        Per drone: mean absolute error between its distances to the other drones
        and the distances of the initial regular polygon formation.

        The formation may translate and rotate freely.
        """
        n = self.n
        current = np.linalg.norm(self.positions[:, None, :] - self.positions[None, :, :], axis=2)
        index_distance = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        index_distance = np.minimum(index_distance, n - index_distance)
        desired = 2.0 * self.cfg.start_spread * np.sin(np.pi * index_distance / n)
        error = np.abs(current - desired)
        np.fill_diagonal(error, 0.0)
        return error.sum(axis=1) / (n - 1)

    def _update_search(self):
        # Returns per drone the number of cells it searched for the first time (shared if several drones are first); may flip target_found.
        c = self.cfg
        d = np.linalg.norm(self.positions[:, None, :] - self.cell_centers[None, :, :], axis=2)
        seen = d <= c.target_sense_range
        fresh = seen & ~self.searched[None, :]
        newly = (fresh / np.maximum(fresh.sum(axis=0, keepdims=True), 1)).sum(axis=1)
        self.searched |= seen.any(axis=0)
        if not self.target_found and np.linalg.norm(self.positions - self.target, axis=1).min() <= c.target_sense_range:
            self.target_found = True
            to = self.target - self.positions
            self.frame_angle = np.arctan2(to[:, 1], to[:, 0]).astype(np.float32)
        return newly

    def _refresh_search_goal(self):
        """Keep the current waypoint until its cell is searched."""
        if self.target_found:
            self.search_goal_idx = None
            return

        # Keep the active waypoint while its cell remains unsearched.
        if (
            self.search_goal_idx is not None
            and not self.searched[self.search_goal_idx]
        ):
            return

        unsearched = np.flatnonzero(~self.searched)

        if len(unsearched) == 0:
            self.search_goal_idx = None
            return

        centroid = self.positions.mean(axis=0)

        distances = np.linalg.norm(
            self.cell_centers[unsearched] - centroid,
            axis=1
        )

        self.search_goal_idx = int(
            unsearched[np.argmin(distances)]
        )

    def _rot(self):
        # (n, 2, 2): per-drone rotation from its own frame to world axes.
        c, s = np.cos(self.frame_angle), np.sin(self.frame_angle)
        return np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1).astype(np.float32)

    def to_frame(self, v):
        # World-frame row vectors (one per drone) -> each drone's own frame.
        return np.einsum("ni,nij->nj", np.asarray(v, dtype=np.float32), self._rot())

    def from_frame(self, v):
        return np.einsum("nj,nij->ni", np.asarray(v, dtype=np.float32), self._rot())

    def _wall_distances(self):
        # Distance to the wall along +x, -x, +y, -y of each drone's frame, normalised by vision_range.
        c = self.cfg
        rot = self._rot().astype(np.float64)
        axes = np.stack([rot[:, :, 0], -rot[:, :, 0], rot[:, :, 1], -rot[:, :, 1]], axis=1)  # (n, 4, 2) in world coordinates
        pos = self.positions.astype(np.float64)[:, None, :]
        moving = np.abs(axes) > 1e-6
        bound = np.where(axes > 0, c.world_size - c.drone_radius, c.drone_radius)
        t = np.where(moving, (bound - pos) / np.where(moving, axes, 1.0), np.inf)  # (n, 4, 2)
        return (np.clip(t.min(axis=2), 0.0, c.vision_range) / c.vision_range).astype(np.float32)

    def _observation(self):
        c, n = self.cfg, self.n
        rot = self._rot()
        found = float(self.target_found)

        if self.target_found:
            to_target = self.to_frame((self.target - self.positions) / c.world_size)
        else:
            to_target = np.zeros((n, 2), dtype=np.float32)
        
        if self.target_found or self.search_goal_idx is None:
            search_goal_vector = np.zeros(
                (n, 2), dtype=np.float32
            )
        else:
            search_goal = self.cell_centers[self.search_goal_idx]

            search_goal_vector = self.to_frame(
                (search_goal - self.positions) / c.world_size
            )

        diff = self.positions[None, :, :] - self.positions[:, None, :]  # [i, j] = pos_j - pos_i
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1)[:, : n - 1]
        rel = np.take_along_axis(diff, order[:, :, None], axis=1)
        rel = np.einsum("nmi,nij->nmj", np.clip(rel, -3.0, 3.0) / 3.0, rot).reshape(n, -1)

        heading = self.headings - self.frame_angle
        if self.target_found:
            own_pos = np.zeros((n, 2), dtype=np.float32)
            searched = np.zeros((n, len(self.cell_centers)), dtype=np.float32)
        else:
            own_pos = 2.0 * self.positions / c.world_size - 1.0
            searched = np.tile(self.searched.astype(np.float32), (n, 1))

        per_drone = np.concatenate(
            [
                self.to_frame(self.velocities / c.max_speed),
                to_target,
                search_goal_vector,
                rel,
                np.stack([np.cos(heading), np.sin(heading)], axis=1),
                self._raycast(),
                self._wall_distances(),
                np.full((n, 1), found),
                own_pos,
                searched,
            ],
            axis=1,
        )
        return per_drone.astype(np.float32)

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
        # Drones start evenly spaced on a circle of radius start_spread; rotated randomly only when the start is random,
        # so a fixed start gives every drone the same slot each episode.
        phase = float(self.np_random.uniform(0.0, 2 * np.pi)) if self.cfg.random_start else 0.0
        angles = phase + 2 * np.pi * np.arange(self.n) / self.n
        c_spread = self.cfg.start_spread * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        self.positions = (center + c_spread).astype(np.float32)
        self.velocities = np.zeros((self.n, 2), dtype=np.float32)
        self.headings = np.full(self.n, self.np_random.uniform(-np.pi, np.pi), dtype=np.float32)
        self.obstacles = self._generate_obstacles()
        self.steps = 0
        self.searched = np.zeros(len(self.cell_centers), dtype=bool)
        self.target_found = False
        self.frame_angle = np.zeros(self.n, dtype=np.float32)

        self.search_goal_idx = None

        self._update_search()
        self._refresh_search_goal()
        if self.target_found:
            # Rays follow the heading, so point it at the known target instead of a random direction.
            to_target = self.target - self.positions
            self.headings = np.arctan2(to_target[:, 1], to_target[:, 0]).astype(np.float32)
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
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.n, 2):
            raise ValueError(
                f"Expected actions of shape {(self.n, 2)}, "
                f"received {actions.shape}"
            )

        # Actions are in each drone's own frame; convert to world axes before the optional coupling.
        actions = self.from_frame(np.clip(actions, -1.0, 1.0))
        shared_action = actions.mean(axis=0, keepdims=True)
        actions = shared_action + c.action_residual_scale * (actions - shared_action)

        old_dist = np.linalg.norm(self.positions - self.target, axis=1)
        # Store the active waypoint before moving.
        old_search_goal = None
        old_search_goal_distance = None

        if not self.target_found and self.search_goal_idx is not None:
            old_search_goal = self.cell_centers[self.search_goal_idx].copy()

            old_centroid = self.positions.mean(axis=0)

            old_search_goal_distance = float(
                np.linalg.norm(old_centroid - old_search_goal)
            )

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
        # Reward movement towards the active search waypoint.
        search_waypoint_progress = 0.0

        if not was_found and old_search_goal is not None:
            new_centroid = self.positions.mean(axis=0)

            new_distance = np.linalg.norm(
                new_centroid - old_search_goal
            )

            search_waypoint_progress = (
                old_search_goal_distance - new_distance
            )
        just_found = not was_found and self.target_found
        target_dist, pair, nearest, obs_clear, wall_clear = self._geometry()
        # Success needs every drone at the target, so weight the farthest drone's progress as well as the mean.
        if self.target_found:
            progress = old_dist - target_dist
        else:
            progress = np.zeros(self.n)
        cohesion_excess = self._cohesion_excess()
        cohesion_error = float(cohesion_excess.mean())

        formation_errors = self._formation_errors()
        formation_error = float(formation_errors.mean())

        min_pair = float(pair.min())

        crash = None
        collision_pos = None
        culprits = np.zeros(self.n, dtype=bool)  # drones that caused the crash; they get the collision penalty
        if obs_clear.min() < 0:
            crash = "obstacle"
            culprits = obs_clear < 0
            collision_pos = self.positions[int(obs_clear.argmin())]
        elif wall_clear.min() < 0:
            crash = "boundary"
            culprits = wall_clear < 0
            collision_pos = self.positions[int(wall_clear.argmin())]
        elif min_pair < 2 * c.drone_radius:
            crash = "drone_collision"
            culprits = nearest < 2 * c.drone_radius
            i, j = np.unravel_index(int(pair.argmin()), pair.shape)
            collision_pos = (self.positions[i] + self.positions[j]) / 2

        collision = crash is not None
        success = bool(np.all(target_dist < c.success_radius)) and not collision

        # Every term is a vector with one value per drone, so each drone is rewarded for its own behaviour.
        terms = {k: np.zeros(self.n) for k in REWARD_TERMS}
        terms["progress"] = w.progress_weight * progress
        terms["cohesion"] = -w.cohesion_weight * cohesion_excess
        terms["formation"] = -w.formation_weight * formation_errors
        terms["step"] = np.full(self.n, -w.step_penalty)
        if not was_found:
            terms["exploration"] = w.explore_weight * newly_searched

            terms["search_waypoint"] = np.full(
                self.n,
                w.search_waypoint_weight * search_waypoint_progress,
                dtype=np.float32
            )
        if just_found:
            terms["target_found"] = np.full(
            self.n,
            w.target_found_bonus,
            dtype=np.float32
            )
        if w.separation_margin > 0:
            gap = nearest - 2 * c.drone_radius
            close = np.maximum(0.0, w.separation_margin - gap) / w.separation_margin
            terms["separation"] = -w.separation_weight * close
        if w.obstacle_margin > 0:
            proximity = np.maximum(0.0, w.obstacle_margin - obs_clear) / w.obstacle_margin
            terms["obstacle_proximity"] = -w.obstacle_weight * proximity
        if w.wall_margin > 0:
            wall_proximity = np.maximum(0.0, w.wall_margin - wall_clear) / w.wall_margin
            terms["wall_proximity"] = -w.wall_weight * wall_proximity
        if collision:
            terms["collision"] = np.full(
                self.n,
                -w.collision_penalty,
                dtype=np.float32
            )
        if success:
            terms["success"] = np.full(self.n, w.success_bonus)  # all drones are at the target, so all share the bonus
        self._refresh_search_goal()
        drone_rewards = np.sum(list(terms.values()), axis=0)
        reward = float(drone_rewards.mean())

        terminated = collision or success
        truncated = self.steps >= self.max_steps and not terminated

        info = {
            "success": success,
            "collision": collision,
            "formation_error": formation_error,
            "termination": crash or ("success" if success else ("timeout" if truncated else "running")),
            "collision_pos": None if collision_pos is None else [float(collision_pos[0]), float(collision_pos[1])],
            "cohesion_error": cohesion_error,
            "target_found": self.target_found,
            "searched_fraction": float(self.searched.mean()),
            "mean_target_distance": float(np.mean(target_dist)),
            "max_target_distance": float(np.max(target_dist)),
            "min_obstacle_clearance": float(obs_clear.min()),
            "min_pair_distance": min_pair,
            "reward_terms": {k: float(v.mean()) for k, v in terms.items()},
            "drone_rewards": drone_rewards.astype(np.float32),
        }
        return self._observation(), reward, terminated, truncated, info



