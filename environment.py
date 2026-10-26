import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SUCCESS_RADIUS = 0.5
WORLD_SIZE = 10.0
NUM_DRONES = 3
DRONE_RADIUS = 0.15
TARGET = np.array([8.5, 8.5], dtype=np.float32)
FORMATION = np.array([
    [-0.5, -0.4],
    [0.5, -0.4],
    [0.0, 0.5],
], dtype=np.float32)
OBSTACLES = [
    (3.0, 3.0, 0.7),
    (5.0, 5.5, 0.9),
    (7.0, 3.5, 0.6),
]

class DroneSwarm2D(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=500):
        super().__init__()
        self.max_steps = max_steps
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_DRONES, 2),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(NUM_DRONES * 4,),
            dtype=np.float32,
        )
        self.reset()

    def _observation(self):
        return np.concatenate(
            [self.positions, self.velocities], axis=1
        ).astype(np.float32).flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        center = np.array([1.0, 1.0], dtype=np.float32)
        self.positions = center + FORMATION.copy()
        self.velocities = np.zeros((NUM_DRONES, 2), dtype=np.float32)
        self.steps = 0
        return self._observation(), {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, -1.0, 1.0)

        old_dist = np.linalg.norm(self.positions - TARGET, axis=1)

        dt = 0.1
        self.velocities += actions * dt
        speeds = np.linalg.norm(self.velocities, axis=1)
        scale = np.minimum(1.0, 1.0 / np.maximum(speeds, 1e-8))
        self.velocities *= scale[:, None]
        self.positions += self.velocities * dt
        self.steps += 1

        distances = np.linalg.norm(self.positions - TARGET, axis=1)
        progress = np.mean(old_dist - distances)

        center = self.positions.mean(axis=0)
        pairwise_distances = np.linalg.norm(
            self.positions[:, None, :] - self.positions[None, :, :],
            axis=2,
        )

        # Exclude each drone's distance to itself.
        np.fill_diagonal(pairwise_distances, np.inf)

        nearest_distances = np.min(pairwise_distances, axis=1)

        formation_error = np.mean(
            np.maximum(nearest_distances - 0.5, 0.0)
        )

        boundary_collision = np.any(
            (self.positions < DRONE_RADIUS)
            | (self.positions > WORLD_SIZE - DRONE_RADIUS)
        )

        obstacle_collision = False
        for ox, oy, radius in OBSTACLES:
            d = np.linalg.norm(
                self.positions - np.array([ox, oy]), axis=1
            )
            if np.any(d < radius + DRONE_RADIUS):
                obstacle_collision = True

        pair_collision = any(
            np.linalg.norm(self.positions[i] - self.positions[j])
            < 2 * DRONE_RADIUS
            for i in range(NUM_DRONES)
            for j in range(i + 1, NUM_DRONES)
        )

        distances = np.linalg.norm(self.positions - TARGET, axis=1)
        success = bool(np.all(distances < SUCCESS_RADIUS))
        collision = bool(
            boundary_collision or obstacle_collision or pair_collision
        )

        reward = (
            5.0 * progress
            - 1 * formation_error
            - 0.01
        )

        if collision:
            reward -= 10.0
        if success:
            reward += 20.0

        terminated = collision or success
        truncated = self.steps >= self.max_steps

        info = {
            "success": success,
            "collision": collision,
            "formation_error": float(formation_error),
            "mean_target_distance": float(np.mean(distances)),
        }

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def save_plot(self, filename="swarm.png"):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect("equal")
        ax.set_title(f"2D Drone Swarm Step {self.steps}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.2)

        ax.add_patch(Circle(TARGET, 0.5, color="green", alpha=0.3))
        ax.scatter(*TARGET, color="green", marker="*", s=150)

        for ox, oy, radius in OBSTACLES:
            ax.add_patch(Circle((ox, oy), radius, color="red", alpha=0.5))

        for i, position in enumerate(self.positions):
            ax.add_patch(Circle(position, DRONE_RADIUS, color="blue"))
            ax.text(position[0] + 0.1, position[1] + 0.1, str(i))

        center = self.positions.mean(axis=0)
        desired = center + FORMATION
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            ax.plot(
                [desired[i, 0], desired[j, 0]],
                [desired[i, 1], desired[j, 1]],
                "--", color="gray",
            )

        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    env = DroneSwarm2D()
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.save_plot()
    print("Observation shape:", obs.shape)
    print("Action shape:", env.action_space.shape)
    print("Steps:", env.steps)
    print("Episode info:", info)
    print("Image saved to swarm.png")