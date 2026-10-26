
import numpy as np
import matplotlib

# Save the animation without requiring a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO

from environment import (
    DroneSwarm2D,
    NUM_DRONES,
    WORLD_SIZE,
    TARGET,
    OBSTACLES,
    DRONE_RADIUS,
)

# Configuration
MODEL_PATH = "ppo_drone_swarm"
OUTPUT_PATH = "trained_swarm.gif"
MAX_FRAMES = 500
FPS = 10

# Create environment and load trained policy
env = DroneSwarm2D()
obs, info = env.reset(seed=42)
model = PPO.load(MODEL_PATH)

# Figure and axes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, WORLD_SIZE)
ax.set_ylim(0, WORLD_SIZE)
ax.set_aspect("equal")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.grid(True, alpha=0.3)

# Target
target = np.asarray(TARGET, dtype=float)
ax.scatter(
    target[0],
    target[1],
    marker="*",
    s=250,
    color="green",
    label="Target",
    zorder=5,
)

# Obstacles
for ox, oy, radius in OBSTACLES:
    obstacle = plt.Circle(
        (ox, oy),
        radius,
        color="red",
        alpha=0.5,
        zorder=2,
    )
    ax.add_patch(obstacle)

# Drones and trails
colors = ["tab:blue", "tab:orange", "tab:purple"]
drones = []
trails = []

trail_x = [[] for _ in range(NUM_DRONES)]
trail_y = [[] for _ in range(NUM_DRONES)]

for i in range(NUM_DRONES):
    drone, = ax.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        color=colors[i],
        markersize=9,
        label=f"Drone {i + 1}",
        zorder=4,
    )

    trail, = ax.plot(
        [],
        [],
        color=colors[i],
        alpha=0.7,
        linewidth=1.5,
        zorder=3,
    )

    drones.append(drone)
    trails.append(trail)

ax.legend(loc="upper left")
title = ax.set_title("Trained PPO Drone Swarm")

# Animation state
done = False
step_count = 0
final_message = None


def update(frame):
    global obs, done, step_count, final_message

    # Advance one step using the trained policy.
    if not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated

    positions = np.asarray(env.positions)

    # Update drone positions and movement trails.
    for i in range(NUM_DRONES):
        x, y = positions[i]

        drones[i].set_data([x], [y])

        trail_x[i].append(x)
        trail_y[i].append(y)

        trails[i].set_data(trail_x[i], trail_y[i])

    title.set_text(f"Trained PPO Drone Swarm — Step {step_count}")

    # Display the final outcome once the episode ends.
    if done and final_message is None:
        distances = np.linalg.norm(positions - target, axis=1)
        success = np.all(distances < 0.5)

        if success:
            message = "SUCCESS: All drones reached the target"
            message_color = "green"
        else:
            message = "Episode ended without reaching the target"
            message_color = "darkred"

        final_message = ax.text(
            0.5,
            0.97,
            message,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color=message_color,
            fontweight="bold",
        )

    return drones + trails


animation = FuncAnimation(
    fig,
    update,
    frames=MAX_FRAMES,
    interval=1000 / FPS,
    blit=False,
    repeat=False,
)

# Save animation as a GIF.
animation.save(
    OUTPUT_PATH,
    writer=PillowWriter(fps=FPS),
)

plt.close(fig)

print(f"Animation saved to {OUTPUT_PATH}")
print(f"Steps simulated: {step_count}")
