
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from environment import (
    DroneSwarm2D, WORLD_SIZE, NUM_DRONES,
    DRONE_RADIUS, TARGET, OBSTACLES
)

env = DroneSwarm2D()
env.reset(seed=42)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, WORLD_SIZE)
ax.set_ylim(0, WORLD_SIZE)
ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, alpha=0.2)

ax.add_patch(Circle(TARGET, 0.5, color="green", alpha=0.3))
ax.scatter(*TARGET, color="green", marker="*", s=150)

for ox, oy, radius in OBSTACLES:
    ax.add_patch(Circle((ox, oy), radius, color="red", alpha=0.5))

drones = [
    Circle(env.positions[i], DRONE_RADIUS, color="blue")
    for i in range(NUM_DRONES)
]
for drone in drones:
    ax.add_patch(drone)

labels = [
    ax.text(*env.positions[i], str(i), color="white",
            ha="center", va="center")
    for i in range(NUM_DRONES)
]

title = ax.set_title("Random drone actions")
trajectory = [[] for _ in range(NUM_DRONES)]
lines = [
    ax.plot([], [], "--", alpha=0.5)[0]
    for _ in range(NUM_DRONES)
]

state = {"done": False, "step": 0}

def update(frame):
    if state["done"]:
        return drones + labels + lines + [title]

    actions = env.np_random.uniform(-1, 1, size=(NUM_DRONES, 2))
    _, _, terminated, truncated, info = env.step(actions)

    for i, position in enumerate(env.positions):
        drones[i].center = position
        labels[i].set_position(position)
        trajectory[i].append(position.copy())
        points = np.array(trajectory[i])
        lines[i].set_data(points[:, 0], points[:, 1])

    state["step"] += 1
    title.set_text(
        f"Random actions | Step {state['step']} | "
        f"Collision: {info['collision']} | "
        f"Success: {info['success']}"
    )

    if terminated or truncated:
        state["done"] = True

    return drones + labels + lines + [title]

animation = FuncAnimation(
    fig, update, frames=200, interval=100, blit=False
)
animation.save("swarm.gif", writer=PillowWriter(fps=10))
plt.close(fig)

print("Animation saved to swarm.gif")
print("Open swarm.gif in VS Code to watch the drones move.")
