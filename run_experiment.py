import argparse
import csv
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import DroneSwarm2D, NUM_DRONES, OBSTACLES

WORLD_SIZE = 10.0
TARGET = np.array([8.5, 8.5], dtype=float)
MAX_EPISODE_STEPS = 500

LEARNING_RATE = 3e-4
N_STEPS = 1024
BATCH_SIZE = 64
GAMMA = 0.99


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def create_run_directory(name):
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name
    )
    run_dir = Path("runs") / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train_model(run_dir, timesteps, seed):
    env = DroneSwarm2D()
    monitored_env = Monitor(env, filename=str(run_dir / "monitor.csv"))
    model = PPO(
        "MlpPolicy",
        monitored_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        verbose=1,
        seed=seed,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    
    with open(run_dir / "training_log.txt", "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            model.learn(total_timesteps=timesteps)
            
    model.save(str(run_dir / "model"))
    monitored_env.close()
    return model


def evaluate_model(model, run_dir, episodes, seed):
    env = DroneSwarm2D()
    rows = []

    for episode in range(episodes):
        episode_seed = seed + episode
        obs, info = env.reset(seed=episode_seed)

        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0
        success = False
        collision = False

        for steps in range(1, MAX_EPISODE_STEPS + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            if info.get("success", False):
                success = True
            if info.get("collision", False):
                collision = True

            if terminated or truncated:
                break

        positions = np.asarray(env.positions)
        distances = np.linalg.norm(positions - TARGET, axis=1)

        rows.append({
            "episode": episode + 1,
            "seed": episode_seed,
            "success": success,
            "collision": collision,
            "reward": total_reward,
            "steps": steps,
            "mean_final_target_distance": float(np.mean(distances)),
        })

        print(
            f"Episode {episode + 1:02d}/{episodes}: "
            f"reward={total_reward:.2f}, "
            f"steps={steps}, "
            f"success={success}, "
            f"collision={collision}"
        )

    env.close()

    metrics_path = run_dir / "episode_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "episodes": episodes,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_episode_length": float(np.mean([row["steps"] for row in rows])),
        "mean_final_target_distance": float(
            np.mean([row["mean_final_target_distance"] for row in rows])
        ),
    }

    save_json(run_dir / "evaluation.json", summary)
    return summary


def create_animation(model, run_dir, seed=42, max_frames=MAX_EPISODE_STEPS, fps=10):
    env = DroneSwarm2D()
    obs, info = env.reset(seed=seed)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True, alpha=0.3)

    ax.scatter(
        TARGET[0], TARGET[1], marker="*", s=250, color="green", label="Target", zorder=5
    )

    for ox, oy, radius in OBSTACLES:
        obstacle = plt.Circle((ox, oy), radius, color="red", alpha=0.5, zorder=2)
        ax.add_patch(obstacle)

    colors = ["tab:blue", "tab:orange", "tab:purple"]
    drones = []
    trails = []

    trail_x = [[] for _ in range(NUM_DRONES)]
    trail_y = [[] for _ in range(NUM_DRONES)]

    for i in range(NUM_DRONES):
        drone, = ax.plot(
            [], [], marker="o", linestyle="None", color=colors[i], markersize=9, label=f"Drone {i + 1}", zorder=4
        )
        trail, = ax.plot(
            [], [], color=colors[i], alpha=0.65, linewidth=1.5, zorder=3
        )
        drones.append(drone)
        trails.append(trail)

    ax.legend(loc="upper left")
    title = ax.set_title("Trained PPO Drone Swarm")

    state = {
        "done": False,
        "steps": 0,
        "final_message": None,
    }

    def update(frame):
        nonlocal obs

        if not state["done"]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            state["steps"] += 1
            state["done"] = terminated or truncated

        positions = np.asarray(env.positions)

        for i, (x, y) in enumerate(positions):
            drones[i].set_data([x], [y])
            trail_x[i].append(x)
            trail_y[i].append(y)
            trails[i].set_data(trail_x[i], trail_y[i])

        title.set_text(f"Trained PPO Drone Swarm - Step {state['steps']}")

        if state["done"] and state["final_message"] is None:
            success = info.get("success", False)

            if success:
                message = "SUCCESS: All drones reached the target"
                color = "green"
            else:
                message = "Episode ended without reaching the target"
                color = "darkred"

            state["final_message"] = ax.text(
                0.5, 0.97, message, transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color=color, fontweight="bold"
            )

        return drones + trails

    animation = FuncAnimation(
        fig, update, frames=max_frames, interval=1000 / fps, blit=False, repeat=False
    )
    
    gif_path = run_dir / "swarm.gif"
    animation.save(str(gif_path), writer=PillowWriter(fps=fps))
    
    plt.close(fig)
    env.close()
    
    print(f"Animation saved: {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, and record a drone swarm experiment.")
    parser.add_argument("--name", default="ppo_baseline", help="Short name describing the experiment.")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Number of PPO training timesteps.")
    parser.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Training and animation random seed.")
    args = parser.parse_args()

    if args.timesteps <= 0 or args.episodes <= 0:
        parser.error("--timesteps and --episodes must be positive.")

    run_dir = create_run_directory(args.name)
    git_commit = get_git_commit()

    config = {
        "experiment_name": args.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "algorithm": "PPO",
        "total_timesteps": args.timesteps,
        "evaluation_episodes": args.episodes,
        "seed": args.seed,
        "learning_rate": LEARNING_RATE,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "target": TARGET.tolist(),
        "world_size": WORLD_SIZE,
        "obstacles": [list(obstacle) for obstacle in OBSTACLES],
        "git_commit": git_commit,
        "python_version": sys.version,
    }

    save_json(run_dir / "config.json", config)
    shutil.copy2("environment.py", run_dir / "environment.py")

    save_json(run_dir / "metadata.json", {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit,
    })

    print(f"\nExperiment: {args.name}")
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Training timesteps: {args.timesteps}")
    print(f"Evaluation episodes: {args.episodes}\n")

    try:
        print("1/3 - Training PPO...")
        model = train_model(run_dir, timesteps=args.timesteps, seed=args.seed)

        print("\n2/3 - Evaluating trained policy...")
        summary = evaluate_model(model, run_dir, episodes=args.episodes, seed=args.seed + 1000)

        print("\n3/3 - Generating GIF...")
        create_animation(model, run_dir, seed=args.seed)

        save_json(run_dir / "metadata.json", {
            "status": "completed",
            "started_at": config["created_at"],
            "completed_at": datetime.now().astimezone().isoformat(),
            "git_commit": git_commit,
        })

        print("\nExperiment completed.")
        print(f"Results directory: {run_dir.resolve()}")
        print("\nEvaluation summary:")
        print(json.dumps(summary, indent=2))

    except Exception as error:
        save_json(run_dir / "metadata.json", {
            "status": "failed",
            "failed_at": datetime.now().astimezone().isoformat(),
            "error": repr(error),
            "git_commit": git_commit,
        })
        raise


if __name__ == "__main__":
    main()