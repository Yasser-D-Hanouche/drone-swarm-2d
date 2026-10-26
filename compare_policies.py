
import numpy as np
from stable_baselines3 import PPO

from environment import DroneSwarm2D

MODEL_PATH = "ppo_drone_swarm"
N_EPISODES = 30
MAX_STEPS = 500


def evaluate_policy(env, model=None, seed=0):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    formation_errors = []
    collision = False
    success = False

    for step in range(MAX_STEPS):
        if model is None:
            # Random baseline
            action = env.action_space.sample()
        else:
            # Trained PPO policy
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if "formation_error" in info:
            formation_errors.append(info["formation_error"])

        collision = collision or info.get("collision", False)
        success = success or info.get("success", False)

        if terminated or truncated:
            break

    return {
        "success": success,
        "collision": collision,
        "reward": total_reward,
        "steps": step + 1,
        "formation_error": (
            float(np.mean(formation_errors))
            if formation_errors else float("nan")
        ),
    }


def main():
    env = DroneSwarm2D()
    model = PPO.load(MODEL_PATH)

    for name, policy in [
        ("Random", None),
        ("PPO", model),
    ]:
        results = [
            evaluate_policy(env, policy, seed=seed)
            for seed in range(N_EPISODES)
        ]

        print(f"\n{name} policy — {N_EPISODES} episodes")
        print(f"Success rate: {np.mean([r['success'] for r in results]):.1%}")
        print(f"Collision rate: {np.mean([r['collision'] for r in results]):.1%}")
        print(f"Mean reward: {np.mean([r['reward'] for r in results]):.2f}")
        print(f"Mean episode length: {np.mean([r['steps'] for r in results]):.1f}")
        print(
            "Mean formation error: "
            f"{np.nanmean([r['formation_error'] for r in results]):.3f}"
        )


if __name__ == "__main__":
    main()
