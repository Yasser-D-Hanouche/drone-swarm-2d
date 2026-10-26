import numpy as np
from stable_baselines3 import PPO
from environment import DroneSwarm2D

env = DroneSwarm2D()
model = PPO.load("ppo_drone_swarm")

episodes = 20
successes = 0
episode_rewards = []
episode_lengths = []

for episode in range(episodes):
    obs, info = env.reset(seed=episode)
    total_reward = 0.0
    episode_success = False

    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info.get("success", False):
            episode_success = True

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)
    episode_lengths.append(step + 1)
    successes += int(episode_success)

    print(
        f"Episode {episode + 1:02d}: "
        f"reward={total_reward:.2f}, "
        f"steps={step + 1}, "
        f"success={episode_success}"
    )

print("\nEvaluation summary")
print(f"Success rate: {successes / episodes:.1%}")
print(f"Mean reward: {np.mean(episode_rewards):.2f}")
print(f"Mean episode length: {np.mean(episode_lengths):.1f}")