from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment import DroneSwarm2D

# Create and validate the environment
env = DroneSwarm2D()
check_env(env, warn=True)

# Create the PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    seed=42,
)

# Train the policy
model.learn(total_timesteps=50_000)

# Save the trained policy
model.save("ppo_drone_swarm")

print("Training complete. Model saved to ppo_drone_swarm.zip")
