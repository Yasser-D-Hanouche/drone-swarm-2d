from stable_baselines3.common.env_checker import check_env
from environment import DroneSwarm2D

env = DroneSwarm2D()

print("Checking the environment...")
check_env(env, warn=True)
print("Environment check completed.")
