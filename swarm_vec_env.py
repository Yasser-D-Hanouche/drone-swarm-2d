"""Parameter sharing for SB3: expose every drone of every swarm environment as its own row of a vector env."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


class SharedPolicyVecEnv(VecEnv):
    """Wraps a VecEnv of swarm envs (obs (n, d), action (n, 2)) into num_envs * n rows of (d,) and (2,).

    All drones of a swarm share one episode (same done flag) but each row gets that drone's own reward
    (info["drone_rewards"]). PPO trains one network on the rows of every drone.
    """

    def __init__(self, venv: VecEnv, n_drones: int):
        self.venv = venv
        self.n = n_drones
        obs_space = spaces.Box(-np.inf, np.inf, shape=venv.observation_space.shape[1:], dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=venv.action_space.shape[1:], dtype=np.float32)
        super().__init__(venv.num_envs * n_drones, obs_space, act_space)
        self.render_mode = getattr(venv, "render_mode", None)

    def reset(self):
        obs = self.venv.reset()
        return obs.reshape(-1, obs.shape[-1])

    def step_async(self, actions):
        self.venv.step_async(np.asarray(actions).reshape(self.venv.num_envs, self.n, -1))

    def step_wait(self):
        obs, _, dones, infos = self.venv.step_wait()
        rows, rewards = [], []
        for info in infos:
            for j in range(self.n):
                row = dict(info)
                if "terminal_observation" in info:
                    row["terminal_observation"] = info["terminal_observation"][j]
                rows.append(row)
                rewards.append(info["drone_rewards"][j])
        return obs.reshape(-1, obs.shape[-1]), np.asarray(rewards, dtype=np.float32), np.repeat(dones, self.n), rows

    def close(self):
        self.venv.close()

    def get_attr(self, attr_name, indices=None):
        return [v for v in self.venv.get_attr(attr_name) for _ in range(self.n)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
