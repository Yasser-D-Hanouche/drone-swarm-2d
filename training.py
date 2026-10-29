"""PPO training with a callback that records outcomes, reward terms and periodic evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from config import Config
from diagnostics import ppo_policy, run_episodes, summarize
from environment import OUTCOMES, REWARD_TERMS, DroneSwarm2D


class TrainingMonitor(BaseCallback):
    def __init__(self, cfg: Config, console: bool = True):
        super().__init__()
        self.cfg = cfg
        self.console = console
        self.episodes: list[dict] = []
        self.eval_curve: list[dict] = []
        self._since_log = 0
        self._next_eval = cfg.run.eval_freq

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self._acc = [dict.fromkeys(REWARD_TERMS, 0.0) for _ in range(n_envs)]
        self._len = [0] * n_envs

    def _on_step(self):
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            for k, v in info["reward_terms"].items():
                self._acc[i][k] += v
            self._len[i] += 1
            if done:
                row = {
                    "timesteps": self.num_timesteps,
                    "outcome": info["termination"],
                    "reward": sum(self._acc[i].values()),
                    "length": self._len[i],
                    "final_mean_dist": info["mean_target_distance"],
                }
                row.update({f"r_{k}": v for k, v in self._acc[i].items()})
                self.episodes.append(row)
                self._acc[i] = dict.fromkeys(REWARD_TERMS, 0.0)
                self._len[i] = 0

        run = self.cfg.run
        if run.eval_freq and self.num_timesteps >= self._next_eval:
            self._periodic_eval()
            self._next_eval += run.eval_freq
        return True

    def _periodic_eval(self):
        run = self.cfg.run
        env = self.cfg.env
        random_env = env.start_jitter > 0 or env.random_obstacles or env.random_start or env.random_target
        n = run.periodic_eval_episodes if random_env else 1
        s = summarize(run_episodes(self.cfg, ppo_policy(self.model), n, run.seed + 5000))
        self.eval_curve.append({
            "timesteps": self.num_timesteps,
            "success_rate": s["success_rate"],
            "collision_rate": s["collision_rate"],
            "mean_reward": s["mean_reward"],
            "mean_final_dist": s["mean_final_dist"],
        })
        self.logger.record("eval/success_rate", s["success_rate"])
        self.logger.record("eval/mean_reward", s["mean_reward"])

    def _on_rollout_end(self):
        recent = self.episodes[self._since_log:]
        self._since_log = len(self.episodes)
        if not recent:
            return
        rate = {o: np.mean([r["outcome"] == o for r in recent]) for o in OUTCOMES}
        for o, v in rate.items():
            self.logger.record(f"outcome/{o}", float(v))
        for k in REWARD_TERMS:
            self.logger.record(f"reward_term/{k}", float(np.mean([r[f"r_{k}"] for r in recent])))
        if self.console:
            print(
                f"[{self.num_timesteps:>8,}/{self.cfg.run.timesteps:,}] episodes={len(recent):>3} "
                f"success={rate['success']:.0%} obstacle={rate['obstacle']:.0%} boundary={rate['boundary']:.0%} "
                f"drone={rate['drone_collision']:.0%} timeout={rate['timeout']:.0%} "
                f"return={np.mean([r['reward'] for r in recent]):.1f}"
            )


def _log_formats():
    try:
        import torch.utils.tensorboard  # noqa: F401
        return ["csv", "tensorboard"]
    except ImportError:
        return ["csv"]


def train(cfg: Config, run_dir: Path, console: bool = True):
    p = cfg.ppo
    env = make_vec_env(
        lambda: DroneSwarm2D(cfg.env, cfg.reward),
        n_envs=p.n_envs,
        seed=cfg.run.seed,
        vec_env_cls=SubprocVecEnv if p.n_envs > 1 else None,
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=p.learning_rate,
        n_steps=p.n_steps,
        batch_size=p.batch_size,
        n_epochs=p.n_epochs,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        clip_range=p.clip_range,
        ent_coef=p.ent_coef,
        policy_kwargs={"net_arch": list(p.net_arch), "log_std_init": p.log_std_init},
        seed=cfg.run.seed,
        device="cpu",
        verbose=0,
    )
    model.set_logger(configure(str(run_dir / "sb3_log"), _log_formats()))
    monitor = TrainingMonitor(cfg, console=console)
    model.learn(total_timesteps=cfg.run.timesteps, callback=monitor)
    model.save(str(run_dir / "model"))
    env.close()
    return model, monitor
