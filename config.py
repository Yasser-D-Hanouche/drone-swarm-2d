"""Single place for every tunable value. Override from the CLI with --set section.name=value."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class EnvConfig:
    world_size: float = 10.0
    drone_radius: float = 0.15
    dt: float = 0.1                    # simulation time step
    max_speed: float = 1.0
    max_accel: float = 1.0             # action in [-1, 1] maps to this acceleration
    max_steps: int = 500               # episode is truncated (timeout) after this many steps
    target: list = field(default_factory=lambda: [8.5, 8.5])
    success_radius: float = 0.5        # ALL drones must be within this distance of the target
    start_center: list = field(default_factory=lambda: [1.0, 1.0])
    start_jitter: float = 0.0          # +/- uniform noise on the start centre; 0 = identical episodes
    formation: list = field(default_factory=lambda: [[-0.5, -0.4], [0.5, -0.4], [0.0, 0.5]])
    obstacles: list = field(default_factory=lambda: [[3.0, 3.0, 0.7], [5.0, 5.5, 0.9], [7.0, 3.5, 0.6]])  # x, y, radius
    cohesion_distance: float = 0.5     # a drone farther than this from its nearest neighbour is penalised


@dataclass
class RewardConfig:
    progress_weight: float = 5.0       # + weight * mean decrease in distance to target per step
    formation_weight: float = 1.0      # - weight * cohesion error
    step_penalty: float = 0.01         # - constant per step (pushes for speed)
    collision_penalty: float = 10.0    # - applied once; the episode ends on collision
    success_bonus: float = 20.0        # + applied once; the episode ends on success
    obstacle_margin: float = 0.0       # >0 enables shaping: penalty when closer than this to an obstacle
    obstacle_weight: float = 0.0       # - weight * mean(max(0, margin - clearance) / margin) per step


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    net_arch: list = field(default_factory=lambda: [64, 64])


@dataclass
class RunConfig:
    name: str = "baseline"
    hypothesis: str = ""               # one sentence: what do you expect this run to show? goes in the report
    seed: int = 42
    timesteps: int = 50_000
    eval_episodes: int = 30            # final evaluation episodes (same seeds for every policy)
    eval_freq: int = 5_000             # periodic evaluation during training; 0 disables
    periodic_eval_episodes: int = 10
    gif: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(key: str, current, value):
    expected = type(current)
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    ok = (
        isinstance(value, expected)
        and (expected is bool or not isinstance(value, bool))
    )
    if not ok:
        raise ValueError(f"'{key}' expects {expected.__name__}, got {value!r}")
    return value


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply 'section.name=value' strings. Values are parsed as JSON, falling back to plain strings."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        section_name, _, name = key.partition(".")
        section = getattr(cfg, section_name, None)
        if not sep or section is None or not hasattr(section, name):
            raise ValueError(
                f"Bad override '{item}'. Use section.name=value; see --list-params for valid names."
            )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        setattr(section, name, _coerce(key, getattr(section, name), value))
    return cfg


def from_dict(data: dict) -> Config:
    cfg = Config()
    for section_name, values in data.items():
        section = getattr(cfg, section_name)
        for name, value in values.items():
            if not hasattr(section, name):
                raise ValueError(f"Unknown parameter '{section_name}.{name}' in config file")
            setattr(section, name, value)
    return cfg


def flatten(cfg: Config) -> dict:
    return {
        f"{s.name}.{f.name}": getattr(getattr(cfg, s.name), f.name)
        for s in fields(cfg)
        for f in fields(getattr(cfg, s.name))
    }


def diff_from_default(cfg: Config) -> dict:
    default = flatten(Config())
    return {k: v for k, v in flatten(cfg).items() if default[k] != v}


def validate(cfg: Config) -> None:
    if len(cfg.env.formation) < 2:
        raise ValueError("env.formation needs at least 2 drones")
    if cfg.run.timesteps <= 0 or cfg.run.eval_episodes <= 0:
        raise ValueError("run.timesteps and run.eval_episodes must be positive")
    if cfg.ppo.batch_size > cfg.ppo.n_steps:
        raise ValueError("ppo.batch_size must not exceed ppo.n_steps")
    if cfg.reward.obstacle_weight > 0 and cfg.reward.obstacle_margin <= 0:
        raise ValueError("reward.obstacle_weight needs reward.obstacle_margin > 0")


def print_params() -> None:
    default = Config()
    for section in fields(default):
        print(f"[{section.name}]")
        for f in fields(getattr(default, section.name)):
            print(f"  {section.name}.{f.name} = {json.dumps(getattr(getattr(default, section.name), f.name))}")
