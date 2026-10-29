"""Single place for every tunable value.

Edit the defaults below, or leave them alone and override per run:
  python run_experiment.py --stage 1_known_fixed            # preset from STAGES at the bottom of this file
  python run_experiment.py --set reward.collision_penalty=20 ppo.learning_rate=1e-4
  python run_experiment.py --list-params                    # every parameter and its default
Order of precedence: defaults < --config file < --stage < --timesteps/--seed/... < --set.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields


@dataclass
class EnvConfig:
    # --- world and physics (units: metres, seconds) ---
    world_size: float = 10.0           # square arena, walls at 0 and world_size
    drone_radius: float = 0.15         # two drones collide when centres are closer than 2 * drone_radius
    dt: float = 0.1                    # seconds per step
    max_speed: float = 1.0
    max_accel: float = 1.0
    max_steps: int = 800               # episode time limit (steps); searching needs far more than flying to a known target
    num_drones: int = 3

    # --- start and target ---
    random_start: bool = True          # swarm centre drawn uniformly from [edge_margin, world_size - edge_margin]^2
    random_target: bool = True         # same for the target
    edge_margin: float = 1.0           # keeps random starts/targets away from the walls
    min_start_target_distance: float = 5.0  # resample until start centre and target are at least this far apart
    start_center: list = field(default_factory=lambda: [1.5, 1.5])  # used only if random_start=false
    start_jitter: float = 0.3          # random shift of start_center, +- per axis (random_start=false only)
    target: list = field(default_factory=lambda: [8.5, 8.5])  # used only if random_target=false
    start_spread: float = 0.5          # drones start evenly spaced on a circle of this radius around the start centre
    success_radius: float = 0.8        # success = every drone within this distance of the target
    cohesion_radius: float = 1.0       # max distance from the swarm centroid before the cohesion penalty applies

    # --- target search (the target is hidden from the policy until found) ---
    target_sense_range: float = 1.5    # a drone detects the target, and marks map cells searched, within this distance
    explore_cell: float = 1.0          # cell size of the searched map: (world_size / cell)^2 values in the observation

    # --- obstacle vision (rays point along each drone's velocity) ---
    num_rays: int = 7
    vision_angle: float = 120.0        # degrees covered by the rays
    vision_range: float = 2.0          # metres; stopping distance at max_speed is max_speed^2 / (2 * max_accel)

    # --- obstacles ---
    random_obstacles: bool = True      # new random obstacles every episode; false = use the fixed list below
    min_obstacles: int = 3             # set min=max=0 for an obstacle-free world
    max_obstacles: int = 5
    min_obstacle_radius: float = 0.5
    max_obstacle_radius: float = 0.9
    obstacle_safe_margin: float = 0.6  # random obstacles keep this much free space around the start drones and the target
    obstacles: list = field(default_factory=lambda: [[3.0, 3.0, 0.7], [5.0, 5.5, 0.9], [7.0, 3.5, 0.6]])  # [x, y, radius]; random_obstacles=false only


@dataclass
class RewardConfig:
    # Per-step terms; the success/collision terms are paid once because the episode ends.
    progress_weight: float = 2.0       # + weight * mean decrease in distance to the target per step (only once found)
    explore_weight: float = 0.2        # + weight per map cell searched for the first time (only until the target is found)
    cohesion_weight: float = 0.5       # - weight * mean(max(0, distance to centroid - env.cohesion_radius)) per step
    separation_weight: float = 1.0     # - weight * mean(max(0, margin - gap to nearest drone) / margin) per step
    separation_margin: float = 0.3     # gap (m, surface to surface) below which drones are penalised; 0 disables
    obstacle_weight: float = 1.0       # - weight * mean(max(0, margin - clearance) / margin) per step
    obstacle_margin: float = 1.0       # m; shaping distance, keep <= env.vision_range (the policy cannot see farther)
    step_penalty: float = 0.01         # - constant per step (pushes for speed)
    collision_penalty: float = 10.0    # - once; the episode ends on any collision (obstacle, wall, drone)
    success_bonus: float = 50.0        # + once; the episode ends on success


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_envs: int = 1                    # parallel environments (separate processes if > 1); n_steps below is per environment
    n_steps: int = 1024                # rollout length per environment per update
    batch_size: int = 64               # must not exceed n_steps; ideally divides n_steps * n_envs
    n_epochs: int = 10
    gamma: float = 0.99                # discount; 0.99 looks ~100 steps (10 s) ahead
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0              # raise (e.g. 0.005) if the policy stops exploring too early
    log_std_init: float = -1.0         # initial action noise = exp(value); SB3 default 0 (std 1.0) drowns a [-1, 1] action range
    net_arch: list = field(default_factory=lambda: [64, 64])  # hidden layer sizes; try [128, 128] for the larger observation


@dataclass
class RunConfig:
    name: str = "baseline"             # run folder suffix; runs with the same name are grouped as seeds by compare_runs.py
    hypothesis: str = ""               # one sentence: what do you expect this run to show? goes in the report
    seed: int = 42
    timesteps: int = 100_000           # total training steps over all environments
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


# Curriculum presets for --stage. Each is a set of overrides on the defaults above.
# Move to the next stage once the current one reaches ~80% success over several seeds.
# run.timesteps are rough starting budgets: adjust them from eval_curve.csv / training_curves.png.
_NO_OBSTACLES = {"env.min_obstacles": 0, "env.max_obstacles": 0}
_KNOWN_TARGET = {"env.target_sense_range": 20.0}  # larger than the world: the target is found at step 0
STAGES: dict[str, dict] = {
    "1_known_fixed": {  # one fixed layout, target known, no obstacles: proves the setup works at all
        **_NO_OBSTACLES, **_KNOWN_TARGET,
        "env.random_start": False, "env.random_target": False, "run.timesteps": 300_000,
    },
    "2_known_random": {  # random start and target, still known and no obstacles
        **_NO_OBSTACLES, **_KNOWN_TARGET, "run.timesteps": 500_000,
    },
    "3_search": {  # target hidden: the swarm must explore
        **_NO_OBSTACLES, "run.timesteps": 1_000_000,
    },
    "4_search_few_obstacles": {  # search plus 1-2 random obstacles
        "env.min_obstacles": 1, "env.max_obstacles": 2, "run.timesteps": 1_000_000,
    },
    "5_full": {"run.timesteps": 3_000_000},  # defaults: hidden target, 3-5 random obstacles, everything random
}


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


def apply_stage(cfg: Config, stage: str) -> Config:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {', '.join(STAGES)}")
    return apply_overrides(cfg, [f"{k}={json.dumps(v)}" for k, v in STAGES[stage].items()])


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


def print_params() -> None:
    default = Config()
    for section in fields(default):
        print(f"[{section.name}]")
        for f in fields(getattr(default, section.name)):
            print(f"  {section.name}.{f.name} = {json.dumps(getattr(getattr(default, section.name), f.name))}")
    print("\n[stages]  (use --stage NAME)")
    for name, overrides in STAGES.items():
        print(f"  {name}: {json.dumps(overrides)}")


def validate(cfg: Config) -> None:
    e, r, p = cfg.env, cfg.reward, cfg.ppo
    if e.num_drones < 2:
        raise ValueError("env.num_drones must be at least 2")
    if cfg.run.timesteps <= 0 or cfg.run.eval_episodes <= 0:
        raise ValueError("run.timesteps and run.eval_episodes must be positive")
    if p.n_envs < 1:
        raise ValueError("ppo.n_envs must be at least 1")
    if p.batch_size > p.n_steps:
        raise ValueError("ppo.batch_size must not exceed ppo.n_steps")
    if r.obstacle_weight > 0 and r.obstacle_margin <= 0:
        raise ValueError("reward.obstacle_weight needs reward.obstacle_margin > 0")
    if r.separation_weight > 0 and r.separation_margin <= 0:
        raise ValueError("reward.separation_weight needs reward.separation_margin > 0")
    if e.min_obstacles < 0 or e.max_obstacles < e.min_obstacles:
        raise ValueError("Invalid obstacle count range")
    if e.min_obstacle_radius <= 0 or e.max_obstacle_radius < e.min_obstacle_radius:
        raise ValueError("Invalid obstacle radius range")
    if e.num_rays < 1 or e.vision_range <= 0 or not 0 < e.vision_angle <= 360:
        raise ValueError("Need env.num_rays >= 1, env.vision_range > 0 and 0 < env.vision_angle <= 360")
    if e.success_radius <= 0 or e.drone_radius <= 0:
        raise ValueError("env.success_radius and env.drone_radius must be positive")
    if e.target_sense_range <= 0 or e.explore_cell <= 0:
        raise ValueError("env.target_sense_range and env.explore_cell must be positive")
    if e.target_sense_range < e.success_radius:
        raise ValueError("env.target_sense_range must be >= env.success_radius")

    spacing = 2 * e.start_spread * math.sin(math.pi / e.num_drones)
    if spacing <= 2 * e.drone_radius:
        raise ValueError(f"env.start_spread too small: start spacing {spacing:.2f} <= 2*drone_radius")
    if r.cohesion_weight > 0 and e.start_spread > e.cohesion_radius:
        raise ValueError("env.start_spread exceeds env.cohesion_radius, so every episode starts penalised")

    lo, hi = e.drone_radius, e.world_size - e.drone_radius
    if e.random_start or e.random_target:
        free = e.world_size - 2 * e.edge_margin
        if e.edge_margin < e.start_spread + e.drone_radius or free <= 0:
            raise ValueError("env.edge_margin must be >= start_spread + drone_radius and < world_size / 2")
        if e.min_start_target_distance >= free:
            raise ValueError("env.min_start_target_distance is too large for the area left by env.edge_margin")
    if not e.random_start:
        reach = e.start_spread + e.start_jitter
        if not all(lo + reach <= v <= hi - reach for v in e.start_center):
            raise ValueError("env.start_center +- (start_spread + start_jitter) leaves the world; move it inward")
    if not e.random_target and not all(lo <= v <= hi for v in e.target):
        raise ValueError("env.target leaves the world")
