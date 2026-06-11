"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_FLUID_MODES = frozenset({"still", "drift", "turbulent"})


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class TargetConfig:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class SpawnConfig:
    x_min: float
    x_max: float


@dataclass(frozen=True)
class ObstacleConfig:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class ArenaConfig:
    width: int
    height: int
    target: TargetConfig
    spawn: SpawnConfig
    obstacles: tuple[ObstacleConfig, ...]
    max_steps: int
    dt: float


@dataclass(frozen=True)
class AgentConfig:
    count: int
    radius: float
    mass: float
    max_thrust: float
    drag: float
    max_speed: float
    delivery_radius: float
    sensing_radius: float
    neighbor_count: int
    obs_dim: int
    action_dim: int


@dataclass(frozen=True)
class DriftConfig:
    vx: float
    vy: float


@dataclass(frozen=True)
class TurbulentConfig:
    update_interval: int
    noise_scale: float
    smooth_sigma: float


@dataclass(frozen=True)
class PulsatileConfig:
    enabled: bool
    frequency_hz: float


@dataclass(frozen=True)
class FluidConfig:
    mode: str
    grid_cell_size: int
    flow_strength: float
    grad_strength: float
    drift: DriftConfig
    turbulent: TurbulentConfig
    poiseuille: bool
    pulsatile: PulsatileConfig


@dataclass(frozen=True)
class GradientConfig:
    blur_passes: int
    edge_concentration: float
    noise_scale: float


@dataclass(frozen=True)
class RewardConfig:
    distance_coeff: float
    delivery_bonus: float
    collision_penalty: float
    thrust_cost_coeff: float
    coverage_coeff: float
    time_penalty: float


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str
    rollout_length: int
    batch_size: int
    ppo_epochs: int
    clip_epsilon: float
    gamma: float
    gae_lambda: float
    learning_rate: float
    entropy_coeff: float
    max_grad_norm: float
    total_timesteps: int
    hidden_dim: int
    checkpoint_dir: str


@dataclass(frozen=True)
class NanoNavConfig:
    experiment: ExperimentConfig
    arena: ArenaConfig
    agents: AgentConfig
    fluid: FluidConfig
    gradient: GradientConfig
    reward: RewardConfig
    training: TrainingConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _parse_obstacles(raw: list[dict[str, Any]] | None) -> tuple[ObstacleConfig, ...]:
    if not raw:
        return ()
    return tuple(
        ObstacleConfig(
            x=float(item["x"]),
            y=float(item["y"]),
            width=float(item["width"]),
            height=float(item["height"]),
        )
        for item in raw
    )


def _parse_config(data: dict[str, Any]) -> NanoNavConfig:
    experiment = data["experiment"]
    arena = data["arena"]
    agents = data["agents"]
    fluid = data["fluid"]
    gradient = data["gradient"]
    reward = data["reward"]
    training = data["training"]

    fluid_mode = str(fluid["mode"])
    if fluid_mode not in VALID_FLUID_MODES:
        raise ValueError(
            f"Invalid fluid mode '{fluid_mode}'. Expected one of: {sorted(VALID_FLUID_MODES)}"
        )

    return NanoNavConfig(
        experiment=ExperimentConfig(
            name=str(experiment["name"]),
            seed=int(experiment["seed"]),
        ),
        arena=ArenaConfig(
            width=int(arena["width"]),
            height=int(arena["height"]),
            target=TargetConfig(
                x=float(arena["target"]["x"]),
                y=float(arena["target"]["y"]),
                radius=float(arena["target"]["radius"]),
            ),
            spawn=SpawnConfig(
                x_min=float(arena["spawn"]["x_min"]),
                x_max=float(arena["spawn"]["x_max"]),
            ),
            obstacles=_parse_obstacles(arena.get("obstacles")),
            max_steps=int(arena["max_steps"]),
            dt=float(arena["dt"]),
        ),
        agents=AgentConfig(
            count=int(agents["count"]),
            radius=float(agents["radius"]),
            mass=float(agents["mass"]),
            max_thrust=float(agents["max_thrust"]),
            drag=float(agents["drag"]),
            max_speed=float(agents["max_speed"]),
            delivery_radius=float(agents["delivery_radius"]),
            sensing_radius=float(agents["sensing_radius"]),
            neighbor_count=int(agents["neighbor_count"]),
            obs_dim=int(agents["obs_dim"]),
            action_dim=int(agents["action_dim"]),
        ),
        fluid=FluidConfig(
            mode=fluid_mode,
            grid_cell_size=int(fluid["grid_cell_size"]),
            flow_strength=float(fluid["flow_strength"]),
            grad_strength=float(fluid["grad_strength"]),
            drift=DriftConfig(
                vx=float(fluid["drift"]["vx"]),
                vy=float(fluid["drift"]["vy"]),
            ),
            turbulent=TurbulentConfig(
                update_interval=int(fluid["turbulent"]["update_interval"]),
                noise_scale=float(fluid["turbulent"]["noise_scale"]),
                smooth_sigma=float(fluid["turbulent"]["smooth_sigma"]),
            ),
            poiseuille=bool(fluid["poiseuille"]),
            pulsatile=PulsatileConfig(
                enabled=bool(fluid["pulsatile"]["enabled"]),
                frequency_hz=float(fluid["pulsatile"]["frequency_hz"]),
            ),
        ),
        gradient=GradientConfig(
            blur_passes=int(gradient["blur_passes"]),
            edge_concentration=float(gradient["edge_concentration"]),
            noise_scale=float(gradient["noise_scale"]),
        ),
        reward=RewardConfig(
            distance_coeff=float(reward["distance_coeff"]),
            delivery_bonus=float(reward["delivery_bonus"]),
            collision_penalty=float(reward["collision_penalty"]),
            thrust_cost_coeff=float(reward["thrust_cost_coeff"]),
            coverage_coeff=float(reward["coverage_coeff"]),
            time_penalty=float(reward["time_penalty"]),
        ),
        training=TrainingConfig(
            algorithm=str(training["algorithm"]),
            rollout_length=int(training["rollout_length"]),
            batch_size=int(training["batch_size"]),
            ppo_epochs=int(training["ppo_epochs"]),
            clip_epsilon=float(training["clip_epsilon"]),
            gamma=float(training["gamma"]),
            gae_lambda=float(training["gae_lambda"]),
            learning_rate=float(training["learning_rate"]),
            entropy_coeff=float(training["entropy_coeff"]),
            max_grad_norm=float(training["max_grad_norm"]),
            total_timesteps=int(training["total_timesteps"]),
            hidden_dim=int(training["hidden_dim"]),
            checkpoint_dir=str(training["checkpoint_dir"]),
        ),
    )


def load_config(
    config_path: str | Path,
    *,
    base_config_path: str | Path | None = None,
) -> NanoNavConfig:
    """Load a config file, optionally merging overrides onto a base config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if base_config_path is None:
        default_path = path.parent / "default.yaml"
        if path.name != "default.yaml" and default_path.exists():
            base_config_path = default_path

    if base_config_path is not None:
        base_data = _load_yaml(Path(base_config_path))
        override_data = _load_yaml(path)
        data = _deep_merge(base_data, override_data)
    else:
        data = _load_yaml(path)

    return _parse_config(data)
