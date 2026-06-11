"""pymunk physics arena: walls, obstacles, spawn zone, and target."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pymunk

from nanonav.config import ArenaConfig, ObstacleConfig

WALL_ELASTICITY = 1.0
WALL_FRICTION = 0.0
OBSTACLE_ELASTICITY = 1.0
OBSTACLE_FRICTION = 0.0


@dataclass(frozen=True)
class SpawnBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class Arena:
    """Bounded 2D arena using pymunk for collision resolution.

    Public positions use screen coordinates: origin at top-left, y increases downward.
    """

    def __init__(self, config: ArenaConfig) -> None:
        self._config = config
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)
        self._static_bodies: list[pymunk.Body] = []
        self._dynamic_bodies: list[pymunk.Body] = []
        self._build_static_geometry()

    @property
    def config(self) -> ArenaConfig:
        return self._config

    @property
    def space(self) -> pymunk.Space:
        return self._space

    @property
    def width(self) -> int:
        return self._config.width

    @property
    def height(self) -> int:
        return self._config.height

    @property
    def target_pos(self) -> tuple[float, float]:
        target = self._config.target
        return (target.x, target.y)

    @property
    def target_radius(self) -> float:
        return self._config.target.radius

    @property
    def spawn_bounds(self) -> SpawnBounds:
        margin = self._config.target.radius
        return SpawnBounds(
            x_min=self._config.spawn.x_min,
            x_max=self._config.spawn.x_max,
            y_min=margin,
            y_max=self._config.height - margin,
        )

    def reset(self) -> None:
        """Remove all dynamic bodies and clear collision callbacks."""
        for body in self._dynamic_bodies:
            self._space.remove(body, *body.shapes)
        self._dynamic_bodies.clear()

    def step(self, dt: float | None = None) -> None:
        """Advance the pymunk simulation by one timestep."""
        self._space.step(self._config.dt if dt is None else dt)

    def add_circle(
        self,
        position: tuple[float, float],
        radius: float,
        *,
        mass: float = 1.0,
        velocity: tuple[float, float] = (0.0, 0.0),
        elasticity: float = 1.0,
        friction: float = 0.0,
    ) -> pymunk.Body:
        """Create a dynamic circle body in screen coordinates."""
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0.0, radius))
        body.position = self._to_pymunk(*position)
        body.velocity = self._velocity_to_pymunk(*velocity)
        shape = pymunk.Circle(body, radius)
        shape.elasticity = elasticity
        shape.friction = friction
        shape.collision_type = 1
        self._space.add(body, shape)
        self._dynamic_bodies.append(body)
        return body

    def get_position(self, body: pymunk.Body) -> tuple[float, float]:
        return self._from_pymunk(body.position.x, body.position.y)

    def get_velocity(self, body: pymunk.Body) -> tuple[float, float]:
        return self._velocity_from_pymunk(body.velocity.x, body.velocity.y)

    def set_velocity(self, body: pymunk.Body, velocity: tuple[float, float]) -> None:
        body.velocity = self._velocity_to_pymunk(*velocity)

    def sample_spawn_position(self, rng: random.Random | None = None) -> tuple[float, float]:
        """Sample a random spawn point inside the left-edge spawn strip."""
        source = rng or random
        bounds = self.spawn_bounds
        return (
            source.uniform(bounds.x_min, bounds.x_max),
            source.uniform(bounds.y_min, bounds.y_max),
        )

    def is_in_target_zone(self, position: tuple[float, float]) -> bool:
        tx, ty = self.target_pos
        px, py = position
        dx = px - tx
        dy = py - ty
        return (dx * dx + dy * dy) <= self.target_radius ** 2

    def is_in_bounds(self, position: tuple[float, float], *, margin: float = 0.0) -> bool:
        x, y = position
        return (
            margin <= x <= self.width - margin
            and margin <= y <= self.height - margin
        )

    def ray_cast(
        self,
        origin: tuple[float, float],
        direction: tuple[float, float],
        max_distance: float,
    ) -> float:
        """Cast a ray and return hit distance, or max_distance if no hit."""
        ox, oy = self._to_pymunk(*origin)
        dx, dy = direction
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0.0:
            return max_distance

        nx, ny = dx / length, dy / length
        end_x = ox + nx * max_distance
        end_y = oy + ny * max_distance

        query = self._space.segment_query_first(
            (ox, oy),
            (end_x, end_y),
            0.0,
            pymunk.ShapeFilter(),
        )
        if query is None:
            return max_distance
        return query.alpha * max_distance

    def _build_static_geometry(self) -> None:
        self._add_walls()
        for obstacle in self._config.obstacles:
            self._add_obstacle(obstacle)

    def _add_walls(self) -> None:
        width = float(self._config.width)
        height = float(self._config.height)
        wall_segments = (
            ((0.0, 0.0), (width, 0.0)),
            ((width, 0.0), (width, height)),
            ((width, height), (0.0, height)),
            ((0.0, height), (0.0, 0.0)),
        )
        for start, end in wall_segments:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, start, end, 1.0)
            shape.elasticity = WALL_ELASTICITY
            shape.friction = WALL_FRICTION
            shape.collision_type = 0
            self._space.add(body, shape)
            self._static_bodies.append(body)

    def _add_obstacle(self, obstacle: ObstacleConfig) -> None:
        center_screen = (
            obstacle.x + obstacle.width / 2.0,
            obstacle.y + obstacle.height / 2.0,
        )
        center = self._to_pymunk(*center_screen)
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Poly.create_box(
            body,
            (obstacle.width, obstacle.height),
            radius=0.0,
        )
        body.position = center
        shape.elasticity = OBSTACLE_ELASTICITY
        shape.friction = OBSTACLE_FRICTION
        shape.collision_type = 0
        self._space.add(body, shape)
        self._static_bodies.append(body)

    def _to_pymunk(self, x: float, y: float) -> tuple[float, float]:
        return (x, float(self._config.height) - y)

    def _from_pymunk(self, x: float, y: float) -> tuple[float, float]:
        return (x, float(self._config.height) - y)

    def _velocity_to_pymunk(self, vx: float, vy: float) -> tuple[float, float]:
        return (vx, -vy)

    def _velocity_from_pymunk(self, vx: float, vy: float) -> tuple[float, float]:
        return (vx, -vy)
