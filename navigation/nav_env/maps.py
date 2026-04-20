from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CircularObstacle:
    x: float
    y: float
    radius: float


def random_obstacles(
    rng: np.random.Generator,
    num_obstacles: int,
    world_size: float,
    min_radius: float,
    max_radius: float,
) -> list[CircularObstacle]:
    span = world_size * 0.45
    out = []
    for _ in range(num_obstacles):
        x, y = rng.uniform(-span, span, size=2)
        radius = float(rng.uniform(min_radius, max_radius))
        out.append(CircularObstacle(float(x), float(y), radius))
    return out


def obstacle_costmap(
    robot_xyh: tuple[float, float, float],
    obstacles: list[CircularObstacle],
    map_size: int,
    map_extent_m: float,
    inflation_radius_m: float,
) -> np.ndarray:
    rx, ry, heading = robot_xyh

    half = map_extent_m / 2.0
    grid = np.linspace(-half, half, map_size, dtype=np.float32)
    gx, gy = np.meshgrid(grid, grid, indexing="xy")

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    wx = rx + gx * cos_h - gy * sin_h
    wy = ry + gx * sin_h + gy * cos_h

    costmap = np.zeros((map_size, map_size), dtype=np.float32)
    for obs in obstacles:
        d = np.sqrt((wx - obs.x) ** 2 + (wy - obs.y) ** 2)
        margin = d - obs.radius
        occupied = margin <= 0
        inflated = (margin > 0) & (margin < inflation_radius_m)

        costmap = np.maximum(costmap, occupied.astype(np.float32))
        if np.any(inflated):
            inflated_cost = 1.0 - (margin / inflation_radius_m)
            costmap = np.maximum(costmap, np.clip(inflated_cost, 0.0, 1.0).astype(np.float32))

    return costmap


def collision_distance(
    robot_x: float,
    robot_y: float,
    obstacles: list[CircularObstacle],
) -> float:
    if not obstacles:
        return np.inf

    min_margin = np.inf
    for obs in obstacles:
        d = np.sqrt((robot_x - obs.x) ** 2 + (robot_y - obs.y) ** 2)
        min_margin = min(min_margin, d - obs.radius)

    return float(min_margin)
