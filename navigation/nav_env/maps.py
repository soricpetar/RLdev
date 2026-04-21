from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CircularObstacle:
    x: float
    y: float
    radius: float
    kind: str = "tree"
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class WallObstacle:
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float
    kind: str = "wall"


Object = CircularObstacle | WallObstacle


OBJECT_COSTS = {
    "tree": 1.0,
    "wall": 1.0,
    "person": 1.0,
    "pothole": 0.65,
    "bush": 0.42,
}

COLLIDABLE_KINDS = {"tree", "wall", "person"}
SEMANTIC_CHANNELS = ("hard_static", "soft_vegetation", "terrain_hazard", "dynamic_person", "wall")


def semantic_channel(kind: str) -> int:
    if kind == "tree":
        return 0
    if kind == "bush":
        return 1
    if kind == "pothole":
        return 2
    if kind == "person":
        return 3
    if kind == "wall":
        return 4
    return 0


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
        out.append(CircularObstacle(float(x), float(y), radius, kind="tree"))
    return out


def random_field_objects(
    rng: np.random.Generator,
    world_size: float,
    tree_rows_range: tuple[int, int],
    bushes_range: tuple[int, int],
    potholes_range: tuple[int, int],
    people_range: tuple[int, int],
    walls_range: tuple[int, int],
) -> list[Object]:
    span = world_size * 0.42
    objects: list[Object] = []

    for _ in range(int(rng.integers(tree_rows_range[0], tree_rows_range[1] + 1))):
        y = float(rng.uniform(-span, span))
        x0 = float(rng.uniform(-span, -span * 0.2))
        spacing = float(rng.uniform(2.2, 3.6))
        count = int(rng.integers(4, 9))
        jitter = float(rng.uniform(0.0, 0.35))
        for idx in range(count):
            x = x0 + idx * spacing + float(rng.normal(0.0, jitter))
            if abs(x) > span:
                continue
            objects.append(
                CircularObstacle(
                    x=x,
                    y=y + float(rng.normal(0.0, jitter)),
                    radius=float(rng.uniform(0.35, 0.75)),
                    kind="tree",
                )
            )

    for _ in range(int(rng.integers(bushes_range[0], bushes_range[1] + 1))):
        x, y = rng.uniform(-span, span, size=2)
        objects.append(CircularObstacle(float(x), float(y), float(rng.uniform(0.45, 1.25)), kind="bush"))

    for _ in range(int(rng.integers(potholes_range[0], potholes_range[1] + 1))):
        x, y = rng.uniform(-span, span, size=2)
        objects.append(CircularObstacle(float(x), float(y), float(rng.uniform(0.35, 0.95)), kind="pothole"))

    for _ in range(int(rng.integers(people_range[0], people_range[1] + 1))):
        x, y = rng.uniform(-span, span, size=2)
        heading = float(rng.uniform(-np.pi, np.pi))
        speed = float(rng.uniform(0.15, 0.55))
        objects.append(
            CircularObstacle(
                float(x),
                float(y),
                float(rng.uniform(0.32, 0.45)),
                kind="person",
                vx=speed * float(np.cos(heading)),
                vy=speed * float(np.sin(heading)),
            )
        )

    for _ in range(int(rng.integers(walls_range[0], walls_range[1] + 1))):
        cx, cy = rng.uniform(-span, span, size=2)
        length = float(rng.uniform(4.0, 10.0))
        angle = float(rng.uniform(-np.pi, np.pi))
        dx = 0.5 * length * float(np.cos(angle))
        dy = 0.5 * length * float(np.sin(angle))
        objects.append(
            WallObstacle(
                x1=float(cx - dx),
                y1=float(cy - dy),
                x2=float(cx + dx),
                y2=float(cy + dy),
                thickness=float(rng.uniform(0.25, 0.55)),
            )
        )

    return objects


def object_cost(kind: str) -> float:
    return OBJECT_COSTS.get(kind, 1.0)


def point_segment_distance(px, py, x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    denom = vx * vx + vy * vy
    t = np.clip((wx * vx + wy * vy) / (denom + 1e-8), 0.0, 1.0)
    cx = x1 + t * vx
    cy = y1 + t * vy
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def object_margin(x: float, y: float, obj: Object) -> float:
    if isinstance(obj, WallObstacle):
        d = point_segment_distance(x, y, obj.x1, obj.y1, obj.x2, obj.y2)
        return float(d - obj.thickness * 0.5)

    d = np.sqrt((x - obj.x) ** 2 + (y - obj.y) ** 2)
    return float(d - obj.radius)


def move_dynamic_objects(objects: list[Object], dt: float, world_size: float) -> None:
    half = world_size / 2.0
    for obj in objects:
        if not isinstance(obj, CircularObstacle) or obj.kind != "person":
            continue
        obj.x += obj.vx * dt
        obj.y += obj.vy * dt
        if obj.x < -half or obj.x > half:
            obj.x = float(np.clip(obj.x, -half, half))
            obj.vx *= -1.0
        if obj.y < -half or obj.y > half:
            obj.y = float(np.clip(obj.y, -half, half))
            obj.vy *= -1.0


def obstacle_costmap(
    robot_xyh: tuple[float, float, float],
    obstacles: list[Object],
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
        if isinstance(obs, WallObstacle):
            d = point_segment_distance(wx, wy, obs.x1, obs.y1, obs.x2, obs.y2)
            margin = d - obs.thickness * 0.5
        else:
            d = np.sqrt((wx - obs.x) ** 2 + (wy - obs.y) ** 2)
            margin = d - obs.radius
        occupied = margin <= 0
        inflated = (margin > 0) & (margin < inflation_radius_m)
        cost = object_cost(obs.kind)

        costmap = np.maximum(costmap, occupied.astype(np.float32) * cost)
        if np.any(inflated):
            inflated_cost = cost * (1.0 - (margin / inflation_radius_m))
            costmap = np.maximum(costmap, np.clip(inflated_cost, 0.0, 1.0).astype(np.float32))

    return costmap


def semantic_costmap(
    robot_xyh: tuple[float, float, float],
    objects: list[Object],
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

    costmap = np.zeros((len(SEMANTIC_CHANNELS), map_size, map_size), dtype=np.float32)
    for obj in objects:
        if isinstance(obj, WallObstacle):
            d = point_segment_distance(wx, wy, obj.x1, obj.y1, obj.x2, obj.y2)
            margin = d - obj.thickness * 0.5
        else:
            d = np.sqrt((wx - obj.x) ** 2 + (wy - obj.y) ** 2)
            margin = d - obj.radius

        occupied = margin <= 0
        inflated = (margin > 0) & (margin < inflation_radius_m)
        channel = semantic_channel(obj.kind)
        cost = object_cost(obj.kind)

        layer = costmap[channel]
        layer[:] = np.maximum(layer, occupied.astype(np.float32) * cost)
        if np.any(inflated):
            inflated_cost = cost * (1.0 - (margin / inflation_radius_m))
            layer[:] = np.maximum(layer, np.clip(inflated_cost, 0.0, 1.0).astype(np.float32))

    return costmap


def collision_distance(
    robot_x: float,
    robot_y: float,
    obstacles: list[Object],
) -> float:
    if not obstacles:
        return np.inf

    min_margin = np.inf
    for obs in obstacles:
        if obs.kind not in COLLIDABLE_KINDS:
            continue
        min_margin = min(min_margin, object_margin(robot_x, robot_y, obs))

    return float(min_margin)


def semantic_contact_penalty(
    robot_x: float,
    robot_y: float,
    robot_radius: float,
    objects: list[Object],
) -> tuple[float, dict[str, float]]:
    penalties = {"bush": 0.0, "pothole": 0.0, "person": 0.0}
    penalty = 0.0

    for obj in objects:
        margin = object_margin(robot_x, robot_y, obj) - robot_radius
        if obj.kind == "bush" and margin < 0.25:
            hit = 0.25 - margin
            penalties["bush"] += hit
            penalty += 0.08 * hit
        elif obj.kind == "pothole" and margin < 0.0:
            hit = -margin
            penalties["pothole"] += hit
            penalty += 1.25 * hit
        elif obj.kind == "person" and margin < 1.2:
            hit = 1.2 - margin
            penalties["person"] += hit
            penalty += 0.35 * hit

    return float(penalty), penalties
