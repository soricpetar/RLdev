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
BEHAVIOR_CHANNELS = ("static_obstacle", "moving_obstacle", "soft_hazard")
POLAR_CHANNELS = BEHAVIOR_CHANNELS
SEMANTIC_CHANNEL_BY_KIND = {"tree": 0, "bush": 1, "pothole": 2, "person": 3, "wall": 4}
BEHAVIOR_CHANNEL_BY_KIND = {"person": 1, "bush": 2, "pothole": 2}


def semantic_channel(kind: str) -> int:
    return SEMANTIC_CHANNEL_BY_KIND.get(kind, 0)


def behavior_channel(kind: str) -> int:
    return BEHAVIOR_CHANNEL_BY_KIND.get(kind, 0)


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
        return float(d - 0.5 * obj.thickness)

    d = np.sqrt((x - obj.x) ** 2 + (y - obj.y) ** 2)
    return float(d - obj.radius)


def object_margin_grid(wx: np.ndarray, wy: np.ndarray, obj: Object) -> np.ndarray:
    if isinstance(obj, WallObstacle):
        return point_segment_distance(wx, wy, obj.x1, obj.y1, obj.x2, obj.y2) - 0.5 * obj.thickness
    return np.sqrt((wx - obj.x) ** 2 + (wy - obj.y) ** 2) - obj.radius


def robot_grid(
    robot_xyh: tuple[float, float, float],
    map_size: int,
    map_extent_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    rx, ry, heading = robot_xyh
    grid = np.linspace(-0.5 * map_extent_m, 0.5 * map_extent_m, map_size, dtype=np.float32)
    gx, gy = np.meshgrid(grid, grid, indexing="xy")
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    return rx + gx * cos_h - gy * sin_h, ry + gx * sin_h + gy * cos_h


def paint_object(layer: np.ndarray, margin: np.ndarray, cost: float, inflation_radius_m: float) -> None:
    layer[:] = np.maximum(layer, (margin <= 0).astype(np.float32) * cost)
    inflated = (margin > 0) & (margin < inflation_radius_m)
    if np.any(inflated):
        inflated_cost = cost * (1.0 - margin / inflation_radius_m)
        layer[:] = np.maximum(layer, np.clip(inflated_cost, 0.0, 1.0).astype(np.float32))


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
    wx, wy = robot_grid(robot_xyh, map_size, map_extent_m)
    costmap = np.zeros((map_size, map_size), dtype=np.float32)
    for obs in obstacles:
        paint_object(costmap, object_margin_grid(wx, wy, obs), object_cost(obs.kind), inflation_radius_m)
    return costmap


def semantic_costmap(
    robot_xyh: tuple[float, float, float],
    objects: list[Object],
    map_size: int,
    map_extent_m: float,
    inflation_radius_m: float,
) -> np.ndarray:
    wx, wy = robot_grid(robot_xyh, map_size, map_extent_m)
    costmap = np.zeros((len(SEMANTIC_CHANNELS), map_size, map_size), dtype=np.float32)
    for obj in objects:
        paint_object(
            costmap[semantic_channel(obj.kind)],
            object_margin_grid(wx, wy, obj),
            object_cost(obj.kind),
            inflation_radius_m,
        )
    return costmap


def behavioral_costmap(
    robot_xyh: tuple[float, float, float],
    objects: list[Object],
    map_size: int,
    map_extent_m: float,
    inflation_radius_m: float,
) -> np.ndarray:
    wx, wy = robot_grid(robot_xyh, map_size, map_extent_m)
    costmap = np.zeros((len(BEHAVIOR_CHANNELS), map_size, map_size), dtype=np.float32)
    for obj in objects:
        paint_object(
            costmap[behavior_channel(obj.kind)],
            object_margin_grid(wx, wy, obj),
            object_cost(obj.kind),
            inflation_radius_m,
        )
    return costmap


def _polar_bin_range(center: float, extent: float, bins: int, lo: float, hi: float) -> tuple[int, int]:
    if hi <= lo:
        return 0, bins - 1
    start = int(np.floor((center - extent - lo) / (hi - lo) * bins))
    end = int(np.ceil((center + extent - lo) / (hi - lo) * bins))
    return max(0, start), min(bins - 1, end)


def _polar_wrap_bins(start: int, end: int, bins: int) -> list[tuple[int, int]]:
    if bins <= 0:
        return []
    if end < start:
        return [(0, bins - 1)]
    if start < 0 or end >= bins:
        return [((start % bins), (end % bins))]
    return [(start, end)]


def polar_costmap(
    robot_xyh: tuple[float, float, float],
    objects: list[Object],
    angle_bins: int,
    distance_bins: int,
    max_distance_m: float,
    inflation_radius_m: float,
) -> np.ndarray:
    rx, ry, heading = robot_xyh
    map_ = np.zeros((len(POLAR_CHANNELS), angle_bins, distance_bins), dtype=np.float32)
    if angle_bins <= 0 or distance_bins <= 0 or max_distance_m <= 0:
        return map_

    def mark_point(px: float, py: float, kind: str, radius: float) -> None:
        dx = px - rx
        dy = py - ry
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        lx = dx * cos_h + dy * sin_h
        ly = -dx * sin_h + dy * cos_h
        dist = float(np.sqrt(lx * lx + ly * ly))
        if dist > max_distance_m:
            return

        angle = float(np.arctan2(ly, lx))
        channel = behavior_channel(kind)
        cost = object_cost(kind)
        radial_extent = max(radius, inflation_radius_m)
        angular_extent = float(np.arctan2(radial_extent, max(dist, 1e-3)))

        a_start = int(np.floor((angle - angular_extent + np.pi) / (2.0 * np.pi) * angle_bins))
        a_end = int(np.ceil((angle + angular_extent + np.pi) / (2.0 * np.pi) * angle_bins))
        d_start = int(np.floor((dist - radial_extent) / max_distance_m * distance_bins))
        d_end = int(np.ceil((dist + radial_extent) / max_distance_m * distance_bins))

        d_start = max(0, d_start)
        d_end = min(distance_bins - 1, d_end)
        if d_end < d_start:
            return

        for a0, a1 in _polar_wrap_bins(a_start, a_end, angle_bins):
            if a1 < a0:
                spans = [(a0, angle_bins - 1), (0, a1)]
            else:
                spans = [(a0, a1)]
            for aa0, aa1 in spans:
                aa0 = max(0, aa0)
                aa1 = min(angle_bins - 1, aa1)
                if aa1 < aa0:
                    continue
                for ab in range(aa0, aa1 + 1):
                    for db in range(d_start, d_end + 1):
                        map_[channel, ab, db] = max(map_[channel, ab, db], cost)

    for obj in objects:
        if isinstance(obj, WallObstacle):
            seg_len = float(np.sqrt((obj.x2 - obj.x1) ** 2 + (obj.y2 - obj.y1) ** 2))
            sample_count = max(3, int(np.ceil(seg_len / 1.5)))
            for i in range(sample_count + 1):
                t = i / max(1, sample_count)
                px = obj.x1 + t * (obj.x2 - obj.x1)
                py = obj.y1 + t * (obj.y2 - obj.y1)
                mark_point(px, py, obj.kind, obj.thickness * 0.5)
        else:
            sample_count = 6 if obj.kind == "tree" else 4
            for i in range(sample_count):
                theta = (2.0 * np.pi * i) / sample_count
                px = obj.x + np.cos(theta) * obj.radius
                py = obj.y + np.sin(theta) * obj.radius
                mark_point(px, py, obj.kind, obj.radius)
            mark_point(obj.x, obj.y, obj.kind, obj.radius)

    return map_


def front_camera_polar_costmap(
    robot_xyh: tuple[float, float, float],
    objects: list[Object],
    angle_bins: int,
    distance_bins: int,
    max_distance_m: float,
    inflation_radius_m: float,
    fov_deg: float,
) -> np.ndarray:
    visible = np.zeros(angle_bins, dtype=bool)
    half_fov = np.deg2rad(fov_deg) * 0.5
    bin_angles = -np.pi + (np.arange(angle_bins, dtype=np.float32) + 0.5) * (2.0 * np.pi / angle_bins)
    visible[np.abs(bin_angles) <= half_fov] = True

    full = polar_costmap(
        robot_xyh,
        objects,
        angle_bins=angle_bins,
        distance_bins=distance_bins,
        max_distance_m=max_distance_m,
        inflation_radius_m=inflation_radius_m,
    )
    front = np.zeros((len(POLAR_CHANNELS) + 1, angle_bins, distance_bins), dtype=np.float32)
    front[: len(POLAR_CHANNELS), visible, :] = full[:, visible, :]
    front[len(POLAR_CHANNELS), ~visible, :] = 1.0
    return front


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
