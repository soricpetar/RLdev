from __future__ import annotations

import numpy as np

from navigation.nav_env.maps import CircularObstacle, WallObstacle


def objects_to_adapter_points(
    robot_xyh: tuple[float, float, float],
    objects: list[object],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert simulator object centers/samples to adapter robot-frame points.

    This helper is for tests/debug parity, not real deployment. For point-like
    zero-radius circular objects it should match the simulator's polar binning.
    """
    rx, ry, heading = robot_xyh
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    points: list[tuple[float, float, float]] = []
    classes: list[str] = []

    def add_world_point(px: float, py: float, kind: str) -> None:
        dx = float(px) - rx
        dy = float(py) - ry
        lx = dx * cos_h + dy * sin_h
        ly = -dx * sin_h + dy * cos_h
        points.append((float(lx), float(ly), 0.0))
        classes.append(kind)

    for obj in objects:
        if isinstance(obj, WallObstacle):
            seg_len = float(np.sqrt((obj.x2 - obj.x1) ** 2 + (obj.y2 - obj.y1) ** 2))
            sample_count = max(3, int(np.ceil(seg_len / 1.5)))
            for i in range(sample_count + 1):
                t = i / max(1, sample_count)
                add_world_point(obj.x1 + t * (obj.x2 - obj.x1), obj.y1 + t * (obj.y2 - obj.y1), obj.kind)
        elif isinstance(obj, CircularObstacle):
            add_world_point(obj.x, obj.y, obj.kind)
        else:
            raise TypeError(f"unsupported object type: {type(obj)!r}")

    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.asarray([], dtype=object)
    return np.asarray(points, dtype=np.float32), np.asarray(classes, dtype=object)
