from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def depth_classes_to_robot_points(
    depth_m: np.ndarray,
    class_map: np.ndarray,
    intrinsics: Mapping[str, float],
    camera_to_robot: np.ndarray,
    *,
    max_distance_m: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project aligned depth/class images into robot-frame labeled points.

    Convention for this first adapter contract: camera coordinates are x-forward,
    y-left, z-up before applying `camera_to_robot`. If a real camera uses the
    common z-forward optical frame, supply the calibrated transform that converts
    optical coordinates into this robot frame.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    classes = np.asarray(class_map, dtype=object)
    transform = np.asarray(camera_to_robot, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if classes.shape != depth.shape:
        raise ValueError("class_map shape must match depth_m shape")
    if transform.shape != (4, 4):
        raise ValueError("camera_to_robot must have shape (4, 4)")

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("fx and fy must be non-zero")

    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0.0) & (depth <= max_distance_m))
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.asarray([], dtype=object)

    z = depth[rows, cols]
    # x-forward/y-left/z-up pinhole convention. Pixel rows increase downward, so
    # vertical image offset is negated to produce z-up.
    x = z
    y = -((cols.astype(np.float32) - cx) / fx) * z
    z_up = -((rows.astype(np.float32) - cy) / fy) * z
    camera_points = np.stack([x, y, z_up, np.ones_like(x)], axis=1).astype(np.float32)
    robot_points_h = camera_points @ transform.T
    robot_points = robot_points_h[:, :3].astype(np.float32, copy=False)
    point_classes = classes[rows, cols]
    return robot_points, point_classes
