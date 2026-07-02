from __future__ import annotations

import numpy as np

from .spec import Stage49ObservationSpec


def _bin_span(center: float, extent: float, bins: int, lo: float, hi: float) -> tuple[int, int]:
    start = int(np.floor((center - extent - lo) / (hi - lo) * bins))
    end = int(np.ceil((center + extent - lo) / (hi - lo) * bins))
    return start, end


def points_to_costmap(
    points_robot_xyz: np.ndarray,
    class_names: np.ndarray,
    spec: Stage49ObservationSpec | None = None,
) -> np.ndarray:
    spec = spec or Stage49ObservationSpec()
    costmap = np.zeros((spec.channels, spec.angle_bins, spec.distance_bins), dtype=np.float32)

    visible = spec.visible_angle_mask()
    costmap[3, ~visible, :] = 1.0

    points = np.asarray(points_robot_xyz, dtype=np.float32)
    classes = np.asarray(class_names, dtype=object)
    if points.size == 0:
        return costmap
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_robot_xyz must have shape (N, 3)")
    if len(classes) != len(points):
        raise ValueError("class_names length must match points")

    x = points[:, 0]
    y = points[:, 1]
    dist = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)

    valid = (x > 0.0) & (dist > 0.0) & (dist <= spec.max_distance_m)
    if not np.any(valid):
        return costmap

    valid_indices = np.nonzero(valid)[0]

    for src_idx in valid_indices:
        d = float(dist[src_idx])
        a = float(angle[src_idx])
        radial_extent = spec.inflation_radius_m
        angular_extent = float(np.arctan2(radial_extent, max(d, 1e-3)))
        a_start, a_end = _bin_span(a, angular_extent, spec.angle_bins, -np.pi, np.pi)
        d_start, d_end = _bin_span(d, radial_extent, spec.distance_bins, 0.0, spec.max_distance_m)
        d_start = max(0, d_start)
        d_end = min(spec.distance_bins - 1, d_end)
        if d_end < d_start:
            continue
        channel, cost = spec.class_to_channel_and_cost(classes[src_idx])
        for ab in range(max(0, a_start), min(spec.angle_bins - 1, a_end) + 1):
            if not visible[ab]:
                continue
            costmap[channel, ab, d_start : d_end + 1] = np.maximum(
                costmap[channel, ab, d_start : d_end + 1],
                np.float32(cost),
            )

    return costmap
