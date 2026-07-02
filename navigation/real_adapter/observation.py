from __future__ import annotations

import numpy as np

from .spec import Stage49ObservationSpec


def make_stage49_observation(
    costmap: np.ndarray,
    goal_distance_m: float,
    goal_heading_error_rad: float,
    speed_mps: float,
    yaw_rate_rps: float,
    previous_steering: float,
    spec: Stage49ObservationSpec | None = None,
) -> np.ndarray:
    spec = spec or Stage49ObservationSpec()
    cm = np.asarray(costmap, dtype=np.float32)
    expected_shape = (spec.channels, spec.angle_bins, spec.distance_bins)
    if cm.shape != expected_shape:
        raise ValueError(f"costmap must have shape {expected_shape}, got {cm.shape}")

    goal = np.asarray(
        [
            np.clip(float(goal_distance_m) / spec.goal_norm_distance_m, 0.0, 1.0),
            np.sin(float(goal_heading_error_rad)),
            np.cos(float(goal_heading_error_rad)),
        ],
        dtype=np.float32,
    )
    state = np.asarray(
        [
            np.clip(float(speed_mps) / spec.speed_norm_mps, -1.0, 1.0),
            np.clip(float(yaw_rate_rps) / spec.max_turn_rate_rps, -1.0, 1.0),
            np.clip(float(previous_steering), -1.0, 1.0),
            1.0,
        ],
        dtype=np.float32,
    )
    obs = np.concatenate([cm.reshape(-1), goal, state], axis=0).astype(np.float32, copy=False)
    if obs.shape != (spec.flat_size,):
        raise RuntimeError(f"adapter produced observation shape {obs.shape}, expected {(spec.flat_size,)}")
    return obs
