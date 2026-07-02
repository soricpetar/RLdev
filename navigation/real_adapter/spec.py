from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Stage49ObservationSpec:
    channels: int = 4
    angle_bins: int = 32
    distance_bins: int = 16
    max_distance_m: float = 24.0
    front_camera_fov_deg: float = 100.0
    inflation_radius_m: float = 1.0
    goal_norm_distance_m: float = 25.0
    speed_norm_mps: float = 3.0
    max_turn_rate_rps: float = 1.0

    @property
    def costmap_size(self) -> int:
        return self.channels * self.angle_bins * self.distance_bins

    @property
    def vector_size(self) -> int:
        return 7

    @property
    def flat_size(self) -> int:
        return self.costmap_size + self.vector_size

    @property
    def distance_bin_width_m(self) -> float:
        return self.max_distance_m / self.distance_bins

    def angle_centers_rad(self) -> np.ndarray:
        return -np.pi + (np.arange(self.angle_bins, dtype=np.float32) + 0.5) * (2.0 * np.pi / self.angle_bins)

    def visible_angle_mask(self) -> np.ndarray:
        half_fov = np.deg2rad(self.front_camera_fov_deg) * 0.5
        return np.abs(self.angle_centers_rad()) <= half_fov

    def class_to_channel_and_cost(self, class_name: str) -> tuple[int, float]:
        key = str(class_name).lower()
        if key in {"person", "human", "pedestrian", "cyclist", "animal"}:
            return 1, 1.0
        if key in {"bush", "grass", "vegetation", "soft_vegetation"}:
            return 2, 0.42
        if key in {"pothole", "hole", "curb", "ditch", "stair", "drop"}:
            return 2, 0.65
        return 0, 1.0
