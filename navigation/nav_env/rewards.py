from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    goal_bonus: float = 20.0
    collision_penalty: float = 20.0
    near_obstacle_scale: float = 0.2
    smoothness_scale: float = 0.02
    time_penalty: float = 0.01


def compute_reward(
    progress: float,
    goal_reached: bool,
    collision: bool,
    nearest_obstacle_distance: float,
    near_distance_threshold: float,
    steering_delta: float,
    cfg: RewardConfig,
) -> float:
    reward = progress

    if goal_reached:
        reward += cfg.goal_bonus
    if collision:
        reward -= cfg.collision_penalty

    if nearest_obstacle_distance < near_distance_threshold:
        reward -= cfg.near_obstacle_scale * (near_distance_threshold - nearest_obstacle_distance)

    reward -= cfg.smoothness_scale * abs(steering_delta)
    reward -= cfg.time_penalty
    return float(reward)
