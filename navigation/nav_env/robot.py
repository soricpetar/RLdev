from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RobotState:
    x: float
    y: float
    heading: float
    speed: float
    yaw_rate: float


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def step_kinematics(
    state: RobotState,
    steering_cmd: float,
    dt: float,
    speed: float,
    max_turn_rate: float,
) -> RobotState:
    yaw_rate = float(steering_cmd) * max_turn_rate
    heading = wrap_to_pi(state.heading + yaw_rate * dt)

    x = state.x + speed * np.cos(heading) * dt
    y = state.y + speed * np.sin(heading) * dt

    return RobotState(x=x, y=y, heading=heading, speed=speed, yaw_rate=yaw_rate)
