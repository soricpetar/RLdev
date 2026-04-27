from __future__ import annotations

from collections import OrderedDict
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback for constrained environments
    from . import spaces as spaces  # type: ignore

    class _BaseEnv:
        def reset(self, *, seed=None):
            if seed is not None:
                np.random.seed(seed)

    class _FallbackGym:
        Env = _BaseEnv

    gym = _FallbackGym()  # type: ignore

from .maps import (
    BEHAVIOR_CHANNELS,
    behavioral_costmap,
    COLLIDABLE_KINDS,
    collision_distance,
    move_dynamic_objects,
    object_margin,
    obstacle_costmap,
    random_field_objects,
    random_obstacles,
    polar_costmap,
    semantic_contact_penalty,
    semantic_costmap,
    SEMANTIC_CHANNELS,
)
from .rewards import RewardConfig, compute_reward
from .robot import RobotState, step_kinematics, wrap_to_pi


class FieldNavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}
    costmap_channels = SEMANTIC_CHANNELS

    def __init__(
        self,
        map_size: int = 64,
        map_extent_m: float = 12.0,
        polar_observation: bool = False,
        polar_angle_bins: int = 32,
        polar_distance_bins: int = 16,
        polar_max_distance_m: float = 24.0,
        behavioral_observation: bool = False,
        foveated_observation: bool = False,
        local_map_size: int = 32,
        global_map_size: int = 32,
        local_map_extent_m: float = 6.0,
        global_map_extent_m: float = 18.0,
        world_size_m: float = 50.0,
        max_steps: int = 500,
        dt: float = 0.2,
        fixed_speed_mps: float = 1.0,
        max_turn_rate_rps: float = 1.0,
        num_obstacles_range: tuple[int, int] = (14, 40),
        obstacle_radius_range_m: tuple[float, float] = (0.35, 1.6),
        tree_rows_range: tuple[int, int] = (2, 4),
        bushes_range: tuple[int, int] = (6, 14),
        potholes_range: tuple[int, int] = (4, 10),
        people_range: tuple[int, int] = (2, 6),
        walls_range: tuple[int, int] = (1, 4),
        curriculum_enabled: bool = False,
        curriculum_warmup_steps: int = 0,
        min_goal_distance_m: float = 10.0,
        max_goal_distance_m: float = 22.0,
        goal_tolerance_m: float = 0.65,
        robot_radius_m: float = 0.35,
        inflation_radius_m: float = 1.0,
        near_obstacle_threshold_m: float = 2.0,
        reset_start_clearance_m: float = 0.0,
        reset_goal_clearance_m: float = 0.0,
        reset_forward_clearance_m: float = 0.0,
        reset_forward_margin_m: float = 0.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.map_size = map_size
        self.map_extent_m = map_extent_m
        self.polar_observation = polar_observation
        self.polar_angle_bins = polar_angle_bins
        self.polar_distance_bins = polar_distance_bins
        self.polar_max_distance_m = polar_max_distance_m
        self.behavioral_observation = behavioral_observation
        self.foveated_observation = foveated_observation
        self.local_map_size = local_map_size
        self.global_map_size = global_map_size
        self.local_map_extent_m = local_map_extent_m
        self.global_map_extent_m = global_map_extent_m
        self.world_size_m = world_size_m
        self.max_steps = max_steps
        self.dt = dt
        self.fixed_speed_mps = fixed_speed_mps
        self.max_turn_rate_rps = max_turn_rate_rps
        self.num_obstacles_range = num_obstacles_range
        self.obstacle_radius_range_m = obstacle_radius_range_m
        self.tree_rows_range = tree_rows_range
        self.bushes_range = bushes_range
        self.potholes_range = potholes_range
        self.people_range = people_range
        self.walls_range = walls_range
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_warmup_steps = curriculum_warmup_steps
        self._lifetime_steps = 0
        self.min_goal_distance_m = min_goal_distance_m
        self.max_goal_distance_m = max_goal_distance_m
        self.goal_tolerance_m = goal_tolerance_m
        self.robot_radius_m = robot_radius_m
        self.inflation_radius_m = inflation_radius_m
        self.near_obstacle_threshold_m = near_obstacle_threshold_m
        self.reset_start_clearance_m = reset_start_clearance_m
        self.reset_goal_clearance_m = reset_goal_clearance_m
        self.reset_forward_clearance_m = reset_forward_clearance_m
        self.reset_forward_margin_m = reset_forward_margin_m
        self.render_mode = render_mode

        self.reward_cfg = RewardConfig()
        self._prev_steering = 0.0
        self._steps = 0

        self.steering_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        if self.polar_observation:
            self.observation_space = spaces.Dict(
                OrderedDict(
                    {
                        "costmap": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(len(BEHAVIOR_CHANNELS), polar_angle_bins, polar_distance_bins),
                            dtype=np.float32,
                        ),
                        "goal": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                        "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                    }
                )
            )
        elif self.foveated_observation:
            self.observation_space = spaces.Dict(
                OrderedDict(
                    {
                        "local_costmap": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(len(BEHAVIOR_CHANNELS), local_map_size, local_map_size),
                            dtype=np.float32,
                        ),
                        "global_costmap": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(len(BEHAVIOR_CHANNELS), global_map_size, global_map_size),
                            dtype=np.float32,
                        ),
                        "goal": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                        "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                    }
                )
            )
        elif self.behavioral_observation:
            self.observation_space = spaces.Dict(
                OrderedDict(
                    {
                        "costmap": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(len(BEHAVIOR_CHANNELS), map_size, map_size),
                            dtype=np.float32,
                        ),
                        "goal": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                        "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                    }
                )
            )
        else:
            self.observation_space = spaces.Dict(
                OrderedDict(
                    {
                        "costmap": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(len(self.costmap_channels), map_size, map_size),
                            dtype=np.float32,
                        ),
                        "goal": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                        "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                    }
                )
            )
        self.action_space = spaces.Discrete(len(self.steering_values))

        self._rng = np.random.default_rng()
        self.robot = RobotState(0.0, 0.0, 0.0, 0.0, 0.0)
        self.goal_xy = np.zeros(2, dtype=np.float32)
        self.obstacles = []
        self.object_contact = {"bush": 0.0, "pothole": 0.0, "person": 0.0}
        self._prev_goal_distance = np.inf

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._steps = 0
        self._prev_steering = 0.0

        self.robot = RobotState(
            x=float(self._rng.uniform(-3.0, 3.0)),
            y=float(self._rng.uniform(-3.0, 3.0)),
            heading=float(self._rng.uniform(-np.pi, np.pi)),
            speed=self.fixed_speed_mps,
            yaw_rate=0.0,
        )

        progress = self._curriculum_progress()
        min_goal_distance = self._curriculum_float(progress, 6.0, self.min_goal_distance_m)
        max_goal_distance = self._curriculum_float(progress, 12.0, self.max_goal_distance_m)
        if max_goal_distance < min_goal_distance:
            max_goal_distance = min_goal_distance

        self.goal_xy = self._sample_goal_far_from_robot(
            min_dist=min_goal_distance,
            max_dist=max_goal_distance,
        )

        for _ in range(64):
            if options and options.get("legacy_obstacles"):
                nobs = int(self._rng.integers(self.num_obstacles_range[0], self.num_obstacles_range[1] + 1))
                self.obstacles = random_obstacles(
                    self._rng,
                    nobs,
                    self.world_size_m,
                    min_radius=self.obstacle_radius_range_m[0],
                    max_radius=self.obstacle_radius_range_m[1],
                )
            else:
                self.obstacles = random_field_objects(
                    self._rng,
                    self.world_size_m,
                    tree_rows_range=self._curriculum_range(progress, (1, 2), self.tree_rows_range),
                    bushes_range=self._curriculum_range(progress, (3, 8), self.bushes_range),
                    potholes_range=self._curriculum_range(progress, (1, 4), self.potholes_range),
                    people_range=self._curriculum_range(progress, (0, 2), self.people_range),
                    walls_range=self._curriculum_range(progress, (0, 1), self.walls_range),
                )
            if self._reset_layout_valid():
                break
        self.object_contact = {"bush": 0.0, "pothole": 0.0, "person": 0.0}

        self._prev_goal_distance = self._goal_distance()
        obs = self._observation()
        info = self._info(collision=False, goal_reached=False)
        return obs, info

    def step(self, action: int):
        steering = float(self.steering_values[int(action)])
        self.robot = step_kinematics(
            self.robot,
            steering_cmd=steering,
            dt=self.dt,
            speed=self.fixed_speed_mps,
            max_turn_rate=self.max_turn_rate_rps,
        )
        self._steps += 1
        self._lifetime_steps += 1
        move_dynamic_objects(self.obstacles, self.dt, self.world_size_m)

        goal_distance = self._goal_distance()
        progress = self._prev_goal_distance - goal_distance
        self._prev_goal_distance = goal_distance

        nearest_margin = collision_distance(self.robot.x, self.robot.y, self.obstacles) - self.robot_radius_m
        collision = nearest_margin <= 0.0
        goal_reached = goal_distance <= self.goal_tolerance_m
        out_of_bounds = self._out_of_bounds()
        semantic_penalty, object_contact = semantic_contact_penalty(
            self.robot.x,
            self.robot.y,
            self.robot_radius_m,
            self.obstacles,
        )

        reward = compute_reward(
            progress=progress,
            goal_reached=goal_reached,
            collision=collision,
            nearest_obstacle_distance=nearest_margin,
            near_distance_threshold=self.near_obstacle_threshold_m,
            steering_delta=steering - self._prev_steering,
            cfg=self.reward_cfg,
        )
        reward -= semantic_penalty
        self.object_contact = object_contact
        self._prev_steering = steering

        terminated = bool(collision or goal_reached or out_of_bounds)
        truncated = self._steps >= self.max_steps

        if out_of_bounds:
            reward -= 10.0

        obs = self._observation()
        info = self._info(collision=collision, goal_reached=goal_reached)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        dist = self._goal_distance()
        print(
            f"step={self._steps} pos=({self.robot.x:.2f},{self.robot.y:.2f}) "
            f"heading={self.robot.heading:.2f} goal_dist={dist:.2f} "
            f"contacts={self.object_contact}"
        )

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal_xy - np.array([self.robot.x, self.robot.y], dtype=np.float32)))

    def _curriculum_progress(self) -> float:
        if not self.curriculum_enabled or self.curriculum_warmup_steps <= 0:
            return 1.0
        return float(np.clip(self._lifetime_steps / max(1, self.curriculum_warmup_steps), 0.0, 1.0))

    @staticmethod
    def _curriculum_float(progress: float, start: float, end: float) -> float:
        return float(start + (end - start) * progress)

    @staticmethod
    def _curriculum_range(progress: float, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int]:
        lo = int(round(start[0] + (end[0] - start[0]) * progress))
        hi = int(round(start[1] + (end[1] - start[1]) * progress))
        return lo, max(lo, hi)

    def _goal_features(self) -> np.ndarray:
        dx = self.goal_xy[0] - self.robot.x
        dy = self.goal_xy[1] - self.robot.y
        dist = float(np.sqrt(dx * dx + dy * dy))
        goal_heading = np.arctan2(dy, dx)
        heading_error = wrap_to_pi(goal_heading - self.robot.heading)

        dist_norm = np.clip(dist / 25.0, 0.0, 1.0)
        return np.array([dist_norm, np.sin(heading_error), np.cos(heading_error)], dtype=np.float32)

    def _state_features(self) -> np.ndarray:
        return np.array(
            [
                np.clip(self.robot.speed / 3.0, -1.0, 1.0),
                np.clip(self.robot.yaw_rate / self.max_turn_rate_rps, -1.0, 1.0),
                np.clip(self._prev_steering, -1.0, 1.0),
                1.0,
            ],
            dtype=np.float32,
        )

    def _observation(self) -> dict[str, np.ndarray]:
        if self.polar_observation:
            return {
                "costmap": polar_costmap(
                    (self.robot.x, self.robot.y, self.robot.heading),
                    self.obstacles,
                    angle_bins=self.polar_angle_bins,
                    distance_bins=self.polar_distance_bins,
                    max_distance_m=self.polar_max_distance_m,
                    inflation_radius_m=self.inflation_radius_m,
                ),
                "goal": self._goal_features(),
                "state": self._state_features(),
            }

        if self.foveated_observation:
            robot_xyh = (self.robot.x, self.robot.y, self.robot.heading)
            return {
                "local_costmap": behavioral_costmap(
                    robot_xyh,
                    self.obstacles,
                    map_size=self.local_map_size,
                    map_extent_m=self.local_map_extent_m,
                    inflation_radius_m=self.inflation_radius_m,
                ),
                "global_costmap": behavioral_costmap(
                    robot_xyh,
                    self.obstacles,
                    map_size=self.global_map_size,
                    map_extent_m=self.global_map_extent_m,
                    inflation_radius_m=self.inflation_radius_m,
                ),
                "goal": self._goal_features(),
                "state": self._state_features(),
            }

        if self.behavioral_observation:
            return {
                "costmap": behavioral_costmap(
                    (self.robot.x, self.robot.y, self.robot.heading),
                    self.obstacles,
                    map_size=self.map_size,
                    map_extent_m=self.map_extent_m,
                    inflation_radius_m=self.inflation_radius_m,
                ),
                "goal": self._goal_features(),
                "state": self._state_features(),
            }

        return {
            "costmap": semantic_costmap(
                (self.robot.x, self.robot.y, self.robot.heading),
                self.obstacles,
                map_size=self.map_size,
                map_extent_m=self.map_extent_m,
                inflation_radius_m=self.inflation_radius_m,
            ),
            "goal": self._goal_features(),
            "state": self._state_features(),
        }

    def _out_of_bounds(self) -> bool:
        half = self.world_size_m / 2.0
        return abs(self.robot.x) > half or abs(self.robot.y) > half

    def _point_collidable_clearance(self, x: float, y: float) -> float:
        min_clearance = np.inf
        for obstacle in self.obstacles:
            if obstacle.kind not in COLLIDABLE_KINDS:
                continue
            clearance = object_margin(x, y, obstacle) - self.robot_radius_m
            min_clearance = min(min_clearance, clearance)
        return float(min_clearance)

    def _forward_path_clear(self) -> bool:
        if self.reset_forward_clearance_m <= 0.0:
            return True

        sample_count = max(2, int(np.ceil(self.reset_forward_clearance_m / max(1e-6, self.fixed_speed_mps * self.dt))))
        cos_h = float(np.cos(self.robot.heading))
        sin_h = float(np.sin(self.robot.heading))
        for idx in range(1, sample_count + 1):
            distance = self.reset_forward_clearance_m * idx / sample_count
            x = self.robot.x + cos_h * distance
            y = self.robot.y + sin_h * distance
            if self._point_collidable_clearance(x, y) < self.reset_forward_margin_m:
                return False
        return True

    def _reset_layout_valid(self) -> bool:
        if (
            self.reset_start_clearance_m > 0.0
            and self._point_collidable_clearance(self.robot.x, self.robot.y) < self.reset_start_clearance_m
        ):
            return False
        if (
            self.reset_goal_clearance_m > 0.0
            and self._point_collidable_clearance(float(self.goal_xy[0]), float(self.goal_xy[1])) < self.reset_goal_clearance_m
        ):
            return False
        return self._forward_path_clear()

    def _sample_goal_far_from_robot(self, min_dist: float, max_dist: float) -> np.ndarray:
        for _ in range(200):
            gx, gy = self._rng.uniform(-self.world_size_m * 0.4, self.world_size_m * 0.4, size=2)
            d = np.linalg.norm(np.array([gx - self.robot.x, gy - self.robot.y]))
            if min_dist <= d <= max_dist:
                return np.array([gx, gy], dtype=np.float32)

        return np.array([self.robot.x + max_dist, self.robot.y], dtype=np.float32)

    def _info(self, collision: bool, goal_reached: bool) -> dict:
        return {
            "goal_distance": self._goal_distance(),
            "collision": collision,
            "goal_reached": goal_reached,
            "steps": self._steps,
            "object_counts": self._object_counts(),
            "object_contact": dict(self.object_contact),
        }

    def _object_counts(self) -> dict[str, int]:
        counts = {"tree": 0, "bush": 0, "pothole": 0, "person": 0, "wall": 0}
        for obj in self.obstacles:
            counts[obj.kind] = counts.get(obj.kind, 0) + 1
        return counts
