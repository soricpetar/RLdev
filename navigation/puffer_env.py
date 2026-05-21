from __future__ import annotations

import ctypes
from collections import deque

import numpy as np

from navigation.nav_env import FieldNavEnv
import pufferlib.emulation


class FieldNavVec:
    obs_dtype = "FloatTensor"
    obs_elem_size = 4
    num_atns = 1
    gpu = 0
    int_keys = {
        "map_size",
        "local_map_size",
        "global_map_size",
        "polar_angle_bins",
        "polar_distance_bins",
        "max_steps",
        "curriculum_warmup_steps",
    }
    float_keys = {
        "map_extent_m",
        "local_map_extent_m",
        "global_map_extent_m",
        "polar_max_distance_m",
        "front_camera_fov_deg",
        "heading_alignment_scale",
        "goal_outside_fov_penalty",
        "world_size_m",
        "dt",
        "fixed_speed_mps",
        "min_speed_mps",
        "max_speed_mps",
        "acceleration_mps2",
        "brake_deceleration_mps2",
        "coast_deceleration_mps2",
        "max_turn_rate_rps",
        "min_goal_distance_m",
        "max_goal_distance_m",
        "goal_tolerance_m",
        "robot_radius_m",
        "inflation_radius_m",
        "near_obstacle_threshold_m",
        "reset_start_clearance_m",
        "reset_goal_clearance_m",
        "reset_forward_clearance_m",
        "reset_forward_margin_m",
    }
    int_range_keys = {
        "num_obstacles_range",
        "tree_rows_range",
        "bushes_range",
        "potholes_range",
        "people_range",
        "walls_range",
    }
    float_range_keys = {"obstacle_radius_range_m"}
    bool_keys = {
        "polar_observation",
        "front_camera_observation",
        "speed_control",
        "behavioral_observation",
        "foveated_observation",
        "curriculum_enabled",
    }

    def __init__(self, args):
        vec_cfg = args["vec"]
        env_cfg = dict(args.get("env", {}))
        self.total_agents = int(vec_cfg["total_agents"])
        self.seed = int(args.get("seed", args.get("train", {}).get("seed", 0)))

        self.envs = []
        for _ in range(self.total_agents):
            env_kwargs = self._env_kwargs(env_cfg)
            raw_env = FieldNavEnv(**env_kwargs)
            self.envs.append(pufferlib.emulation.GymnasiumPufferEnv(raw_env))

        sample_obs, _ = self.envs[0].reset(seed=self.seed)
        self.obs_size = int(sample_obs.shape[0])
        self.act_sizes = [int(self.envs[0].action_space.n)]
        self.observations = np.zeros((self.total_agents, self.obs_size), dtype=np.float32)
        self.rewards = np.zeros(self.total_agents, dtype=np.float32)
        self.terminals = np.zeros(self.total_agents, dtype=np.float32)

        self.episode_returns = np.zeros(self.total_agents, dtype=np.float32)
        self.episode_lengths = np.zeros(self.total_agents, dtype=np.int32)
        self.finished_returns = deque(maxlen=256)
        self.finished_lengths = deque(maxlen=256)
        self.finished_success = deque(maxlen=256)
        self.finished_collision = deque(maxlen=256)
        self.episodes_finished = 0

    @staticmethod
    def _range(value, cast):
        if isinstance(value, str):
            value = value.strip("()[]").split(",")
        return tuple(cast(v) for v in value)

    def _env_kwargs(self, cfg):
        converters = {
            **dict.fromkeys(self.int_keys, int),
            **dict.fromkeys(self.float_keys, float),
            **dict.fromkeys(self.bool_keys, lambda value: str(value).lower() in {"1", "true", "yes", "on"}),
            **dict.fromkeys(self.int_range_keys, lambda value: self._range(value, int)),
            **dict.fromkeys(self.float_range_keys, lambda value: self._range(value, float)),
        }
        return {key: converters[key](value) for key, value in cfg.items() if key in converters}

    @property
    def obs_ptr(self):
        return int(self.observations.ctypes.data)

    @property
    def rewards_ptr(self):
        return int(self.rewards.ctypes.data)

    @property
    def terminals_ptr(self):
        return int(self.terminals.ctypes.data)

    def reset(self):
        self.rewards.fill(0.0)
        self.terminals.fill(0.0)
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)
        for idx, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self.seed + idx)
            self.observations[idx] = obs

    def cpu_step(self, actions_ptr):
        raw_actions = (ctypes.c_float * (self.total_agents * self.num_atns)).from_address(actions_ptr)
        actions = np.ctypeslib.as_array(raw_actions).reshape(self.total_agents, self.num_atns)

        for idx, env in enumerate(self.envs):
            action = int(actions[idx, 0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            self.rewards[idx] = float(reward)
            self.terminals[idx] = float(done)
            self.episode_returns[idx] += float(reward)
            self.episode_lengths[idx] += 1

            if done:
                self.finished_returns.append(float(self.episode_returns[idx]))
                self.finished_lengths.append(float(self.episode_lengths[idx]))
                self.finished_success.append(float(info.get("goal_reached", False)))
                self.finished_collision.append(float(info.get("collision", False)))
                self.episodes_finished += 1
                self.episode_returns[idx] = 0.0
                self.episode_lengths[idx] = 0
                obs, _ = env.reset()

            self.observations[idx] = obs

    def log(self):
        returns, lengths, successes, collisions = (
            np.asarray(values, dtype=np.float32)
            for values in (
                self.finished_returns,
                self.finished_lengths,
                self.finished_success,
                self.finished_collision,
            )
        )
        curriculum = np.asarray(
            [env.env._curriculum_progress() for env in self.envs],
            dtype=np.float32,
        )
        return {
            "score": float(returns.mean()) if returns.size else 0.0,
            "episode_length": float(lengths.mean()) if lengths.size else 0.0,
            "success_rate": float(successes.mean()) if successes.size else 0.0,
            "collision_rate": float(collisions.mean()) if collisions.size else 0.0,
            "curriculum_progress": float(curriculum.mean()) if curriculum.size else 1.0,
            "n": float(self.episodes_finished),
        }

    def render(self, env_id=0):
        env_id = int(np.clip(env_id, 0, self.total_agents - 1))
        raw_env = self.envs[env_id].env
        print(
            f"field_nav env={env_id} step={raw_env._steps} "
            f"goal_distance={raw_env._info()['goal_distance']:.3f}"
        )

    def close(self):
        self.envs.clear()


def create_vec(args):
    return FieldNavVec(args)
