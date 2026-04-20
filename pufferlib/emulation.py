from __future__ import annotations

import numpy as np


class GymnasiumPufferEnv:
    """Small compatibility wrapper used for local examples/tests.

    Flattens Dict observations into a single float32 vector.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._obs_spec = self._compute_obs_spec(env.observation_space)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def _compute_obs_spec(self, observation_space):
        if not hasattr(observation_space, "spaces"):
            shape = observation_space.shape
            size = int(np.prod(shape))
            return [("obs", size, shape)]

        spec = []
        for key, space in observation_space.spaces.items():
            size = int(np.prod(space.shape))
            spec.append((key, size, space.shape))
        return spec

    def _flatten_obs(self, obs):
        if not isinstance(obs, dict):
            return np.asarray(obs, dtype=np.float32).reshape(-1)

        parts = []
        for key, _, _ in self._obs_spec:
            parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=0)
