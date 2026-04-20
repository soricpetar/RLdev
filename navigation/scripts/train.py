from __future__ import annotations

"""Minimal training entrypoint wiring FieldNavEnv into PufferLib emulation.

This is intentionally lightweight so you can swap in your preferred PPO trainer.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navigation.nav_env import FieldNavEnv
import pufferlib.emulation


def make_env(seed: int | None = None):
    env = FieldNavEnv()
    if seed is not None:
        env.reset(seed=seed)
    return pufferlib.emulation.GymnasiumPufferEnv(env)


if __name__ == "__main__":
    env = make_env(seed=0)
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("PufferLib wrapper smoke test")
    print("obs_shape", obs.shape)
    print("reward", reward, "terminated", terminated, "truncated", truncated)
