from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navigation.nav_env import FieldNavEnv


def main():
    parser = argparse.ArgumentParser(description="Run random rollouts in FieldNavEnv")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = FieldNavEnv(max_steps=args.max_steps)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(
            f"episode={ep} total_reward={total_reward:.3f} "
            f"steps={info['steps']} goal_distance={info['goal_distance']:.2f} "
            f"goal_reached={info['goal_reached']} collision={info['collision']}"
        )


if __name__ == "__main__":
    main()
