from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navigation.nav_env import FieldNavEnv
from navigation.nav_env.maps import CircularObstacle, WallObstacle
from navigation.nav_env.robot import RobotState
import pufferlib.emulation
from navigation.scripts.render_policy import (
    EpisodeTrace,
    closest_collidable_margin,
    load_policy,
    save_gif,
    save_grid,
    save_summary,
    select_failure_traces,
)


def make_env(model: torch.nn.Module, max_steps: int, fov_deg: float) -> tuple[FieldNavEnv, pufferlib.emulation.GymnasiumPufferEnv]:
    raw_env = FieldNavEnv(
        max_steps=max_steps,
        front_camera_observation=True,
        front_camera_fov_deg=fov_deg,
        polar_max_distance_m=24.0,
        heading_alignment_scale=0.06,
        goal_outside_fov_penalty=0.02,
        fixed_speed_mps=1.0,
        max_speed_mps=1.0,
        speed_control=int(getattr(model, "field_nav_action_size", 5)) > 5,
        min_goal_distance_m=10.0,
        max_goal_distance_m=22.0,
        goal_tolerance_m=0.65,
        inflation_radius_m=1.0,
        near_obstacle_threshold_m=2.0,
    )
    return raw_env, pufferlib.emulation.GymnasiumPufferEnv(raw_env)


def install_blocked_corridor(raw_env: FieldNavEnv, rng: np.random.Generator, variant: str) -> None:
    heading_jitter = float(rng.uniform(-0.18, 0.18))
    goal_distance = float(rng.uniform(16.0, 22.0))
    block_distance = float(rng.uniform(6.0, 11.0))
    half_width = float(rng.uniform(2.6, 4.6))
    lateral_shift = float(rng.uniform(-0.8, 0.8))

    raw_env.robot = RobotState(0.0, 0.0, heading_jitter, raw_env.fixed_speed_mps, 0.0)
    raw_env.goal_xy = np.asarray([goal_distance, float(rng.uniform(-0.7, 0.7))], dtype=np.float32)

    objects = []
    if variant == "wall":
        objects.append(
            WallObstacle(
                x1=block_distance,
                y1=-half_width + lateral_shift,
                x2=block_distance,
                y2=half_width + lateral_shift,
                thickness=0.55,
                kind="wall",
            )
        )
    elif variant == "trees":
        spacing = float(rng.uniform(1.05, 1.35))
        ys = np.arange(-half_width, half_width + 0.001, spacing) + lateral_shift
        for y in ys:
            objects.append(CircularObstacle(block_distance + float(rng.normal(0.0, 0.15)), float(y), 0.55, "tree"))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Add a few off-corridor distractors while keeping a real passage around the block.
    for _ in range(int(rng.integers(2, 5))):
        objects.append(
            CircularObstacle(
                float(rng.uniform(4.0, 18.0)),
                float(rng.choice([-1.0, 1.0]) * rng.uniform(5.5, 9.0)),
                float(rng.uniform(0.35, 0.7)),
                "tree",
            )
        )

    raw_env.obstacles = objects
    raw_env.object_contact = {"bush": 0.0, "pothole": 0.0, "person": 0.0}
    raw_env._steps = 0
    raw_env._prev_steering = 0.0
    raw_env._prev_goal_distance = raw_env._goal_distance()


def run_scenario(model: torch.nn.Module, seed: int, variant: str, max_steps: int, fov_deg: float) -> EpisodeTrace:
    rng = np.random.default_rng(seed)
    raw_env, env = make_env(model, max_steps, fov_deg)
    env.reset(seed=seed)
    install_blocked_corridor(raw_env, rng, variant)
    obs = env._flatten_obs(raw_env._observation())
    info = raw_env._info(collision=False, goal_reached=False)

    positions = [(float(raw_env.robot.x), float(raw_env.robot.y))]
    headings = [float(raw_env.robot.heading)]
    actions: list[int] = []
    rewards: list[float] = []
    total_reward = 0.0
    done = False
    min_clearance_m, closest_obstacle_kind = closest_collidable_margin(raw_env)

    while not done:
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = int(torch.argmax(logits, dim=-1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        actions.append(action)
        rewards.append(float(reward))
        positions.append((float(raw_env.robot.x), float(raw_env.robot.y)))
        headings.append(float(raw_env.robot.heading))
        clearance_m, obstacle_kind = closest_collidable_margin(raw_env)
        if clearance_m < min_clearance_m:
            min_clearance_m = clearance_m
            closest_obstacle_kind = obstacle_kind
        done = terminated or truncated

    position_array = np.asarray(positions, dtype=np.float32)
    path_length_m = float(np.linalg.norm(np.diff(position_array, axis=0), axis=1).sum()) if len(position_array) > 1 else 0.0
    steering_values = [raw_env._decode_action(action)[0] for action in actions]

    return EpisodeTrace(
        seed=seed,
        total_reward=float(total_reward),
        steps=int(info["steps"]),
        goal_reached=bool(info["goal_reached"]),
        collision=bool(info["collision"]),
        final_goal_distance=float(info["goal_distance"]),
        min_clearance_m=float(min_clearance_m),
        closest_obstacle_kind=closest_obstacle_kind,
        path_length_m=path_length_m,
        mean_abs_steering=float(np.mean(np.abs(steering_values))) if steering_values else 0.0,
        action_switches=int(sum(a != b for a, b in zip(actions, actions[1:]))),
        object_counts=dict(info.get("object_counts", {})),
        object_contact={key: float(value) for key, value in info.get("object_contact", {}).items()},
        positions=positions,
        headings=headings,
        actions=actions,
        rewards=rewards,
        goal_xy=(float(raw_env.goal_xy[0]), float(raw_env.goal_xy[1])),
        obstacles=[asdict(obstacle) for obstacle in raw_env.obstacles],
    )


def write_scenario_metadata(path: Path, args: argparse.Namespace, traces: list[EpisodeTrace]) -> None:
    payload = {
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "fov_deg": args.fov_deg,
        "variants": args.variant,
        "hypothesis": "Wide obstacle blocks the goal corridor at distance; policy must detour early rather than recover late.",
        "success_rate": float(np.mean([trace.goal_reached for trace in traces])),
        "collision_rate": float(np.mean([trace.collision for trace in traces])),
        "timeout_rate": float(np.mean([not trace.goal_reached and not trace.collision for trace in traces])),
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FieldNav policy on blocked-corridor scenes")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--fov-deg", type=float, default=100.0)
    parser.add_argument("--variant", choices=["wall", "trees", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-max-frames", type=int, default=120)
    args = parser.parse_args()

    model = load_policy(args.checkpoint)
    variants = ["wall", "trees"] if args.variant == "both" else [args.variant]
    traces = [
        run_scenario(model, args.seed + idx, variants[idx % len(variants)], args.max_steps, args.fov_deg)
        for idx in range(args.episodes)
    ]
    traces = sorted(traces, key=lambda trace: trace.total_reward, reverse=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_grid(traces[: args.top_k], args.out_dir / "top_blocked_corridor_rollouts.png", True, args.fov_deg)
    failures = select_failure_traces(traces, args.top_k)
    if failures:
        save_grid(failures, args.out_dir / "failure_blocked_corridor_rollouts.png", True, args.fov_deg)
        save_gif(failures[0], args.out_dir / "failure_blocked_corridor.gif", args.gif_fps, args.gif_max_frames, True, args.fov_deg)
    save_gif(traces[0], args.out_dir / "best_blocked_corridor.gif", args.gif_fps, args.gif_max_frames, True, args.fov_deg)
    save_summary(traces, args.out_dir / "render_summary.json")
    write_scenario_metadata(args.out_dir / "blocked_corridor_eval.json", args, traces)

    summary = json.loads((args.out_dir / "render_summary.json").read_text())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
