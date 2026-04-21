from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navigation.nav_env import FieldNavEnv
from navigation.scripts.train_policy import make_model
import pufferlib.models
import pufferlib.emulation


@dataclass
class EpisodeTrace:
    seed: int
    total_reward: float
    steps: int
    goal_reached: bool
    collision: bool
    final_goal_distance: float
    positions: list[tuple[float, float]]
    headings: list[float]
    actions: list[int]
    rewards: list[float]
    goal_xy: tuple[float, float]
    obstacles: list[dict[str, float]]


class PufferPolicyAdapter(torch.nn.Module):
    def __init__(self, policy: pufferlib.models.Policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor):
        state = self.policy.initial_state(obs.shape[0], device=obs.device)
        logits, values, _ = self.policy.forward_eval(obs, state)
        return logits, values.squeeze(-1)


def load_policy(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        encoder = pufferlib.models.FieldNavEncoder(obs_size=20487, hidden_size=256, map_channels=5, map_size=64)
        network = pufferlib.models.MLP(hidden_size=256, num_layers=2)
        decoder = pufferlib.models.DefaultDecoder([5], hidden_size=256)
        policy = pufferlib.models.Policy(encoder, decoder, network)
        state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        policy.load_state_dict(state_dict)
        policy.eval()
        return PufferPolicyAdapter(policy)

    architecture = checkpoint.get("architecture", "mlp")
    model = make_model(
        architecture,
        checkpoint["obs_dim"],
        checkpoint["act_dim"],
        map_channels=checkpoint.get("map_channels", 5),
        map_size=checkpoint.get("map_size", 64),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_episode(model: torch.nn.Module, seed: int, max_steps: int) -> EpisodeTrace:
    raw_env = FieldNavEnv(max_steps=max_steps)
    env = pufferlib.emulation.GymnasiumPufferEnv(raw_env)
    obs, info = env.reset(seed=seed)

    positions = [(float(raw_env.robot.x), float(raw_env.robot.y))]
    headings = [float(raw_env.robot.heading)]
    actions: list[int] = []
    rewards: list[float] = []
    total_reward = 0.0
    done = False
    last_info = info

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
        last_info = info
        done = terminated or truncated

    return EpisodeTrace(
        seed=seed,
        total_reward=float(total_reward),
        steps=int(last_info["steps"]),
        goal_reached=bool(last_info["goal_reached"]),
        collision=bool(last_info["collision"]),
        final_goal_distance=float(last_info["goal_distance"]),
        positions=positions,
        headings=headings,
        actions=actions,
        rewards=rewards,
        goal_xy=(float(raw_env.goal_xy[0]), float(raw_env.goal_xy[1])),
        obstacles=[asdict(obstacle) for obstacle in raw_env.obstacles],
    )


def status_label(trace: EpisodeTrace) -> str:
    if trace.goal_reached:
        return "goal"
    if trace.collision:
        return "collision"
    return "timeout/out"


def draw_trace(ax, trace: EpisodeTrace, title: str, trail_until: int | None = None) -> None:
    positions = np.asarray(trace.positions, dtype=np.float32)
    if trail_until is not None:
        trail_until = max(1, min(trail_until, len(positions)))
        positions = positions[:trail_until]

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_facecolor("#f8f7f3")
    ax.grid(True, color="#dedbd2", linewidth=0.7)

    circle_styles = {
        "tree": ("#295135", "#8bbf72", 0.86),
        "bush": ("#74a857", "#b7d892", 0.58),
        "pothole": ("#5c4632", "#d1b487", 0.76),
        "person": ("#c1121f", "#ffccd5", 0.90),
    }
    for obstacle in trace.obstacles:
        kind = obstacle.get("kind", "tree")
        if kind == "wall":
            ax.plot(
                [obstacle["x1"], obstacle["x2"]],
                [obstacle["y1"], obstacle["y2"]],
                color="#2f2f2f",
                linewidth=max(2.0, 10.0 * obstacle["thickness"]),
                solid_capstyle="round",
                alpha=0.92,
                zorder=4,
            )
            continue

        color, halo, alpha = circle_styles.get(kind, ("#4d4d4d", "#e36f4a", 0.78))
        body = plt.Circle(
            (obstacle["x"], obstacle["y"]),
            obstacle["radius"],
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=4,
        )
        inflated = plt.Circle(
            (obstacle["x"], obstacle["y"]),
            obstacle["radius"] + 0.8,
            color=halo,
            alpha=0.14,
            linewidth=0,
            zorder=3,
        )
        ax.add_patch(inflated)
        ax.add_patch(body)

    ax.scatter(*trace.goal_xy, marker="*", s=180, color="#198754", edgecolor="white", linewidth=0.8, zorder=6)
    ax.scatter(positions[0, 0], positions[0, 1], marker="o", s=48, color="#1f77b4", edgecolor="white", zorder=7)
    ax.plot(positions[:, 0], positions[:, 1], color="#0b5ed7", linewidth=2.0, zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], marker="o", s=54, color="#d63384", edgecolor="white", zorder=8)

    heading_idx = len(positions) - 1
    heading = trace.headings[min(heading_idx, len(trace.headings) - 1)]
    ax.arrow(
        positions[-1, 0],
        positions[-1, 1],
        0.8 * np.cos(heading),
        0.8 * np.sin(heading),
        color="#d63384",
        width=0.05,
        head_width=0.35,
        length_includes_head=True,
        zorder=9,
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def save_grid(traces: list[EpisodeTrace], output_path: Path) -> None:
    cols = min(3, len(traces))
    rows = int(np.ceil(len(traces) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, trace in enumerate(traces):
        ax = axes.ravel()[idx]
        ax.axis("on")
        title = (
            f"#{idx + 1} seed={trace.seed} {status_label(trace)}\n"
            f"return={trace.total_reward:.2f} steps={trace.steps} dist={trace.final_goal_distance:.2f}m"
        )
        draw_trace(ax, trace, title)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def fig_to_image(fig) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_gif(trace: EpisodeTrace, output_path: Path, fps: int, max_frames: int) -> None:
    num_points = len(trace.positions)
    frame_count = min(max_frames, max(2, num_points))
    frame_indices = np.unique(np.linspace(1, num_points, frame_count, dtype=int))
    frames = []

    for idx in frame_indices:
        fig, ax = plt.subplots(figsize=(6, 6))
        title = (
            f"seed={trace.seed} {status_label(trace)} "
            f"return={trace.total_reward:.2f} step={idx - 1}/{trace.steps}"
        )
        draw_trace(ax, trace, title, trail_until=int(idx))
        fig.tight_layout()
        frames.append(fig_to_image(fig))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=1.0 / fps)


def save_summary(traces: list[EpisodeTrace], output_path: Path) -> None:
    payload = {
        "episodes": len(traces),
        "success_rate": float(np.mean([t.goal_reached for t in traces])),
        "collision_rate": float(np.mean([t.collision for t in traces])),
        "mean_return": float(np.mean([t.total_reward for t in traces])),
        "best": {
            "seed": traces[0].seed,
            "return": traces[0].total_reward,
            "steps": traces[0].steps,
            "goal_reached": traces[0].goal_reached,
            "collision": traces[0].collision,
            "final_goal_distance": traces[0].final_goal_distance,
        },
        "top_episodes": [
            {
                "rank": idx + 1,
                "seed": trace.seed,
                "return": trace.total_reward,
                "steps": trace.steps,
                "goal_reached": trace.goal_reached,
                "collision": trace.collision,
                "final_goal_distance": trace.final_goal_distance,
            }
            for idx, trace in enumerate(traces[:10])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render trained FieldNavEnv policy rollouts")
    parser.add_argument("--checkpoint", type=Path, default=Path("navigation/artifacts/fieldnav_policy.pt"))
    parser.add_argument("--load-model-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("navigation/artifacts/renders"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-max-frames", type=int, default=120)
    args = parser.parse_args()
    if args.load_model_path is not None:
        args.checkpoint = args.load_model_path

    model = load_policy(args.checkpoint)
    traces = [run_episode(model, args.seed + idx, args.max_steps) for idx in range(args.episodes)]
    traces.sort(key=lambda t: (t.goal_reached, t.total_reward), reverse=True)
    top_traces = traces[: args.top_k]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.out_dir / "top_policy_rollouts.png"
    gif_path = args.out_dir / "best_policy_rollout.gif"
    summary_path = args.out_dir / "render_summary.json"

    save_grid(top_traces, grid_path)
    save_gif(top_traces[0], gif_path, fps=args.gif_fps, max_frames=args.gif_max_frames)
    save_summary(traces, summary_path)

    print(f"saved_grid={grid_path}")
    print(f"saved_gif={gif_path}")
    print(f"saved_summary={summary_path}")
    print(
        json.dumps(
            {
                "episodes": len(traces),
                "success_rate": float(np.mean([t.goal_reached for t in traces])),
                "collision_rate": float(np.mean([t.collision for t in traces])),
                "mean_return": float(np.mean([t.total_reward for t in traces])),
                "best_seed": top_traces[0].seed,
                "best_return": top_traces[0].total_reward,
                "best_steps": top_traces[0].steps,
                "best_final_goal_distance": top_traces[0].final_goal_distance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
