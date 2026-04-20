from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navigation.nav_env import FieldNavEnv
import pufferlib.emulation


@dataclass
class TrainConfig:
    seed: int = 0
    num_envs: int = 16
    rollout_steps: int = 128
    updates: int = 80
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    update_epochs: int = 4
    minibatches: int = 4
    max_grad_norm: float = 0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        return self.pi(h), self.v(h).squeeze(-1)


def make_env(seed: int):
    return pufferlib.emulation.GymnasiumPufferEnv(FieldNavEnv(max_steps=300)).env


def main():
    parser = argparse.ArgumentParser(description="Train PPO policy for FieldNavEnv")
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="navigation/artifacts/fieldnav_policy.pt")
    args = parser.parse_args()

    cfg = TrainConfig(seed=args.seed, num_envs=args.num_envs, rollout_steps=args.rollout_steps, updates=args.updates)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    envs = [make_env(cfg.seed + i) for i in range(cfg.num_envs)]
    wrapped = [pufferlib.emulation.GymnasiumPufferEnv(e) for e in envs]

    obs_list = []
    for i, env in enumerate(wrapped):
        obs, _ = env.reset(seed=cfg.seed + i)
        obs_list.append(obs)
    obs = np.stack(obs_list)

    obs_dim = obs.shape[1]
    act_dim = wrapped[0].action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    episodic_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    finished_returns = []

    for update in range(cfg.updates):
        obs_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.int64)
        logp_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
        val_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)

        for t in range(cfg.rollout_steps):
            obs_buf[t] = obs
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            act_np = action.cpu().numpy()
            logp_np = logp.cpu().numpy()
            val_np = value.cpu().numpy()

            next_obs = []
            for i, env in enumerate(wrapped):
                o, r, term, trunc, _ = env.step(int(act_np[i]))
                done = term or trunc
                episodic_returns[i] += r
                if done:
                    finished_returns.append(float(episodic_returns[i]))
                    episodic_returns[i] = 0.0
                    o, _ = env.reset()
                next_obs.append(o)
                rew_buf[t, i] = r
                done_buf[t, i] = float(done)

            act_buf[t] = act_np
            logp_buf[t] = logp_np
            val_buf[t] = val_np
            obs = np.stack(next_obs)

        with torch.no_grad():
            next_v = model(torch.as_tensor(obs, dtype=torch.float32, device=device))[1].cpu().numpy()

        adv_buf = np.zeros_like(rew_buf)
        lastgaelam = np.zeros(cfg.num_envs, dtype=np.float32)
        for t in reversed(range(cfg.rollout_steps)):
            if t == cfg.rollout_steps - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_values = next_v
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_values = val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * next_values * next_nonterminal - val_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgaelam
            adv_buf[t] = lastgaelam
        ret_buf = adv_buf + val_buf

        b_obs = torch.as_tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=device)
        b_act = torch.as_tensor(act_buf.reshape(-1), dtype=torch.int64, device=device)
        b_logp = torch.as_tensor(logp_buf.reshape(-1), dtype=torch.float32, device=device)
        b_adv = torch.as_tensor(adv_buf.reshape(-1), dtype=torch.float32, device=device)
        b_ret = torch.as_tensor(ret_buf.reshape(-1), dtype=torch.float32, device=device)
        b_val = torch.as_tensor(val_buf.reshape(-1), dtype=torch.float32, device=device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        batch_size = b_obs.shape[0]
        mb_size = batch_size // cfg.minibatches

        idxs = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, mb_size):
                mb_idx = idxs[start : start + mb_size]
                logits, new_val = model(b_obs[mb_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb_idx])
                entropy = dist.entropy().mean()

                logratio = new_logp - b_logp[mb_idx]
                ratio = logratio.exp()

                mb_adv = b_adv[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_val - b_ret[mb_idx]) ** 2
                v_clipped = b_val[mb_idx] + torch.clamp(new_val - b_val[mb_idx], -cfg.clip_coef, cfg.clip_coef)
                v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_ret[mb_idx]) ** 2).mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        if (update + 1) % 10 == 0:
            avg_ret = np.mean(finished_returns[-50:]) if finished_returns else 0.0
            print(f"update={update + 1} avg_return_50={avg_ret:.3f} finished_episodes={len(finished_returns)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "avg_return_50": float(np.mean(finished_returns[-50:])) if finished_returns else 0.0,
        },
        out_path,
    )

    metrics_path = out_path.with_suffix(".json")
    metrics = {
        "updates": cfg.updates,
        "num_envs": cfg.num_envs,
        "rollout_steps": cfg.rollout_steps,
        "episodes_finished": len(finished_returns),
        "avg_return_50": float(np.mean(finished_returns[-50:])) if finished_returns else 0.0,
        "best_return": float(np.max(finished_returns)) if finished_returns else 0.0,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"saved_policy={out_path}")
    print(f"saved_metrics={metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
