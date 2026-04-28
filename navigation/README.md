# Field Navigation Environment (Prototype)

This directory contains a first implementation milestone for local navigation RL:

- `FieldNavEnv`: a 2D local-navigation environment with a robot-centered polar obstacle map,
  a compact goal vector, and robot state features.
- Mixed scene objects: tree rows, bushes, potholes, moving people, and wall segments.
- Discrete steering actions with optional brake/coast/accelerate speed control.
- Reward composed of progress, goal completion, hard-object collision penalty,
  near-obstacle penalty, semantic contact/proximity penalties, steering smoothness,
  and a small time penalty.
- Scripts for random rollouts, PPO training, and policy rendering.

## Quickstart

```bash
python navigation/scripts/random_rollout.py --episodes 5
PATH="$PWD/.venv/bin:$PATH" CC=/opt/homebrew/Cellar/llvm/19.1.7_1.reinstall/bin/clang CXX=/opt/homebrew/Cellar/llvm/19.1.7_1.reinstall/bin/clang++ ./build.sh field_nav --cpu
puffer train field_nav --device auto --checkpoint-dir navigation/artifacts/puffer_checkpoints --log-dir navigation/artifacts/puffer_logs
python navigation/scripts/render_policy.py --checkpoint navigation/artifacts/puffer_checkpoints/field_nav/<run_id>/<step>.bin --out-dir navigation/artifacts/renders_cnn
```

Training outputs:

- Puffer checkpoints under `checkpoint_dir/field_nav/<run_id>/*.bin`
- Puffer metrics under `log_dir/field_nav/<run_id>.json`

Remote monitoring over Tailscale:

```bash
python navigation/scripts/training_dashboard.py \
  --tailscale \
  --checkpoint-dir navigation/artifacts/native_checkpoints \
  --log-dir navigation/artifacts/native_logs
```

Open the printed `tailscale_url` from a phone connected to the same Tailnet.

`field_nav` is registered as a native Ocean/Puffer environment in
`config/field_nav.ini` and implemented in `ocean/field_nav`. The current Puffer
policy uses `FieldNavPolarEncoder`: a `3 x 32 x 16` polar map is encoded with a
compact convolutional stack, then concatenated with the compact goal/state
features before the actor and value heads.

On Apple Silicon, `--device auto` uses MPS when CUDA is not available. You can
force it with `--device mps` or compare against CPU with `--device cpu`.
The default config uses 64 agents, `train.replay_ratio = 2.0`, LR warmup, and a
nonzero LR floor for continuation training.

## Notes on dependencies

The environment uses `gymnasium` if installed. In constrained environments where
`gymnasium` is unavailable, it falls back to a tiny local space implementation
so rollout/training scripts can still run.

## Observation Channels

The default observation uses polar bins centered on the robot and rotated into
the robot heading frame. The three behavior channels are:

1. static collidable obstacles, such as trees and walls
2. moving obstacles, such as people
3. soft hazards, such as bushes and potholes

Reset sampling rejects layouts with collidable objects too close to the robot
start, goal, or immediate forward path. This keeps training focused on reachable
navigation episodes instead of unavoidable first-step collisions.

## Next Steps

1. Add curriculum wrappers for object density, people speed, and sensor noise.
2. Add a simple safety filter module to clamp unsafe policy output.
3. Add evaluation + visualization script for trained checkpoints.
