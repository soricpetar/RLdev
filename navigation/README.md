# Field Navigation Environment (Prototype)

This directory contains a first implementation milestone for local navigation RL:

- `FieldNavEnv`: a 2D local-navigation environment with a robot-centered 5x64x64 semantic risk map,
  a compact goal vector, and robot state features.
- Mixed scene objects: tree rows, bushes, potholes, moving people, and wall segments.
- Discrete steering actions with fixed forward speed.
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
`config/field_nav.ini` and implemented in `ocean/field_nav`. The Puffer policy
uses `FieldNavCompactEncoder`: the 5-channel semantic map is encoded with a
compact convolutional stack, then concatenated with the compact goal/state
features before the actor and value heads.

On Apple Silicon, `--device auto` uses MPS when CUDA is not available. You can
force it with `--device mps` or compare against CPU with `--device cpu`.
The default config favors higher environment SPS with 32 agents and
`train.replay_ratio = 1.0`; increase `--train.replay-ratio` if you want more PPO
optimization passes per collected batch.

## Notes on dependencies

The environment uses `gymnasium` if installed. In constrained environments where
`gymnasium` is unavailable, it falls back to a tiny local space implementation
so rollout/training scripts can still run.

## Costmap Channels

The local map is centered on the robot and rotated into the robot heading frame.
Each cell stores object risk in one semantic channel:

1. hard static objects, such as trees
2. soft vegetation, such as bushes
3. terrain hazards, such as potholes
4. dynamic people
5. wall segments

## Next Steps

1. Add curriculum wrappers for object density, people speed, and sensor noise.
2. Add a simple safety filter module to clamp unsafe policy output.
3. Add evaluation + visualization script for trained checkpoints.
