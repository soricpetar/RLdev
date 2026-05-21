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
./.venv/bin/python navigation/scripts/training_dashboard.py \
  --tailscale \
  --port 8769 \
  --env field_nav_camera_native_hard \
  --checkpoint-dir <current_checkpoint_dir> \
  --log-dir <current_log_dir>
```

Open the printed `tailscale_url` from another machine or phone connected to the
same Tailnet. On this machine the usual remote dashboard URL is
`http://100.120.184.115:8769`; the policy viewer is
`http://100.120.184.115:8769/policy`.

`--tailscale` makes the dashboard listen on `0.0.0.0` when the host was left at
the default `127.0.0.1`, then prints both local and Tailscale URLs. If the port
is already occupied, find and stop the old dashboard before restarting it:

```bash
lsof -nP -iTCP:8769 -sTCP:LISTEN
kill <pid>
```

For RLdev navigation scripts, prefer `./.venv/bin/python`. The system Python on
this machine may be too old for current repo imports.

Current optimized native camera hard settings:

- env: `field_nav_camera_native_hard`
- narrow forward FOV: `100.0` degrees
- agents/threads: `256` agents, `4` env threads
- horizon/minibatch: `192` horizon, `49152` minibatch
- replay schedule: `1.0 -> 2.0` over `15M` steps is stable; the best current
  checkpoint came from a `1.0 -> 3.0` schedule over `20M` with mid-run
  checkpoint selection
- successful continuation LR: `2e-4`; the `1e-4` fine-tune regressed
- expected SPS: `72K-75K` once replay ratio reaches `2.0`; early ramp can
  exceed `100K`

Current best 100-degree FOV hard native policy:

```text
navigation/artifacts/front_camera_native_stage13_hard_100fov_rr3_30m_checkpoints/field_nav_camera_native_hard/1778504318206/0000000024625152.bin
```

Fixed-seed 300-episode eval of this checkpoint: `92.00%` success, `8.00%`
collision, mean return `32.84`. It beat Stage11 (`89.67%`) and the later
Stage14-16 branches on the same seeds.

Blocked-corridor curriculum knobs are available in `field_nav_camera_native_hard`
and default to off:

- `blocked_corridor_prob`
- `blocked_corridor_wall_prob`
- `blocked_corridor_min_distance_m`
- `blocked_corridor_max_distance_m`
- `blocked_corridor_min_half_width_m`
- `blocked_corridor_max_half_width_m`

The separate speed-control follow-up env is
`field_nav_camera_native_hard_speed` with `15` actions. Keep it separate from
the fixed-speed env because old `5`-action checkpoints are not decoder-compatible
without an explicit transfer step.

Current policy size:

- total trainable params: `736,086`
- dominant component: `encoder.map_encoder`, `524,544` params, about `71%`
- MLP trunk: about `18%`
- encoder projection: about `11%`

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
