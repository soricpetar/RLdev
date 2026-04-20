# Field Navigation Environment (Prototype)

This directory contains a first implementation milestone for local navigation RL:

- `FieldNavEnv`: a 2D local-navigation environment with a robot-centered 64x64 costmap,
  a compact goal vector, and robot state features.
- Discrete steering actions with fixed forward speed.
- Reward composed of progress, goal completion, collision penalty, near-obstacle
  penalty, steering smoothness, and a small time penalty.
- Scripts for random rollouts and PPO training.

## Quickstart

```bash
python navigation/scripts/random_rollout.py --episodes 5
python navigation/scripts/train_policy.py --updates 20 --num-envs 8 --rollout-steps 64
```

Training outputs:

- `navigation/artifacts/fieldnav_policy.pt`
- `navigation/artifacts/fieldnav_policy.json`

## Notes on dependencies

The environment uses `gymnasium` if installed. In constrained environments where
`gymnasium` is unavailable, it falls back to a tiny local space implementation
so rollout/training scripts can still run.

## Next Steps

1. Add curriculum wrappers for obstacle density and sensor noise.
2. Add a simple safety filter module to clamp unsafe policy output.
3. Add evaluation + visualization script for trained checkpoints.
