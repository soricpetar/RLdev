#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="navigation/artifacts/polar_30m2_checkpoints/field_nav/1776868172360/0000000018198528.bin"
CHECKPOINT_DIR="navigation/artifacts/polar_entropy_sweep_15m_checkpoints"
LOG_DIR="navigation/artifacts/polar_entropy_sweep_15m_logs"

SEEDS=(1 2 3 4 5)
ENTROPIES=(0.010 0.015 0.020 0.025 0.030)

for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  ent_coef="${ENTROPIES[$idx]}"
  echo "START seed=${seed} ent_coef=${ent_coef}"
  python -m pufferlib.pufferl train field_nav \
    --device auto \
    --eval-episodes 0 \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-dir "${LOG_DIR}" \
    --load-model-path "${CHECKPOINT}" \
    --seed "${seed}" \
    --train.seed "${seed}" \
    --train.total-timesteps 15000000 \
    --vec.total-agents 64 \
    --train.minibatch-size 6144 \
    --train.replay-ratio 2.0 \
    --train.min-lr-ratio 0.3 \
    --train.lr-warmup-steps 500000 \
    --train.ent-coef "${ent_coef}" \
    --policy.hidden-size 256 \
    --policy.num-layers 2 \
    --env.polar-observation 1 \
    --env.polar-angle-bins 32 \
    --env.polar-distance-bins 16 \
    --env.polar-max-distance-m 24.0 \
    --env.curriculum-enabled 0 \
    --env.curriculum-warmup-steps 0
  echo "DONE seed=${seed} ent_coef=${ent_coef}"
done
