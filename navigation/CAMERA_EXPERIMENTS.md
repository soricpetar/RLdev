# Forward-Camera Costmap Experiments

This tracks experiments for policies trained with a simulated forward-only,
camera-derived polar costmap.

## Experiment FC-001 - Stage 1 Smoke

Purpose:

- Test whether the policy can train from a partial polar observation before any
  real camera perception noise is added.
- Keep the polar layout close to the existing policy while exposing rear/side
  bins as unknown.
- Use speed control so the agent can slow down and turn to inspect unknown
  space.

Observation:

- `costmap`: `4 x 32 x 16`
- channels:
  - static obstacle
  - moving obstacle
  - soft hazard
  - unknown
- visible sector: `100 deg` forward FOV
- unseen rear/side bins: obstacle channels `0`, unknown channel `1`
- vector features: `goal(3) + state(4)`

Training setup:

- env: `field_nav_camera`
- config: `config/field_nav_camera.ini`
- backend: Python
- total timesteps: `1,000,000`
- agents: `64`
- horizon: `192`
- action space: `15` actions, speed control enabled
- max speed: `1.2 m/s`
- reset clearance:
  - start: `1.0m`
  - goal: `1.0m`
  - forward: `1.5m`
  - forward margin: `0.25m`
- curriculum warmup: `500,000` steps
- easier Stage 1 object ranges:
  - tree rows: `(1, 2)`
  - bushes: `(3, 8)`
  - potholes: `(1, 4)`
  - people: `(0, 2)`
  - walls: `(0, 1)`

Dashboard:

- local: `http://127.0.0.1:8766`
- Tailscale: `http://100.120.184.115:8766`
- checkpoint dir: `navigation/artifacts/front_camera_stage1_checkpoints`
- log dir: `navigation/artifacts/front_camera_stage1_logs`

Initial live status:

- started: 2026-04-29
- early reading at ~12K steps:
  - score: `-21.9`
  - success: `25%`
  - collision: `75%`
  - curriculum progress: `0.0`

Interpretation:

- Early metrics are not meaningful yet; the policy is random and the partial
  observation task is harder than full 360-degree polar navigation.
- Main smoke criteria are whether the run completes, produces checkpoints/logs,
  and shows improvement by the end of 1M steps.

Final result:

- run id: `1777489970220`
- completed steps: `995,328`
- checkpoint count: `9`
- final checkpoint:
  `navigation/artifacts/front_camera_stage1_checkpoints/field_nav_camera/1777489970220/0000000000995328.bin`
- final score: `-31.32`
- final success: `3.9%`
- final collision: `28.9%`
- final episode length: `397.4`
- final curriculum progress: `0.031`
- best live success was the first early random-policy window at `25%`; after
  enough episodes accumulated, success settled around `3-8%`.
- lowest live collision observed: `22.3%`, but this was not accompanied by
  goal-reaching. The dominant failure mode was timeout/wandering.

Conclusion:

- This setup is a failed learning smoke, not a viable policy.
- The model did not meaningfully update: entropy stayed near maximum
  (`2.707` for 15 actions), KL stayed near zero, and explained variance stayed
  around zero.
- The 100-degree FOV plus long 10-22m goals is too hard as a first partial
  observability curriculum. The agent usually cannot discover a useful
  inspect-then-commit behavior from sparse goal bonus/progress alone.

Next experiment:

- Make the first camera stage much easier:
  - FOV `180 deg`
  - no people or walls
  - goal distance `4-10m`
  - max steps `250`
  - no curriculum warmup delay, or a curriculum that starts at this simple task
    and expands FOV/object difficulty only after success improves
- Add partial-observation shaping:
  - small heading-alignment reward toward the goal
  - blind-forward penalty when moving into unknown bins
  - optional inspection bonus for reducing unknown bins near the goal direction
- Keep the explicit unknown channel. Do not fill unseen bins as free.

## Experiment FC-002 - Easy Front-FOV Stage

Purpose:

- Test a substantially easier partial-observation task after FC-001 failed.
- Determine whether wider FOV, shorter goals, simpler maps, shorter episodes,
  and shaping are enough to get non-random goal seeking.

Observation:

- `costmap`: `4 x 32 x 16`
- visible sector: `180 deg` forward FOV
- rear bins marked in the unknown channel
- max polar range: `16m`

Environment:

- env: `field_nav_camera_easy`
- config: `config/field_nav_camera_easy.ini`
- total timesteps: `1,000,000`
- max steps: `250`
- goal distance: `4-10m`
- speed control: enabled, max speed `1.2m/s`
- people: none
- walls: none
- tree rows: `(0, 1)`
- bushes: `(0, 3)`
- potholes: `(0, 2)`
- reward shaping:
  - heading alignment scale: `0.04`
  - moving while goal is outside FOV penalty: `0.03`

Dashboard:

- local: `http://127.0.0.1:8767`
- Tailscale: `http://100.120.184.115:8767`
- checkpoint dir: `navigation/artifacts/front_camera_stage2_easy_checkpoints`
- log dir: `navigation/artifacts/front_camera_stage2_easy_logs`

Final result:

- run id: `1777490900867`
- completed steps: `995,328`
- checkpoint count: `9`
- final checkpoint:
  `navigation/artifacts/front_camera_stage2_easy_checkpoints/field_nav_camera_easy/1777490900867/0000000000995328.bin`
- final score: `-17.05`
- best logged score: `-15.10`
- final success: `2.7%`
- best logged success: `6.25%`
- final collision: `3.5%`
- best logged collision: `1.6%`
- final episode length: `243.6 / 250`
- entropy stayed near random-policy maximum for 15 actions: `2.707`
- explained variance improved slightly but remained very low: final `0.0028`

Interpretation:

- FC-002 is safer than FC-001 but still not a useful navigation policy.
- Lower collisions mostly came from making the task easier and the policy timing
  out, not from successful navigation.
- The policy is still close to random. The update signal is too weak: entropy
  remains near maximum, KL is effectively zero, and value fit is poor.
- The heading-alignment reward improved score versus FC-001 but did not solve
  sparse goal reaching.

Next recommendation:

- Use an even more direct bootstrapping objective before obstacle navigation:
  - empty maps only
  - `180 deg` FOV
  - goals `3-6m`
  - fixed speed or reduced speed-control action space
  - stronger heading/progress shaping
- Once empty-map success is high, add obstacles back gradually.
- Alternatively, initialize from the existing full-polar policy where the first
  three channels have compatible semantics, then fine-tune with the unknown
  channel and front-only masking.

## Experiment FC-003 - Empty-Map Bootstrap 50M

Purpose:

- Test whether the forward-camera polar observation can learn basic goal seeking
  when obstacle avoidance is removed from the problem.
- Use a smaller fixed-speed steering action space before reintroducing speed
  control and obstacles.
- Run long enough to separate early random behavior from real learning.

Observation:

- `costmap`: `4 x 32 x 16`
- visible sector: `180 deg` forward FOV
- unseen rear bins: unknown channel `1`
- polar max range: `12m`

Environment:

- env: `field_nav_camera_bootstrap`
- config: `config/field_nav_camera_bootstrap.ini`
- total timesteps: `50,000,000`
- max steps: `150`
- goal distance: `3-6m`
- speed control: disabled
- action space: `5` fixed-speed steering actions
- fixed speed: `1.0 m/s`
- objects: none
- reward shaping:
  - heading alignment scale: `0.10`
  - moving while goal is outside FOV penalty: `0.02`

Dashboard:

- local: `http://127.0.0.1:8768`
- Tailscale: `http://100.120.184.115:8768`
- policy viewer: `http://127.0.0.1:8768/policy`
- checkpoint dir: `navigation/artifacts/front_camera_stage3_bootstrap_checkpoints`
- log dir: `navigation/artifacts/front_camera_stage3_bootstrap_logs`

Final result:

- run id: `1777491392094`
- completed steps: `49,999,872`
- wall time: `33m 55s`
- checkpoint count: `42`
- final checkpoint:
  `navigation/artifacts/front_camera_stage3_bootstrap_checkpoints/field_nav_camera_bootstrap/1777491392094/0000000049999872.bin`
- final score: `25.63`
- final success: `100.0%`
- final collision: `0.0%`
- final episode length: `27.3 / 150`
- final explained variance: `0.523`
- final entropy: `0.739` for `5` actions

Training curve:

- ~6.3M steps:
  - score: `0.16`
  - success: `46.4%`
  - episode length: `108.0`
- ~18.8M steps:
  - score: `24.31`
  - success: `95.7%`
  - episode length: `37.6`
- ~31.3M steps:
  - score: `23.76`
  - success: `93.3%`
  - episode length: `39.0`
- ~43.8M steps:
  - score: `25.29`
  - success: `98.7%`
  - episode length: `29.1`
- final:
  - score: `25.63`
  - success: `100.0%`
  - episode length: `27.3`

Interpretation:

- FC-003 is the first successful forward-camera costmap training run.
- The agent can learn goal seeking from a forward-only polar costmap with
  unknown rear bins when the task is reduced to empty maps and fixed-speed
  steering.
- The successful result does not prove obstacle avoidance yet. It establishes a
  bootstrap policy and curriculum starting point.
- The non-monotonic mid-run dip around 24-32M steps suggests checkpoint
  selection matters; final and late checkpoints are strongest for this task.

Next recommendation:

- Reintroduce obstacles gradually from this bootstrap:
  - keep `180 deg` FOV and fixed-speed steering
  - add sparse bushes/potholes first
  - keep people/walls disabled
  - preserve short `3-6m` goals
- Evaluate whether loading the FC-003 checkpoint and fine-tuning is more stable
  than training FC-004 from scratch.

## Native/Ocean Hard Camera Runs - 100-Degree FOV

These runs use the native/Ocean `field_nav_camera_native_hard` port, separate
from the older Python camera envs and the original `field_nav` env.

Common setup:

- env: `field_nav_camera_native_hard`
- config: `config/field_nav_camera_native_hard.ini`
- observation: `4 x 32 x 16` front-FOV polar costmap plus compact vector state
- action space: `5` fixed-speed steering actions
- hard world distribution: `10-22m` goals, `400` max steps, dense obstacles,
  moving people, and walls
- optimized throughput params: `256` agents, `4` env threads, horizon `192`,
  minibatch `49152`
- replay schedule: `1.0 -> 2.0` over `15M` steps
- successful continuation LR: `2e-4`
- dashboard command:

```bash
./.venv/bin/python navigation/scripts/training_dashboard.py \
  --tailscale \
  --port 8769 \
  --env field_nav_camera_native_hard \
  --checkpoint-dir <current_checkpoint_dir> \
  --log-dir <current_log_dir>
```

On this machine, the usual remote URL is `http://100.120.184.115:8769` and the
policy viewer is `http://100.120.184.115:8769/policy`. If port `8769` is still
serving an old run, use `lsof -nP -iTCP:8769 -sTCP:LISTEN`, stop the old
dashboard process, then restart the command above with the new dirs.

### Stage9 - Wide/Native Hard Continuation

- run id: `1778077794243`
- best checkpoint used as the narrow-FOV starting point:
  `navigation/artifacts/front_camera_native_stage9_hard_25m_continuation_checkpoints/field_nav_camera_native_hard/1778077794243/0000000019365888.bin`
- live best observed during the run: `97.037%` success, score `35.667` at
  `19,476,480` agent steps
- compact logged summary later showed final/best at `91.67%` success, score
  `33.10`
- interpretation: strongest wide/native hard source policy, but not directly
  comparable to later narrow-FOV results

### Stage10 - 100-Degree FOV 50M Continuation

- source checkpoint: Stage9 best-success checkpoint `0000000019365888.bin`
- run id: `1778155415835`
- FOV: `100.0` degrees
- total steps: `50M`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage10_hard_narrow_fov_rr_schedule_50m_checkpoints/field_nav_camera_native_hard/1778155415835/0000000049987584.bin`
- final success: `87.99%`
- final score: `30.67`
- final SPS: `72.6K`
- interpretation: recovered from the transfer hit caused by narrowing from the
  wide/native source policy, but did not beat the previous hard policy

### Stage11 - 100-Degree FOV 30M Continuation

- source checkpoint: Stage10 final checkpoint `0000000049987584.bin`
- run id: `1778156638753`
- total steps: `30M`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage11_hard_narrow_fov_rr_schedule_30m_checkpoints/field_nav_camera_native_hard/1778156638753/0000000029982720.bin`
- final success: `90.78%`
- final score: `32.10`
- final SPS: `74.2K`
- interpretation: improved the 100-degree FOV policy by `+2.79` success points
  and `+1.42` score over Stage10. This is the current best 100-degree FOV hard
  native checkpoint.

### Stage12 - 100-Degree FOV LR 1e-4 Fine-Tune

- source checkpoint: Stage11 final checkpoint `0000000029982720.bin`
- run id: `1778159660309`
- change tested: lower LR from `2e-4` to `1e-4`
- total steps: `30M`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage12_hard_narrow_fov_lr1e4_30m_checkpoints/field_nav_camera_native_hard/1778159660309/0000000029982720.bin`
- best success: `87.57%`
- best score: `30.47`
- final success: `86.38%`
- final score: `29.59`
- interpretation: regression. Do not use Stage12 as the current best policy;
  keep Stage11 as the best 100-degree FOV checkpoint.

### Stage13 - 100-Degree FOV Replay Ratio 3 Test

- source checkpoint: Stage11 final checkpoint `0000000029982720.bin`
- run id: `1778504318206`
- change tested: replay-ratio schedule `1.0 -> 3.0` over `20M`, LR `2e-4`
- total steps: `30M`
- best checkpoint by live success:
  `navigation/artifacts/front_camera_native_stage13_hard_100fov_rr3_30m_checkpoints/field_nav_camera_native_hard/1778504318206/0000000024625152.bin`
- best live success: `91.19%`
- best live score: `32.30`
- final success: `88.20%`
- final score: `30.88`
- interpretation: final checkpoint regressed, but the best mid-run checkpoint is
  stronger than Stage11 on fixed-seed eval.

### Stage14 - 120-Degree FOV Bridge

- Stage14a source checkpoint: Stage11 final checkpoint `0000000029982720.bin`
- Stage14a run id: `1778504770296`
- Stage14a change tested: widen to `120 deg` FOV for `15M` steps, replay
  schedule `1.0 -> 2.0`
- Stage14a best checkpoint:
  `navigation/artifacts/front_camera_native_stage14a_hard_120fov_bridge_15m_checkpoints/field_nav_camera_native_hard/1778504770296/0000000002506752.bin`
- Stage14a best live success: `94.50%`
- Stage14b source checkpoint: Stage14a final checkpoint `0000000014991360.bin`
- Stage14b run id: `1778504975554`
- Stage14b change tested: return to `100 deg` FOV for `30M` steps
- Stage14b best checkpoint:
  `navigation/artifacts/front_camera_native_stage14b_hard_120to100fov_30m_checkpoints/field_nav_camera_native_hard/1778504975554/0000000025853952.bin`
- Stage14b best live success: `91.39%`
- interpretation: the direct 120-FOV checkpoint is not comparable because eval
  uses 100-FOV rays; the 120-to-100 continuation improved over Stage11, but did
  not beat Stage13 on fixed-seed eval.

### Stage15 - Lower Entropy Continuation

- source checkpoint: Stage11 final checkpoint `0000000029982720.bin`
- run id: `1778505478549`
- change tested: entropy coefficient `0.002`, LR `2e-4`, replay schedule
  `1.0 -> 2.0` over `15M`
- best checkpoint:
  `navigation/artifacts/front_camera_native_stage15_hard_100fov_ent002_30m_checkpoints/field_nav_camera_native_hard/1778505478549/0000000013565952.bin`
- best live success: `92.03%`
- best live score: `33.05`
- final success: `87.85%`
- final score: `30.77`
- interpretation: useful mid-run improvement but not better than Stage13 on
  fixed-seed eval; final checkpoint degraded.

### Stage16 - Static-Clutter Failure Focus

- source checkpoint: Stage15 best-window checkpoint `0000000013565952.bin`
- run id: `1778505966448`
- change tested: denser hard static clutter with tree rows `3-5` and walls
  `2-5`, keeping `100 deg` FOV, entropy `0.002`, LR `2e-4`, replay schedule
  `1.0 -> 2.0`
- best checkpoint:
  `navigation/artifacts/front_camera_native_stage16_hard_100fov_static_clutter_30m_checkpoints/field_nav_camera_native_hard/1778505966448/0000000024625152.bin`
- best live success on the harder clutter training distribution: `87.13%`
- final success on the harder clutter training distribution: `80.79%`
- interpretation: made the train distribution harder, but did not improve the
  normal hard-env fixed-seed eval.

### Fixed-Seed 100-FOV Eval After Stage13-16

All rows below use `300` episodes, seed `900000`, normal hard camera eval,
`100 deg` FOV, reset clearances enabled, and render outputs under
`navigation/artifacts/policy_evals/compare_*_e300_s900000/`.

| policy | success | collision | mean return | collision kinds |
| --- | ---: | ---: | ---: | --- |
| Stage11 best | `89.67%` | `10.33%` | `31.61` | `21 tree, 5 person, 5 wall` |
| Stage13 best RR3 | `92.00%` | `8.00%` | `32.84` | `16 tree, 4 person, 4 wall` |
| Stage14a 120-FOV checkpoint evaluated at 100 FOV | `86.67%` | `13.67%` | `29.74` | `29 tree, 4 person, 8 wall` |
| Stage14b best 120-to-100 | `91.00%` | `9.00%` | `32.20` | `17 tree, 5 person, 5 wall` |
| Stage15 best lower entropy | `90.33%` | `9.67%` | `31.83` | `20 tree, 4 person, 5 wall` |
| Stage16 best static clutter | `89.33%` | `10.67%` | `31.59` | `20 tree, 6 person, 6 wall` |

Current best 100-degree FOV hard native policy:

```text
navigation/artifacts/front_camera_native_stage13_hard_100fov_rr3_30m_checkpoints/field_nav_camera_native_hard/1778504318206/0000000024625152.bin
```

Policy size note:

- total trainable params: `736,086`
- `encoder.map_encoder`: `524,544`, about `71%`
- MLP trunk: `131,584`, about `18%`
- encoder projection: `78,080`, about `11%`
- decoder/value/vector heads are small; policy size is dominated by the first
  map projection from the `4 x 32 x 16` polar observation.

## Blocked-Corridor Curriculum and Speed-Control Follow-Up

The blocked-corridor eval confirmed that the current best 100-FOV policy is
weak when a wide obstacle blocks the goal corridor at distance:

- Stage13 best on normal hard eval: `92.00%` success
- Stage13 best on mixed blocked corridors: `69.50%` success, `30.50%`
  collision
- wall-only blocked corridors: `62.00%` success, `38.00%` collision
- tree-row blocked corridors: `75.00%` success, `25.00%` collision
- artifacts:
  `navigation/artifacts/policy_evals/blocked_corridor_stage13_best_100fov_e200_s910000/`

Implementation added:

- `field_nav_camera_native_hard` now supports optional blocked-corridor
  sampling in the native/Ocean env.
- Defaults keep old behavior unchanged: `blocked_corridor_prob = 0.0`.
- New env knobs:
  - `blocked_corridor_prob`
  - `blocked_corridor_wall_prob`
  - `blocked_corridor_min_distance_m`
  - `blocked_corridor_max_distance_m`
  - `blocked_corridor_min_half_width_m`
  - `blocked_corridor_max_half_width_m`
- The block is placed across the robot-goal corridor after the normal random
  hard objects are sampled, with either a wall segment or a row of trees.

Recommended Stage17 fixed-speed curriculum:

```bash
./.venv/bin/python -m pufferlib.pufferl train field_nav_camera_native_hard \
  --load-model-path navigation/artifacts/front_camera_native_stage13_hard_100fov_rr3_30m_checkpoints/field_nav_camera_native_hard/1778504318206/0000000024625152.bin \
  --checkpoint-dir navigation/artifacts/front_camera_native_stage17_blocked_corridor_30m_checkpoints \
  --log-dir navigation/artifacts/front_camera_native_stage17_blocked_corridor_30m_logs \
  --env.blocked-corridor-prob 0.30 \
  --env.blocked-corridor-wall-prob 0.60 \
  --env.near-obstacle-threshold-m 4.0 \
  --train.total-timesteps 30000000 \
  --train.learning-rate 0.0002 \
  --train.replay-ratio 2.0 \
  --train.replay-ratio-start 1.0 \
  --train.replay-ratio-schedule-steps 15000000 \
  --train.experiment-name front_camera_native_stage17_blocked_corridor_30m \
  --train.experiment-description "30M-step fixed-speed continuation from Stage13 best with 30% blocked-corridor curriculum, 60% wall blocks, and near-obstacle threshold 4m. Tests early detour learning for wide distant obstacles while preserving 5-action checkpoint compatibility."
```

Success criteria:

- normal hard fixed-seed eval remains near Stage13: `>= 92%`
- mixed blocked-corridor eval improves from `69.5%` to `> 80%`
- wall-only blocked-corridor eval improves from `62%` to `> 75%`

Speed control was added as a separate native env so it does not silently mix
with the 5-action fixed-speed checkpoints:

- env: `field_nav_camera_native_hard_speed`
- config: `config/field_nav_camera_native_hard_speed.ini`
- native binding: `ocean/field_nav_camera_native_hard_speed/binding.c`
- action space: `15` actions = `5` steering commands x `3` throttle commands
- default speed envelope: `0.0-1.2m/s`

Use speed control only after Stage17 has a fixed-speed checkpoint worth
distilling or transferring from, because old 5-action checkpoints cannot be
loaded directly into the 15-action decoder without an explicit transfer step.

Native backend build commands on this macOS machine:

```bash
PATH="$PWD/.venv/bin:$PATH" \
CC=/opt/homebrew/Cellar/llvm/19.1.7_1.reinstall/bin/clang \
CXX=/opt/homebrew/Cellar/llvm/19.1.7_1.reinstall/bin/clang++ \
./build.sh field_nav_camera_native_hard --float --cpu

PATH="$PWD/.venv/bin:$PATH" \
CC=/opt/homebrew/Cellar/llvm/19.1.7_1.reinstall/bin/clang \
CXX=/opt/homebrew/Cellar/llvm/19.1.7_1.reinstall/bin/clang++ \
./build.sh field_nav_camera_native_hard_speed --float --cpu
```

The compiled native extension is env-specific. Rebuild when switching between
`field_nav_camera_native_hard` and `field_nav_camera_native_hard_speed`, or the
trainer will reject the run with a `build.sh was run for ...` assertion.

### Stage17 - Fixed-Speed Blocked-Corridor Curriculum

- source checkpoint: Stage13 best RR3 checkpoint
  `navigation/artifacts/front_camera_native_stage13_hard_100fov_rr3_30m_checkpoints/field_nav_camera_native_hard/1778504318206/0000000024625152.bin`
- run id: `1778507597209`
- change tested:
  - `blocked_corridor_prob = 0.30`
  - `blocked_corridor_wall_prob = 0.60`
  - `near_obstacle_threshold_m = 4.0`
  - replay schedule `1.0 -> 2.0` over `15M`, LR `2e-4`
- dashboard URL during run: `http://100.120.184.115:8772`
- best live training window:
  - step: `8,257,536`
  - checkpoint:
    `navigation/artifacts/front_camera_native_stage17_blocked_corridor_30m_checkpoints/field_nav_camera_native_hard/1778507597209/0000000008650752.bin`
  - success: `88.00%`
  - score: `15.14`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage17_blocked_corridor_30m_checkpoints/field_nav_camera_native_hard/1778507597209/0000000029982720.bin`
- final live training window:
  - success: `73.92%`
  - score: `11.40`

Fixed-seed evals:

| policy/eval | success | collision | timeout | mean return | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Stage13 normal hard baseline | `92.00%` | `8.00%` | `0.00%` | `32.84` | previous best |
| Stage17 best normal hard | `88.67%` | `7.33%` | `4.00%` | `30.97` | safer but stalls |
| Stage17 final normal hard | `88.33%` | `11.67%` | `0.00%` | `31.23` | below Stage13 |
| Stage13 mixed blocked corridor | `69.50%` | `30.50%` | `0.00%` | `23.37` | baseline failure mode |
| Stage17 best mixed blocked corridor | `44.00%` | `0.50%` | `55.50%` | `13.18` | over-avoids and times out |
| Stage17 final mixed blocked corridor | `83.50%` | `16.50%` | `0.00%` | `31.61` | target improvement |
| Stage13 wall-only blocked corridor | `62.00%` | `38.00%` | `0.00%` | `19.16` | hardest case |
| Stage17 final wall-only blocked corridor | `70.00%` | `29.00%` | `1.00%` | `24.46` | improved but missed target |
| Stage13 tree-only blocked corridor | `75.00%` | `25.00%` | `0.00%` | `26.66` | baseline |
| Stage17 final tree-only blocked corridor | `87.00%` | `13.00%` | `0.00%` | `33.57` | strong improvement |

Interpretation:

- The curriculum worked for the targeted blocked-corridor distribution,
  especially tree rows.
- The `4m` near-obstacle threshold plus `30%` blocked-corridor probability is
  too strong for retaining normal hard-map performance.
- Stage17 final should not replace Stage13 as the general current best policy,
  but it is the best checkpoint so far for blocked-corridor behavior.

Next adjustment:

- Continue from Stage13 or Stage17 final with a milder mix:
  - `blocked_corridor_prob = 0.15-0.20`
  - `near_obstacle_threshold_m = 3.0`
  - keep `blocked_corridor_wall_prob = 0.60`
- Success target remains: normal hard `>= 92%`, mixed blocked corridor `> 80%`,
  wall-only blocked corridor `> 75%`.

### Stage18 - Fresh Speed-Control 100-FOV Hard Run

- env: `field_nav_camera_native_hard_speed`
- run id: `1778508278333`
- setup:
  - trained from scratch because the action decoder changed from fixed-speed
    `5` actions to speed-control `15` actions
  - `100 deg` FOV
  - total steps: `25M`
  - LR `2e-4`
  - replay schedule `1.0 -> 2.0` over `12.5M`
- dashboard URL during run: `http://100.120.184.115:8773`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage18_speed_control_100fov_25m_checkpoints/field_nav_camera_native_hard_speed/1778508278333/0000000024969216.bin`
- best live window:
  - step: `98,304`
  - success: `4.44%`
  - score: `-44.78`
  - collision: `85.56%`
  - entropy: `2.707`
- final live window:
  - step: `24,969,216`
  - success: `0.00%`
  - score: `-39.39`
  - collision: `49.71%`
  - entropy: `2.707`

Interpretation:

- From-scratch speed-control on the full hard 100-FOV task did not learn in
  `25M` steps.
- Entropy stayed at about `ln(15) = 2.708`, so the policy remained nearly
  uniform over the 15 actions.
- The issue is probably task/bootstrap difficulty rather than throughput.

Next speed-control attempt should not start on the full hard task:

- train `field_nav_camera_native_hard_speed` on an easier speed-control
  bootstrap first:
  - no blocked corridor
  - no people/walls
  - shorter `5-12m` goals
  - sparse tree rows
  - maybe `max_speed_mps = 1.0` or `1.2`
- once success is high, continue into normal hard and then blocked-corridor
  hard.

### Stage19 - Sparse Speed-Control Bootstrap

- env: `field_nav_camera_native_hard_speed`
- run id: `1778508661520`
- setup:
  - fresh 15-action speed-control run
  - `100 deg` FOV
  - `25M` total steps
  - LR `2e-4`
  - replay schedule `1.0 -> 2.0` over `12.5M`
  - easier distribution: `5-12m` goals, sparse static obstacles, no people,
    no walls, no blocked corridors
- dashboard URL during run: `http://100.120.184.115:8774`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage19_speed_bootstrap_25m_checkpoints/field_nav_camera_native_hard_speed/1778508661520/0000000024969216.bin`
- best live window:
  - step: `98,304`
  - success: `21.43%`
  - score: `-32.09`
  - collision: `57.14%`
- final live window:
  - step: `24,969,216`
  - success: `4.46%`
  - score: `-26.85`
  - collision: `7.14%`
  - SPS: `86.8K`

Interpretation:

- The early best was random variance with a high collision rate, not a useful
  policy.
- The sparse bootstrap still did not produce learning; live entropy stayed near
  `ln(15)`, KL stayed near zero, and most episodes timed out.

### Stage20 - Empty Short-Goal Speed-Control Bootstrap

- env: `field_nav_camera_native_hard_speed`
- run id: `1778508945822`
- setup:
  - fresh 15-action speed-control run
  - `100 deg` FOV
  - `25M` total steps
  - LR `2e-4`
  - replay schedule `1.0 -> 2.0` over `12.5M`
  - empty map, no obstacles, no blocked corridors
  - `3-6m` goals, `150` max steps, `max_speed_mps = 1.0`
- dashboard URL during run: `http://100.120.184.115:8775`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage20_speed_empty_short_25m_checkpoints/field_nav_camera_native_hard_speed/1778508945822/0000000024969216.bin`
- best live window:
  - step: `5,799,936`
  - success: `9.71%`
  - score: `-8.99`
  - collision: `0.00%`
  - SPS: `114.3K`
- final live window:
  - step: `24,969,216`
  - success: `5.03%`
  - score: `-9.42`
  - collision: `0.00%`
  - SPS: `92.4K`

Interpretation:

- Even the empty short-goal bootstrap did not learn.
- Because there are no obstacles and no collisions, this is not an obstacle
  curriculum problem.
- The current speed-control setup is failing to move the policy distribution:
  entropy stayed effectively uniform and KL stayed near zero.
- Next diagnostic should remove entropy pressure and use a larger LR on the
  same empty short-goal task before changing the environment further.

### Stage21 - Empty Speed-Control Optimizer Diagnostic

- env: `field_nav_camera_native_hard_speed`
- run id: `1778509239682`
- setup:
  - same empty-map, `3-6m` goal task as Stage20
  - `5M` total steps
  - LR `1e-3`
  - replay ratio fixed at `1.0`
  - entropy coefficient `0.0`
- dashboard:
  - a new Tailscale dashboard bind for this run was blocked by the local
    approval system; metrics were monitored from the training session/logs
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage21_speed_empty_lr1e3_noent_5m_checkpoints/field_nav_camera_native_hard_speed/1778509239682/0000000004964352.bin`
- best live window:
  - step: `491,520`
  - success: `8.74%`
  - score: `-9.54`
  - collision: `0.00%`
  - SPS: `147.8K`
- final live window:
  - step: `4,964,352`
  - success: `6.95%`
  - score: `-9.14`
  - collision: `0.00%`
  - SPS: `143.3K`

Interpretation:

- Removing entropy pressure and increasing LR still did not move the policy
  distribution enough to learn the empty task.
- This points to a speed-control environment/training-interface issue or a
  missing dense learning signal, not just a bad curriculum schedule.
- Before spending more long runs, inspect action-head gradients/logits and
  verify a supervised or scripted action can solve the empty speed-control env.

### Stage22 - 100M Sparse Speed-Control Bootstrap

- env: `field_nav_camera_native_hard_speed`
- run id: `1778586817032`
- setup:
  - fresh 15-action speed-control run
  - `100 deg` FOV
  - `100M` total steps
  - LR `2e-4`
  - replay schedule `1.0 -> 2.0` over `50M`
  - same sparse distribution as Stage19:
    - `5-12m` goals
    - `0-1` tree rows
    - `2-5` bushes
    - `1-3` potholes
    - no people, no walls, no blocked corridors
- dashboard URL during run: `http://100.120.184.115:8776`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage22_speed_sparse_bootstrap_100m_checkpoints/field_nav_camera_native_hard_speed/1778586817032/0000000099975168.bin`
- best success checkpoint:
  `navigation/artifacts/front_camera_native_stage22_speed_sparse_bootstrap_100m_checkpoints/field_nav_camera_native_hard_speed/1778586817032/0000000095895552.bin`
- best score checkpoint:
  `navigation/artifacts/front_camera_native_stage22_speed_sparse_bootstrap_100m_checkpoints/field_nav_camera_native_hard_speed/1778586817032/0000000090980352.bin`
- live windows:

| window | step | success | collision | score | entropy | explained var | SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best success | `95,944,704` | `22.88%` | `1.69%` | `-2.59` | `2.692` | `0.263` | `86.2K` |
| best score | `90,832,896` | `19.84%` | `1.59%` | `-2.43` | `2.691` | `0.203` | `85.3K` |
| final | `99,975,168` | `13.25%` | `4.64%` | `-6.08` | `2.692` | `0.246` | `82.9K` |

Interpretation:

- Running the sparse bootstrap for `100M` steps did improve over the failed
  `25M` sparse run, but it is still far from a usable bootstrap.
- The best live window reached only `22.88%` success on the sparse task.
- Entropy stayed very high (`2.69` vs `ln(15) = 2.708`) and KL stayed near
  zero, so the policy distribution barely specialized even after `100M`.
- More steps alone is not the right lever. The next useful work is to debug the
  speed-control learning path directly: scripted-controller sanity check,
  action-logit/gradient inspection, then likely stronger action/reward shaping
  or a simpler action factorization before returning to obstacle curricula.

### Stage23 - Higher-LR Sparse Speed-Control Continuation

- env: `field_nav_camera_native_hard_speed`
- run id: `1778588009158`
- setup:
  - loaded Stage22 best-success checkpoint:
    `navigation/artifacts/front_camera_native_stage22_speed_sparse_bootstrap_100m_checkpoints/field_nav_camera_native_hard_speed/1778586817032/0000000095895552.bin`
  - target was `50M` continuation steps, stopped early at `33.18M` after
    success saturated above `98%`
  - LR `5e-4`
  - replay ratio fixed at `2.0`
  - same sparse distribution as Stage22
- dashboard URL during run: `http://100.120.184.115:8777`
- best success checkpoint:
  `navigation/artifacts/front_camera_native_stage23_speed_sparse_lr5e4_50m_checkpoints/field_nav_camera_native_hard_speed/1778588009158/0000000019709952.bin`
- best score checkpoint:
  `navigation/artifacts/front_camera_native_stage23_speed_sparse_lr5e4_50m_checkpoints/field_nav_camera_native_hard_speed/1778588009158/0000000031997952.bin`
- final saved checkpoint:
  `navigation/artifacts/front_camera_native_stage23_speed_sparse_lr5e4_50m_checkpoints/field_nav_camera_native_hard_speed/1778588009158/0000000033226752.bin`
- live windows:

| window | step | success | collision | score | entropy | explained var | SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best success | `19,316,736` | `99.46%` | `0.54%` | `28.01` | `2.405` | `0.323` | `88.8K` |
| best score | `32,489,472` | `98.86%` | `1.14%` | `28.84` | `2.171` | `0.232` | `87.7K` |
| final | `33,177,600` | `98.23%` | `1.77%` | `28.32` | `2.169` | `0.295` | `87.8K` |

Interpretation:

- Higher LR was the missing lever for the sparse speed-control bootstrap.
- Continuing the weak Stage22 policy with LR `5e-4` rapidly turned it into a
  high-success sparse policy.
- The best-success checkpoint is the safest sparse bootstrap seed; the
  best-score checkpoint may be slightly faster/more decisive but has marginally
  more collision.
- Next step should continue from Stage23 best-success into a harder curriculum:
  add people/walls gradually or move to the normal hard distribution before
  reintroducing blocked corridors.

### Stage24 - Hard Curriculum With Too-Slow Warmup

- env: `field_nav_camera_native_hard_speed`
- run id: `1778588531004`
- setup:
  - loaded Stage23 best-success checkpoint
  - target was `50M`, stopped early at `32.19M`
  - LR `5e-4`, replay ratio fixed at `2.0`
  - native curriculum enabled toward normal hard distribution
  - `curriculum_warmup_steps = 30,000,000`
- dashboard URL during run: `http://100.120.184.115:8778`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage24_speed_hard_curriculum_50m_checkpoints/field_nav_camera_native_hard_speed/1778588531004/0000000031997952.bin`

| window | step | success | collision | score | curriculum | entropy | explained var | SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best success | `5,505,024` | `95.31%` | `4.69%` | `26.07` | `0.001` | - | - | - |
| best score | `32,096,256` | `93.49%` | `0.47%` | `26.22` | `0.004` | - | - | - |
| final | `32,194,560` | `90.81%` | `1.04%` | `23.92` | `0.004` | `1.953` | `0.883` | `80.4K` |

Interpretation:

- The run did not meaningfully reach harder curriculum.
- The native `curriculum_warmup_steps` knob is measured in per-env lifetime
  steps, not global agent steps.
- With `256` agents, `30M` per-env warmup would require about `7.68B` global
  agent steps to finish.

### Stage25 - Hard Curriculum With 5M Per-Env Warmup

- env: `field_nav_camera_native_hard_speed`
- run id: `1778589202291`
- setup:
  - loaded Stage23 best-success checkpoint
  - `25M` total steps
  - LR `5e-4`, replay ratio fixed at `2.0`
  - native curriculum enabled toward normal hard distribution
  - `curriculum_warmup_steps = 5,000,000`
- dashboard URL during run: `http://100.120.184.115:8779`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage25_speed_hard_curriculum_5mwarmup_25m_checkpoints/field_nav_camera_native_hard_speed/1778589202291/0000000024969216.bin`
- best checkpoint:
  `navigation/artifacts/front_camera_native_stage25_speed_hard_curriculum_5mwarmup_25m_checkpoints/field_nav_camera_native_hard_speed/1778589202291/0000000003735552.bin`

| window | step | success | collision | score | curriculum | entropy | explained var | SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best success/score | `4,177,920` | `95.71%` | `4.29%` | `26.34` | `0.003` | `2.355` | `0.447` | `84.3K` |
| final | `24,969,216` | `91.12%` | `2.35%` | `24.56` | `0.019` | `1.947` | `0.842` | `79.3K` |

Interpretation:

- `5M` per-env warmup was still too slow. At `256` agents it corresponds to
  roughly `1.28B` global agent steps.
- For a true `5M` global-step warmup, the per-env warmup should be about
  `5,000,000 / 256 = 19,531`.

### Stage26 - Corrected 20K Per-Env Hard Curriculum

- env: `field_nav_camera_native_hard_speed`
- run id: `1778589621006`
- setup:
  - loaded Stage23 best-success checkpoint
  - `25M` total steps
  - LR `5e-4`, replay ratio fixed at `2.0`
  - native curriculum enabled toward normal hard distribution
  - `curriculum_warmup_steps = 20,000`, about `5.1M` global steps with `256`
    agents
- dashboard URL during run: `http://100.120.184.115:8780`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage26_speed_hard_curriculum_20kperenv_25m_checkpoints/field_nav_camera_native_hard_speed/1778589621006/0000000024969216.bin`
- best full-hard checkpoint:
  `navigation/artifacts/front_camera_native_stage26_speed_hard_curriculum_20kperenv_25m_checkpoints/field_nav_camera_native_hard_speed/1778589621006/0000000023396352.bin`

| window | step | success | collision | score | curriculum | entropy | explained var | SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| early best success | `196,608` | `93.97%` | `6.03%` | `25.41` | `0.034` | `2.395` | `0.338` | `80.5K` |
| best full hard | `22,855,680` | `71.94%` | `9.35%` | `13.95` | `1.000` | `2.016` | `0.864` | `73.0K` |
| final full hard | `24,969,216` | `65.45%` | `11.64%` | `10.10` | `1.000` | `2.018` | `0.850` | `71.3K` |

Interpretation:

- Corrected warmup reached the full hard distribution by about `6M` global
  steps, exposing the real difficulty jump.
- The policy recovered from the distribution shift but plateaued around
  `65-72%` success on full hard within `25M`.
- Next logical curriculum should be less abrupt than Stage26, for example
  `40K-80K` per-env warmup or a staged target that adds people/walls before
  increasing goal range to `10-22m`.

### Stage27 - Full-Hard Continuation With LR 7e-4

- env: `field_nav_camera_native_hard_speed`
- run id: `1778590198696`
- setup:
  - loaded Stage26 best full-hard checkpoint:
    `navigation/artifacts/front_camera_native_stage26_speed_hard_curriculum_20kperenv_25m_checkpoints/field_nav_camera_native_hard_speed/1778589621006/0000000023396352.bin`
  - `50M` total steps
  - full hard distribution from the start, curriculum disabled
  - no blocked corridors
  - LR increased from `5e-4` to `7e-4`
  - replay ratio fixed at `2.0`
- dashboard URL during run: `http://100.120.184.115:8781`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage27_speed_full_hard_lr7e4_50m_checkpoints/field_nav_camera_native_hard_speed/1778590198696/0000000049987584.bin`
- best logged checkpoint:
  `navigation/artifacts/front_camera_native_stage27_speed_full_hard_lr7e4_50m_checkpoints/field_nav_camera_native_hard_speed/1778590198696/0000000000049152.bin`

| window | step | success | collision | score | entropy | explained var | SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best logged | `49,152` | `92.54%` | `7.89%` | `31.27` | `2.030` | `0.835` | `68.3K` |
| final | `49,987,584` | `78.44%` | `3.13%` | `23.69` | `1.494` | `0.904` | `70.4K` |

Interpretation:

- LR `7e-4` did not destabilize the run, and late training improved over
  Stage26 final.
- The first logged window was the best by success/score, probably reflecting
  the loaded Stage26 checkpoint before continued full-hard training changed the
  policy.
- By the end, collision was much lower than Stage26 best full-hard, but success
  was still only `78.44%`.
- Full-hard speed-control is now learning, but it is still below the fixed-speed
  Stage13 hard baseline. Next experiments should either evaluate Stage27
  checkpoints with fixed seeds or try a milder LR/entropy schedule to retain the
  high initial success while reducing collision.

### Stage28 - Full-Hard Continuation With LR 5e-4

- env: `field_nav_camera_native_hard_speed`
- run id: `1778591099678`
- setup:
  - loaded Stage27 final checkpoint:
    `navigation/artifacts/front_camera_native_stage27_speed_full_hard_lr7e4_50m_checkpoints/field_nav_camera_native_hard_speed/1778590198696/0000000049987584.bin`
  - `50M` total steps
  - full hard distribution from the start, curriculum disabled
  - no blocked corridors
  - LR reduced from `7e-4` to `5e-4`
  - replay ratio fixed at `2.0`
- dashboard URL during run: `http://100.120.184.115:8782`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage28_speed_full_hard_lr5e4_50m_checkpoints/field_nav_camera_native_hard_speed/1778591099678/0000000049987584.bin`
- best logged checkpoint:
  `navigation/artifacts/front_camera_native_stage28_speed_full_hard_lr5e4_50m_checkpoints/field_nav_camera_native_hard_speed/1778591099678/0000000000049152.bin`

| window | step | success | collision | score | entropy | explained var | SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best logged | `98,304` | `94.88%` | `5.12%` | `32.30` | `1.493` | `0.901` | `75.0K` |
| final | `49,987,584` | `78.27%` | `3.62%` | `28.84` | `1.008` | `0.928` | `70.3K` |

Interpretation:

- Stage28 ended almost identical to Stage27 on success, with slightly higher
  collision but better score.
- The best live windows continue to be at the start of the continuation,
  suggesting the loaded policy has a high-success mode that continued PPO
  training quickly shifts away from.
- Continuing full-hard PPO alone is not currently improving beyond the
  `~78%` final plateau.
- Before another long continuation, run fixed-seed evals of the best early
  checkpoints versus final checkpoints, then consider lower LR, no LR warmup
  reset, lower entropy coefficient, or checkpoint selection by fixed-seed eval
  instead of live training windows.

### Fixed-Seed Full-Hard Speed-Control Checkpoint Eval

All candidates were evaluated with `navigation/scripts/render_policy.py` on the
same `300` seeds starting at `920000`, `100` degree FOV, full hard camera env,
`400` max steps, and reset clearances matching training. The render helper's
hard speed-control eval path uses `max_speed_mps = 1.2` to match training.

| checkpoint | success | collision | timeout | mean return |
| --- | ---: | ---: | ---: | ---: |
| Stage26 best full-hard | `58.00%` | `6.33%` | `35.67%` | `20.82` |
| Stage27 early live-best | `58.33%` | `6.00%` | `35.67%` | `21.01` |
| Stage27 final | `77.00%` | `4.33%` | `18.67%` | `26.33` |
| Stage28 early live-best | `77.33%` | `4.00%` | `18.67%` | `26.69` |
| Stage28 final | `79.33%` | `5.67%` | `15.00%` | `28.82` |
| Stage29 final | `79.33%` | `4.67%` | `16.00%` | `29.03` |
| Stage30 final | `78.00%` | `4.67%` | `17.33%` | `28.83` |
| Stage31 final | `80.00%` | `4.67%` | `15.33%` | `29.51` |
| Stage32 final | `80.33%` | `4.67%` | `15.00%` | `29.48` |

Artifacts:

- `navigation/artifacts/policy_evals/stage26_best_fixedhard_e300_s920000`
- `navigation/artifacts/policy_evals/stage27_early_fixedhard_e300_s920000`
- `navigation/artifacts/policy_evals/stage27_final_fixedhard_e300_s920000`
- `navigation/artifacts/policy_evals/stage28_early_fixedhard_e300_s920000`
- `navigation/artifacts/policy_evals/stage28_final_fixedhard_e300_s920000`
- `navigation/artifacts/policy_evals/stage29_final_fixedhard_e300_s920000`
- `navigation/artifacts/policy_evals/stage30_final_fixedhard_e300_s920000`
- `navigation/artifacts/policy_evals/stage31_final_fixedhard_e300_s920000`
- `navigation/artifacts/policy_evals/stage32_final_fixedhard_e300_s920000`

Interpretation:

- The early live-best spikes in Stage27/Stage28 did not hold up on fixed seeds.
- Stage28 final was the best verified starting point for another continuation.
- Stage29 did not increase success above Stage28 final on fixed seeds, but it
  reduced collision by `1.00` percentage point and slightly improved mean return.
- Stage30/31 added blocked-corridor exposure. Stage32 is the new best verified
  normal-hard checkpoint so far at `80.33%` success, while Stage31 has the
  cleanest blocked-corridor result with zero collisions.

### Stage29 - Low-LR Full-Hard Continuation With Replay 1.0

- env: `field_nav_camera_native_hard_speed`
- run id: `1778593966975`
- setup:
  - loaded Stage28 final checkpoint:
    `navigation/artifacts/front_camera_native_stage28_speed_full_hard_lr5e4_50m_checkpoints/field_nav_camera_native_hard_speed/1778591099678/0000000049987584.bin`
  - `10M` total steps
  - full hard distribution from the start, curriculum disabled
  - no blocked corridors
  - LR reduced to `1e-4`
  - replay ratio fixed at `1.0`
- dashboard URL during run: `http://100.120.184.115:8783`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage29_speed_full_hard_lr1e4_replay1_10m_checkpoints/field_nav_camera_native_hard_speed/1778593966975/0000000009977856.bin`

| window | step | success | collision | score | entropy | explained var | SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best live success | `8,847,360` | `83.58%` | `3.23%` | `29.80` | `1.019` | `0.896` | `104.7K` |
| final | `9,977,856` | `77.09%` | `5.31%` | `26.58` | `0.991` | `0.896` | `101.4K` |

Interpretation:

- Lower LR and replay `1.0` did not break the policy and improved throughput to
  about `100K SPS`.
- Fixed-seed eval shows Stage29 is roughly a lateral move from Stage28 final:
  same success, lower collision, slightly higher return.
- The current plateau is likely not solved by simply continuing PPO on the same
  distribution. Next useful changes should target data distribution or objective:
  blocked-corridor/late-detour curriculum, explicit slowdown/clearance reward,
  or checkpoint selection by periodic fixed-seed eval instead of noisy live
  windows.

### Stage30 - Blocked-Corridor Mix Continuation

- env: `field_nav_camera_native_hard_speed`
- run id: `1778594628035`
- setup:
  - loaded Stage29 final checkpoint:
    `navigation/artifacts/front_camera_native_stage29_speed_full_hard_lr1e4_replay1_10m_checkpoints/field_nav_camera_native_hard_speed/1778593966975/0000000009977856.bin`
  - `15M` total steps
  - full hard distribution with `blocked_corridor_prob = 0.25`
  - LR `1e-4`, replay ratio fixed at `1.0`
- dashboard URL during run: `http://100.120.184.115:8784`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage30_speed_blockedmix_lr1e4_replay1_15m_checkpoints/field_nav_camera_native_hard_speed/1778594628035/0000000014991360.bin`

| eval | success | collision | timeout | mean return |
| --- | ---: | ---: | ---: | ---: |
| normal hard fixed-seed | `78.00%` | `4.67%` | `17.33%` | `28.83` |
| blocked corridor | `74.67%` | `0.33%` | `25.00%` | `29.97` |

Interpretation:

- The `25%` blocked-corridor mix trained useful detour behavior: blocked eval
  reached `74.67%` with only one collision in `300` episodes.
- It cost normal-hard success versus Stage29 by `1.33` percentage points.
- Next step was to reduce blocked-corridor mix while keeping the learned detour
  skill.

### Stage31 - Lower Blocked-Corridor Mix Repair

- env: `field_nav_camera_native_hard_speed`
- run id: `1778675858382`
- setup:
  - loaded Stage30 final checkpoint:
    `navigation/artifacts/front_camera_native_stage30_speed_blockedmix_lr1e4_replay1_15m_checkpoints/field_nav_camera_native_hard_speed/1778594628035/0000000014991360.bin`
  - `15M` total steps
  - full hard distribution with `blocked_corridor_prob = 0.10`
  - LR `5e-5`, replay ratio fixed at `1.0`
- dashboard URL during run: `http://100.120.184.115:8785`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage31_speed_blocked10_lr5e5_replay1_15m_checkpoints/field_nav_camera_native_hard_speed/1778675858382/0000000014991360.bin`

| eval | success | collision | timeout | mean return |
| --- | ---: | ---: | ---: | ---: |
| normal hard fixed-seed | `80.00%` | `4.67%` | `15.33%` | `29.51` |
| blocked corridor | `76.00%` | `0.00%` | `24.00%` | `30.16` |

Interpretation:

- Stage31 improved both benchmarks versus Stage30: normal-hard fixed-seed
  success recovered to a new best `80.00%`, and blocked-corridor success
  improved to `76.00%` with zero collisions.
- This suggests the blocked-corridor skill is useful, but the mix must stay
  low enough to avoid overfitting away from the normal hard distribution.
- Next logical iteration is an even lighter retention mix, for example
  `blocked_corridor_prob = 0.05` with LR `5e-5` or `2.5e-5`.

### Stage32 - Very Low Blocked-Corridor Retention Mix

- env: `field_nav_camera_native_hard_speed`
- run id: `1778676408116`
- setup:
  - loaded Stage31 final checkpoint:
    `navigation/artifacts/front_camera_native_stage31_speed_blocked10_lr5e5_replay1_15m_checkpoints/field_nav_camera_native_hard_speed/1778675858382/0000000014991360.bin`
  - `15M` total steps
  - full hard distribution with `blocked_corridor_prob = 0.05`
  - LR `5e-5`, replay ratio fixed at `1.0`
- dashboard URL during run: `http://100.120.184.115:8786`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage32_speed_blocked05_lr5e5_replay1_15m_checkpoints/field_nav_camera_native_hard_speed/1778676408116/0000000014991360.bin`

| eval | success | collision | timeout | mean return |
| --- | ---: | ---: | ---: | ---: |
| normal hard fixed-seed | `80.33%` | `4.67%` | `15.00%` | `29.48` |
| blocked corridor | `76.33%` | `0.33%` | `23.33%` | `29.85` |

Interpretation:

- Stage32 is the best normal-hard fixed-seed checkpoint so far by success,
  slightly improving Stage31 from `80.00%` to `80.33%`.
- Blocked-corridor success also increased slightly, but one wall collision
  returned. Stage31 remains preferable if zero blocked-corridor collisions is
  prioritized.
- The best overall checkpoint depends on selection metric:
  - normal-hard success: Stage32 final
  - blocked-corridor safety: Stage31 final
  - balanced default: Stage32 final, because it has the best normal-hard
    success and only one collision in `300` blocked-corridor eval episodes.

### Stage33 - Higher-LR Continuation From Stage32

- env: `field_nav_camera_native_hard_speed`
- run id: `1778676703657`
- setup:
  - loaded Stage32 final checkpoint:
    `navigation/artifacts/front_camera_native_stage32_speed_blocked05_lr5e5_replay1_15m_checkpoints/field_nav_camera_native_hard_speed/1778676408116/0000000014991360.bin`
  - requested `50M` total steps, interrupted before completion at about `48.37M`
  - full hard distribution with `blocked_corridor_prob = 0.05`
  - LR increased to `1e-4`, replay ratio fixed at `1.0`
- dashboard URL during run: `http://100.120.184.115:8787`
- latest checkpoint:
  `navigation/artifacts/front_camera_native_stage33_speed_blocked05_lr1e4_replay1_50m_checkpoints/field_nav_camera_native_hard_speed/1778676703657/0000000047972352.bin`

| window | step | success | collision | score | SPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| latest live | `48,365,568` | `78.73%` | `3.81%` | `28.35` | `102.7K` |

Interpretation:

- Doubling LR from the Stage32 setting did not clearly improve the live
  policy. The live success stayed below Stage32's fixed-seed result.
- PPO update diagnostics still looked very small (`approx_kl` near zero,
  `clipfrac = 0`), so the next test increased both LR and replay ratio to
  check whether stronger sustained updates would move the policy.

### Stage34 - Stronger PPO Updates From Stage32

- env: `field_nav_camera_native_hard_speed`
- run id: `1778699808021`
- setup:
  - loaded Stage32 final checkpoint:
    `navigation/artifacts/front_camera_native_stage32_speed_blocked05_lr5e5_replay1_15m_checkpoints/field_nav_camera_native_hard_speed/1778676408116/0000000014991360.bin`
  - `25M` total steps
  - full hard distribution with `blocked_corridor_prob = 0.05`
  - LR increased to `2e-4`, `min_lr_ratio = 0.75`
  - replay ratio fixed at `2.0`
- dashboard URL during run: `http://100.120.184.115:8788`
- final checkpoint:
  `navigation/artifacts/front_camera_native_stage34_speed_blocked05_lr2e4_replay2_25m_checkpoints/field_nav_camera_native_hard_speed/1778699808021/0000000024969216.bin`
- eval artifacts:
  - normal hard:
    `navigation/artifacts/policy_evals/stage34_final_fixedhard_e300_s920000/`
  - blocked corridor:
    `navigation/artifacts/blocked_corridor_evals/stage34_final_blocked_e300_s910000/`

| eval | success | collision | timeout | mean return |
| --- | ---: | ---: | ---: | ---: |
| normal hard fixed-seed | `80.00%` | `4.67%` | `15.33%` | `30.34` |
| blocked corridor | `75.00%` | `0.00%` | `25.00%` | `30.97` |

| final train metric | value |
| --- | ---: |
| live success | `80.65%` |
| live collision | `5.28%` |
| live score | `29.28` |
| SPS | `72.0K` |
| replay ratio | `2.0` |
| minibatches | `2` |
| approx KL | `0.0000025` |
| clipfrac | `0.0` |

Interpretation:

- Stage34 did not beat Stage32 on normal-hard fixed-seed success:
  `80.00%` versus Stage32 `80.33%`.
- It also did not beat the best blocked-corridor success: Stage31 reached
  `76.00%`, Stage32 reached `76.33%`, while Stage34 reached `75.00%`.
- Replay ratio `2.0` worked mechanically and cut throughput from about
  `100K SPS` to about `72K SPS`, but it still produced tiny PPO updates
  (`approx_kl` near zero and no clipping). The plateau is unlikely to be fixed
  by more of the same optimizer pressure alone.
- Current default best remains Stage32 final. Stage31 remains the safest
  blocked-corridor checkpoint by zero collisions and slightly higher blocked
  success than Stage34.

### Stage35 / CV2 - From-Scratch Improved Curriculum Attempt

Implementation change:

- Added explicit native curriculum-start knobs to
  `ocean/field_nav_camera_native_hard/field_nav.h` and the hard/hard-speed
  bindings:
  - start obstacle ranges for trees, bushes, potholes, people, and walls
  - start goal-distance range
  - scheduled blocked-corridor probability
- This fixes the old limitation where `curriculum_enabled = 1` still began
  with non-trivial hazards (`1-2` tree rows, `3-8` bushes, `1-4` potholes),
  which was too hard for a cold 15-action speed-control policy.
- Rebuilt native backend:
  `./build.sh field_nav_camera_native_hard_speed --float --cpu`

CV2-001:

- env: `field_nav_camera_native_hard_speed`
- run id: `1779362521716`
- setup:
  - from scratch
  - `50M` requested, stopped early at about `13.9M`
  - LR `2e-4`, replay schedule `1.0 -> 2.0`
  - old built-in curriculum start, `blocked_corridor_prob = 0.08`
- dashboard URL during run: `http://100.120.184.115:8789`
- reason stopped:
  - entropy stayed uniform at about `2.707`
  - success stayed around a few percent
  - curriculum progress reached about `0.36`, meaning the task was ramping
    while the policy had not learned the easy behavior

CV2-002:

- env: `field_nav_camera_native_hard_speed`
- run id: `1779362764256`
- setup:
  - from scratch
  - `50M` requested, stopped early at `12.53M`
  - corrected sparse-start curriculum:
    - start goals `3-7m`
    - start trees `0-0`
    - start bushes `0-1`
    - start potholes/people/walls `0`
    - blocked-corridor probability ramps from `0.0` to `0.08`
    - full hard target remains the normal hard-speed distribution
  - LR `5e-4`, replay ratio fixed at `2.0`, entropy coefficient `0.002`
- dashboard URL during run: `http://100.120.184.115:8790`
- latest checkpoint before stop:
  `navigation/artifacts/front_camera_native_stage35_cv2_scratch_sparse_start_lr5e4_50m_checkpoints/field_nav_camera_native_hard_speed/1779362764256/0000000012337152.bin`

| window | step | success | collision | score | curriculum | entropy | approx KL | SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stopped | `12,533,760` | `5.76%` | `20.14%` | `-30.87` | `0.326` | `2.704` | `0.00000003` | `85.8K` |

Interpretation:

- The improved curriculum implementation worked mechanically: CV2-002 started
  from genuinely sparse/short-goal layouts and ramped toward hard.
- It still did not produce a usable from-scratch speed-control policy. Entropy
  remained near `ln(15)`, KL stayed effectively zero, and success was still
  single-digit after `12.5M` steps.
- This strengthens the earlier conclusion from Stages20-22: the blocker is not
  only obstacle curriculum. The 15-action speed-control policy does not reliably
  specialize from scratch under the current PPO/reward setup.
- The next useful experiment should not be another from-scratch hard curriculum.
  Use either:
  - a shaped/supervised speed-control bootstrap on the empty short-goal task,
    or
  - the known Stage23 sparse-bootstrap checkpoint as the starting point, then
    apply the new sparse-start curriculum toward full hard.
