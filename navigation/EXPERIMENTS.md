# Field Nav Experiment Log

This file tracks the main `field_nav` runs we used to compare observation layouts, throughput, and learning quality.

## 1. Polar SPS Benchmark

Setup:

- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- curriculum warmup: enabled
- training steps: `24,576`

Results:

- `SPS`: `22,156`
- final score: `-33.53`
- success rate: `4.6%`
- collision rate: `29.2%`
- explained variance: `0.00032`
- best score during run: `-26.64`

Interpretation:

- This run was only a throughput probe.
- Polar bins are fast, and the smaller encoder cuts forward-pass cost sharply.
- Learning quality was not informative at this tiny budget.

## 2. Polar 10M Run

Setup:

- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- LR warmup: `1M` aggregate steps
- curriculum warmup: `1M` aggregate steps
- training steps: `10,000,000`

Results:

- `SPS`: `21,528`
- final score: `-24.60`
- success rate: `3.6%`
- collision rate: `81.9%`
- explained variance: `0.217`
- best score during run: `-19.14` at `8.07M` steps

Interpretation:

- The run learned late, but the policy was still poor at the end.
- The LR schedule decayed too aggressively for the back half of training.
- Curriculum helped stabilize the setup, but not enough to reach useful navigation performance.

## 3. Polar 30M Continuation

Setup:

- initialization: best saved checkpoint from the 10M run
- loaded checkpoint: `0000000007876608.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- minimum LR ratio: `0.2`
- curriculum warmup: `1M` aggregate steps
- training steps: `30,000,000`

Results:

- `SPS`: `19,468`
- final score: `14.25`
- success rate: `65.3%`
- collision rate: `34.7%`
- explained variance: `0.912`
- best score during run: `21.11` at `26.59M` steps

Interpretation:

- This was the first run that clearly converged into a usable policy.
- The higher LR floor mattered. The optimizer stayed active late in training instead of collapsing to near-zero updates.
- Polar bins preserved enough structure to learn the task while keeping throughput high.

## 4. Polar 30M Continuation, Higher LR Floor

Setup:

- initialization: best saved checkpoint from the first 30M continuation
- loaded checkpoint: `0000000026554368.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- curriculum warmup: `1M` aggregate steps
- training steps: `30,000,000`

Results:

- `SPS`: `19,299`
- final score: `22.66`
- success rate: `82.7%`
- collision rate: `17.3%`
- explained variance: `0.919`
- best score during run: `25.75` at `18.12M` steps
- best success/collision at the score peak: `88.5%` success, `13.5%` collision

Interpretation:

- This is the best polar run so far.
- Raising the LR floor from `0.2` to `0.3` improved both final policy quality and peak policy quality.
- The run peaked before the end, then partially regressed, but still finished above the previous 30M run.
- The value function is now healthy, with explained variance around `0.92`.
- Entropy fell to about `0.98`, so the policy is becoming substantially more decisive than earlier runs.

## 5. Polar 15M Full-Difficulty Refinement

Setup:

- initialization: nearest saved checkpoint to the `polar_30m2` score peak
- loaded checkpoint: `0000000018198528.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- entropy coefficient: `0.005`
- curriculum: disabled, full difficulty from the first reset
- training steps: `15,000,000`

Results:

- `SPS`: `59,355`
- final score: `21.81`
- success rate: `81.0%`
- collision rate: `19.0%`
- explained variance: `0.925`
- best score during run: `24.72` at `7.96M` steps
- best success/collision at the score peak: `83.8%` success, `16.2%` collision

Interpretation:

- The refinement run did not beat the best `polar_30m2` checkpoint.
- It preserved a good policy, but lower entropy and full-difficulty-only training did not improve the peak.
- The policy became more deterministic, with entropy dropping to about `0.76`, but that did not reduce collisions enough.
- This suggests the remaining gap is not simply solved by more conservative exploration.

## 6. Polar 15M Entropy Sweep

Setup:

- initialization: nearest saved checkpoint to the `polar_30m2` score peak
- loaded checkpoint: `0000000018198528.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- curriculum: disabled, full difficulty from the first reset
- training steps: `15,000,000` each
- varied seeds and entropy coefficients:
  - seed `1`, entropy `0.010`
  - seed `2`, entropy `0.015`
  - seed `3`, entropy `0.020`
  - seed `4`, entropy `0.025`
  - seed `5`, entropy `0.030`

Results:

| seed | entropy | final score | final success | final collision | best score | best step | best success | best collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.010 | 20.60 | 78.1% | 22.5% | 25.14 | 14.40M | 85.0% | 15.0% |
| 2 | 0.015 | 22.92 | 83.6% | 16.4% | 25.64 | 14.68M | 86.2% | 14.5% |
| 3 | 0.020 | 19.39 | 74.0% | 26.0% | 24.70 | 10.36M | 84.5% | 15.5% |
| 4 | 0.025 | 22.15 | 80.5% | 19.5% | 24.82 | 7.83M | 87.7% | 13.0% |
| 5 | 0.030 | 21.83 | 80.4% | 20.3% | 26.27 | 13.77M | 88.2% | 11.1% |

Interpretation:

- The entropy sweep found a new best score: `26.27` with entropy `0.030`.
- The best sweep run also had the lowest collision rate at its peak: `11.1%`.
- Higher entropy did not monotonically improve final score, but the highest entropy setting produced the best checkpoint.
- Final scores still lag peak scores, so best-checkpoint selection remains more important than final checkpoint selection.
- All sweep runs kept value fit healthy, with explained variance around `0.92-0.95`.

## 7. Polar 50M Continuation, Entropy 0.030

Setup:

- initialization: best checkpoint from the entropy sweep
- loaded checkpoint: `0000000013774848.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- entropy coefficient: `0.030`
- curriculum: disabled, full difficulty from the first reset
- training steps: `50,000,000`

Results:

- `SPS`: `53,030`
- final score: `26.24`
- final success rate: `88.3%`
- final collision rate: `11.7%`
- final explained variance: `0.948`
- final entropy: `1.254`
- best score during run: `27.81` at `41.88M` steps
- best success/collision at the score peak: `91.1%` success, `10.6%` collision

Interpretation:

- This is the best run so far by score, success, and collision.
- Continuing from the best entropy-sweep checkpoint was worthwhile.
- The final checkpoint is also strong, not just the peak checkpoint.
- Entropy around `1.25` remained useful; the policy did not need to become more deterministic to improve.
- Value fit stayed healthy around `0.94-0.95`, so the critic is not the bottleneck.

## 8. Second Polar 50M Continuation, Entropy 0.030

Setup:

- initialization: best checkpoint from the first 50M entropy `0.030` continuation
- loaded checkpoint: `0000000041791488.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- entropy coefficient: `0.030`
- curriculum: disabled, full difficulty from the first reset
- training steps: `50,000,000`

Results:

- `SPS`: `59,229`
- final score: `25.17`
- final success rate: `85.5%`
- final collision rate: `13.7%`
- final explained variance: `0.959`
- final entropy: `1.208`
- best score during run: `29.18` at `25.71M` steps
- best success/collision at the score peak: `93.6%` success, `6.4%` collision

Interpretation:

- This run found a large new peak, but the final checkpoint regressed.
- The best checkpoint is substantially better than the previous best: score improved from `27.81` to `29.18`.
- Collision at the peak dropped from `10.6%` to `6.4%`.
- The final checkpoint is not the right artifact to promote from this run.
- Value fit is still strong, so instability is more likely policy drift after a good region than critic failure.

## 9. Polar 50M From Peak, Lower Entropy 0.015

Setup:

- initialization: best checkpoint from the second 50M entropy `0.030` continuation
- loaded checkpoint: `0000000025817088.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- entropy coefficient: `0.015`
- curriculum: disabled, full difficulty from the first reset
- training steps: `50,000,000`

Results:

- `SPS`: `61,560`
- final score: `25.02`
- final success rate: `85.7%`
- final collision rate: `15.0%`
- final explained variance: `0.956`
- final entropy: `0.890`
- best score during run: `29.21` at `19.71M` steps
- best success/collision at the score peak: `93.2%` success, `6.1%` collision

Interpretation:

- Lowering entropy from `0.030` to `0.015` did not produce a meaningful score improvement.
- It slightly improved peak collision rate, from `6.4%` to `6.1%`, but peak success was slightly lower than the previous best.
- The run again regressed after the peak, so final checkpoint selection is still unsafe.
- Measured entropy dropped below `1.0`, confirming the policy became more deterministic.
- Current evidence favors entropy `0.030` for broad search and `0.015` only as a small refinement variant.

## Current Takeaway

Polar bins are the right observation format for this environment:

- throughput is much better than the dense `64 x 64` map
- learning quality is good once the LR schedule stays active long enough
- the higher LR floor improved the peak score from `21.11` to `25.75`
- the best run now reaches around `93.6%` success with `6.4%` collision at its score peak
- the 15M refinement confirmed that lowering entropy alone is not enough to improve the current best policy
- increasing entropy up to `0.030` did improve the best checkpoint, but not the final checkpoint
- the 50M continuation showed that longer training at entropy `0.030` can improve both peak and final quality
- the second 50M continuation showed the strongest peak yet, but also confirmed late-run drift is real
- lowering entropy to `0.015` from the best peak mostly tied the best score, with marginally lower collision but no clear overall improvement

## Proposed Next Experiment

Run a targeted checkpoint-selection/evaluation pass before spending more training:

- evaluate the top saved checkpoints from runs 8 and 9 with a larger fixed eval set
- compare score, success, collision, and episode length under the same seeds
- promote the best evaluated checkpoint, not the best training-log checkpoint

Then run one of two follow-ups from that promoted checkpoint:

Option A: continue at entropy `0.030`.

- same polar observation and encoder
- same `64` agents
- same replay ratio `2.0`
- run `25M-50M` steps
- keep `min_lr_ratio = 0.3`
- keep `lr_warmup_steps = 500k`
- keep entropy coefficient at `0.030`
- enable full-difficulty training from the start
- checkpoint frequently and select by best rolling score, not final score

Option B: test slightly higher entropy.

- run `15M-30M` steps
- set entropy coefficient to `0.035`
- keep all other settings fixed

Why this is the next useful test:

- Runs 8 and 9 are close enough that training-log noise may decide the apparent winner.
- Entropy `0.030` is proven useful; entropy `0.015` did not clearly improve it.
- A small `0.035` side run would test whether the useful range extends higher.
- Frequent checkpointing remains important because the useful artifact is likely the best checkpoint, not the final checkpoint.

## Visual Failure Analysis - Peak Policies

Artifacts:

- entropy `0.030` peak: `navigation/artifacts/render_polar_50m_ent030_cont_peak/`
- entropy `0.015` peak: `navigation/artifacts/render_polar_50m_ent015_peak/`
- each folder contains `top_policy_rollouts.png`, `best_policy_rollout.gif`, `failure_policy_rollouts.png`, `failure_policy_rollout.gif`, and `render_summary.json`

Fixed deterministic Python-env render/eval:

- checkpoint A: `polar_50m_ent030_cont`, checkpoint `0000000025817088.bin`
- checkpoint B: `polar_50m_ent015_from_peak`, checkpoint `0000000019673088.bin`
- episodes: `250`
- seeds: `200000..200249`
- max steps: `400`

Results:

| checkpoint | success | collision | timeout | mean return | immediate collisions <=4 steps | late collisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| entropy `0.030` peak | `84.4%` | `15.2%` | `0.4%` | `25.38` | `7.6%` | `7.6%` |
| entropy `0.015` peak | `86.4%` | `13.6%` | `0.4%` | `26.06` | `7.6%` | `6.0%` |

Collision types:

- entropy `0.030`: `29` tree, `8` wall, `1` person
- entropy `0.015`: `27` tree, `6` wall, `1` person

Interpretation:

- The lower-entropy peak looks slightly better on this fixed visual eval, mostly because it has fewer late collisions.
- Both policies share the same immediate-collision rate, which strongly suggests environment reset noise.
- The reset currently samples the robot and goal first, then samples obstacles independently. There is no clearance check that keeps collidable objects away from the start or goal.
- Some visible failures are collisions in `1-4` steps with the robot already inside or beside a tree/person row. Those are not useful learning signal for policy improvement.
- Timeouts are rare, so the main failure mode is collision, not indecision or inability to finish.

## Proposed Next Experiment - Reset Clearance + Fixed Eval

Before another long continuation run, test an environment cleanup:

- reject/reset object layouts where the start has less than `1.0m` clearance to collidable objects after robot radius
- reject/reset object layouts where the goal has less than `1.0m` clearance to collidable objects
- optionally reject starts where a collidable object is directly in the first `1-2m` forward arc, because the robot has fixed speed and cannot brake
- keep the reward unchanged for the first test, so the result isolates environment validity from score design
- train from the current best checkpoint for `15M-25M` steps with entropy `0.015`, same LR floor, same replay ratio, same polar observation
- promote by fixed eval success/collision on held-out seeds, not by final training score

Expected signal:

- If success jumps mainly because immediate collisions disappear, the old score was partly measuring bad resets.
- If late collisions remain around `6%`, then the next reward change should target obstacle clearance or wall/tree avoidance.
- If late collisions also drop, the policy had been learning around noisy starts and should improve further with cleaner episodes.

## Experiment 10 - Reset Clearance 25M Continuation

Purpose:

- Test whether the policy was being limited by invalid or near-invalid reset states rather than policy capacity.
- Keep the reward unchanged so the result isolates the environment reset distribution.

Implementation:

- added configurable reset clearance checks to both Python and native `field_nav`
- reject sampled object layouts where the robot start has less than `1.0m` clearance to collidable objects after robot radius
- reject sampled object layouts where the goal has less than `1.0m` clearance to collidable objects after robot radius
- reject starts with a blocked `1.5m` forward arc using a `0.25m` clearance margin
- retain the same polar observation, PPO setup, and reward coefficients

Run:

- run id: `1777285806862`
- checkpoint dir: `navigation/artifacts/polar_reset_clearance_25m_checkpoints/field_nav/1777285806862`
- log dir: `navigation/artifacts/polar_reset_clearance_25m_logs/field_nav`
- initialization: `polar_50m_ent015_from_peak`, checkpoint `0000000019673088.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- agents: `64`
- replay ratio: `2.0`
- entropy coefficient: `0.015`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- curriculum: disabled
- total steps: `25M`

Results:

- final steps: `24,993,792`
- final score: `31.29`
- final success rate: `95.8%`
- final collision rate: `4.2%`
- final episode length: `100.3`
- final explained variance: `0.971`
- final measured entropy: `0.826`
- final SPS: `59,029`
- best score: `33.65` at `15.96M` steps
- best score checkpoint nearest to peak: `0000000015986688.bin`
- best score point success/collision: `100.0%` success, `0.0%` collision
- other zero-collision high-score points appeared at `7.36M`, `20.51M`, and `23.75M`

Interpretation:

- Reset clearance produced the largest single quality jump so far.
- The previous best training-log peak was roughly `93%` success and `6%` collision; this run repeatedly reached near-`100%` success and zero collisions on the training log windows.
- Final quality also improved materially: about `95.8%` success and `4.2%` collision versus the prior lower-entropy final around `85.7%` success and `15.0%` collision.
- The result confirms that a meaningful part of the old collision rate was caused by bad starts/goals, not just policy weakness.
- This changes how scores should be interpreted: the old environment partially rewarded policies for surviving noisy invalid resets; the cleaned environment is a better measure of navigation behavior.

Next recommendation:

- Promote or fixed-evaluate the `0000000015986688.bin` checkpoint from this run before continuing.
- Run the same visual failure analysis on this checkpoint using reset clearance enabled.
- If remaining failures are mostly late wall/tree collisions, the next reward experiment should add a stronger clearance objective or evaluate a promotion metric that weights collision rate above raw return.

## Experiment 11 - Speed Control 25M From Scratch

Purpose:

- Test the new 15-action controller: `5` steering bins times `3` speed bins.
- Speed bins are brake, coast, and accelerate.
- Determine whether allowing braking/acceleration improves navigation quality enough to justify the larger action space.

Setup:

- initialization: fresh policy
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- actions: `15` discrete actions
- speed range: `0.0m/s` to `1.6m/s`
- acceleration: `1.5m/s^2`
- brake deceleration: `2.5m/s^2`
- coast deceleration: `0.3m/s^2`
- agents: `64`
- replay ratio: `2.0`
- entropy coefficient: `0.015`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- reset clearance: enabled
- curriculum: disabled
- total steps: `25M`

Run:

- run id: `1777298593426`
- checkpoint dir: `navigation/artifacts/speed_control_25m_checkpoints/field_nav/1777298593426`
- log dir: `navigation/artifacts/speed_control_25m_logs/field_nav`

Results:

- final steps: `24,993,792`
- final score: `18.29`
- final success rate: `70.6%`
- final collision rate: `29.4%`
- final explained variance: `0.789`
- final measured entropy: `2.001`
- final SPS: `62,540`
- best score: `20.76` at `21.91M` steps
- best score checkpoint nearest to peak: `0000000021884928.bin`
- best score point success/collision: `76.2%` success, `23.9%` collision

Interpretation:

- Speed control learned from scratch, but it is currently much worse than the fixed-speed reset-clearance policy.
- The larger action space keeps the policy substantially more stochastic; final entropy is still around `2.0` versus about `0.83` in the fixed-speed reset-clearance run.
- The value function is usable but weaker than the mature fixed-speed runs, with explained variance below `0.8`.
- The result does not prove braking is bad; it mostly shows the new controller needs either continuation or stronger action regularization.

Next experiment:

- Continue from the best speed-control checkpoint, `0000000021884928.bin`.
- Lower entropy coefficient from `0.015` to `0.005`.
- Keep the speed-control dynamics, reset clearance, LR floor, replay ratio, and polar encoder unchanged.
- Run another `25M` steps and select by best checkpoint, not final checkpoint.

## Experiment 12 - Speed Control 25M, Entropy 0.005 From Peak

Purpose:

- Continue the best speed-control checkpoint from Experiment 11.
- Lower entropy pressure to make the larger 15-action policy more decisive.
- Check whether the high collision rate from Experiment 11 was partly caused by excessive action stochasticity.

Setup:

- initialization: best score checkpoint from Experiment 11
- loaded checkpoint: `navigation/artifacts/speed_control_25m_checkpoints/field_nav/1777298593426/0000000021884928.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- actions: `15` discrete actions
- speed range: `0.0m/s` to `1.6m/s`
- agents: `64`
- replay ratio: `2.0`
- entropy coefficient: `0.005`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- reset clearance: enabled
- curriculum: disabled
- total steps: `25M`

Run:

- run id: `1777302165839`
- checkpoint dir: `navigation/artifacts/speed_control_ent005_25m_checkpoints/field_nav/1777302165839`
- log dir: `navigation/artifacts/speed_control_ent005_25m_logs/field_nav`

Results:

- final steps: `24,993,792`
- final score: `27.16`
- final success rate: `87.8%`
- final collision rate: `12.2%`
- final explained variance: `0.930`
- final measured entropy: `1.041`
- final SPS: `60,654`
- best score: `27.66` at `21.92M` steps
- best score checkpoint nearest to peak: `0000000021884928.bin`
- best score point success/collision: `88.2%` success, `11.8%` collision

Interpretation:

- Lowering entropy was a clear improvement over Experiment 11.
- The policy became much more decisive: measured entropy dropped from about `2.0` to about `1.0`.
- Value fit recovered to the same healthy range as the older fixed-speed policies.
- The speed-control policy is now useful, but still behind the fixed-speed reset-clearance policy, which reached `95.8%` final success and `4.2%` final collision.
- The remaining gap is likely not just policy stochasticity; the speed/action dynamics may be making the agent drive too aggressively because progress reward favors speed.

Next experiment:

- Continue from the best Experiment 12 checkpoint.
- Keep entropy coefficient at `0.005`.
- Reduce `max_speed_mps` from `1.6` to `1.2`.
- Keep reset clearance and the reward unchanged.
- This isolates whether the high collision rate is caused by the faster speed envelope rather than braking itself.

## Experiment 13 - Speed Control 25M, Entropy 0.005, Max Speed 1.2

Purpose:

- Test whether the speed-control policy was underperforming because the `1.6m/s` speed envelope encouraged overly aggressive driving.
- Keep the reward unchanged and isolate the dynamics change.

Setup:

- initialization: best score checkpoint from Experiment 12
- loaded checkpoint: `navigation/artifacts/speed_control_ent005_25m_checkpoints/field_nav/1777302165839/0000000021884928.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- actions: `15` discrete actions
- speed range: `0.0m/s` to `1.2m/s`
- agents: `64`
- replay ratio: `2.0`
- entropy coefficient: `0.005`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- reset clearance: enabled
- curriculum: disabled
- total steps: `25M`

Run:

- run id: `1777302613777`
- checkpoint dir: `navigation/artifacts/speed_control_ent005_max12_25m_checkpoints/field_nav/1777302613777`
- log dir: `navigation/artifacts/speed_control_ent005_max12_25m_logs/field_nav`

Results:

- final steps: `24,993,792`
- final score: `29.76`
- final success rate: `92.8%`
- final collision rate: `7.2%`
- final episode length: `80.2`
- final explained variance: `0.949`
- final measured entropy: `0.952`
- final SPS: `62,076`
- best score: `29.78` at `21.92M` steps
- best score checkpoint nearest to peak: `0000000021884928.bin`
- best score point success/collision: `92.9%` success, `7.1%` collision

Interpretation:

- Reducing max speed from `1.6m/s` to `1.2m/s` was a clear improvement.
- Collision dropped from `12.2%` final in Experiment 12 to `7.2%` final here.
- Success rose from `87.8%` to `92.8%`.
- Episode length increased from about `57.7` to `80.2`, which is expected from the slower speed limit.
- This nearly matches the best older fixed-speed continuation peaks, but still does not beat the reset-clearance fixed-speed run's best score of `33.65`.
- The result strongly suggests that speed-control needs a speed envelope or reward term that prevents progress reward from overvaluing fast driving.

Next experiment:

- Continue from the best Experiment 13 checkpoint.
- Reduce `max_speed_mps` again from `1.2` to `1.0`, matching the old fixed-speed top speed.
- Keep braking/coasting/acceleration available, so the policy can slow below the old baseline but cannot exceed it.
- This tests whether braking itself helps once extra top speed is removed.

## Experiment 14 - Speed Control 25M, Entropy 0.005, Max Speed 1.0

Purpose:

- Test whether braking improves policy quality when the robot cannot exceed the old fixed-speed top speed.
- Compare against Experiment 13 to see whether reducing speed further improves safety or only slows episodes down.

Setup:

- initialization: best score checkpoint from Experiment 13
- loaded checkpoint: `navigation/artifacts/speed_control_ent005_max12_25m_checkpoints/field_nav/1777302613777/0000000021884928.bin`
- observation: polar bins, `3 x 32 x 16 + 7`
- encoder: `FieldNavPolarEncoder`
- actions: `15` discrete actions
- speed range: `0.0m/s` to `1.0m/s`
- agents: `64`
- replay ratio: `2.0`
- entropy coefficient: `0.005`
- minimum LR ratio: `0.3`
- LR warmup: `500k` aggregate steps
- reset clearance: enabled
- curriculum: disabled
- total steps: `25M`

Run:

- run id: `1777303060208`
- checkpoint dir: `navigation/artifacts/speed_control_ent005_max10_25m_checkpoints/field_nav/1777303060208`
- log dir: `navigation/artifacts/speed_control_ent005_max10_25m_logs/field_nav`

Results:

- final steps: `24,993,792`
- final score: `29.45`
- final success rate: `92.3%`
- final collision rate: `7.7%`
- final episode length: `96.4`
- final explained variance: `0.942`
- final measured entropy: `0.750`
- final SPS: `46,926`
- best score: `30.19` at `21.83M` steps
- best score checkpoint nearest to peak: `0000000021884928.bin`
- best score point success/collision: `94.1%` success, `5.8%` collision

Interpretation:

- Lowering max speed from `1.2m/s` to `1.0m/s` improved peak safety but did not improve final score.
- The best checkpoint reached `94.1%` success and `5.8%` collision, which is safer than Experiment 13's best point but lower-score than the fixed-speed reset-clearance best.
- Final quality regressed slightly relative to Experiment 13: score `29.45` versus `29.76`, success `92.3%` versus `92.8%`, collision `7.7%` versus `7.2%`.
- Episode length increased to about `96`, so the time penalty is now meaningfully limiting score.
- Throughput also dipped late, likely because longer episodes reduce reset frequency and change eval/training timing.

Current speed-control takeaway:

- The new 15-action controller can learn useful braking/acceleration behavior.
- Entropy `0.005` is much better than `0.015` for this larger action space.
- The original `1.6m/s` max speed was too aggressive.
- `1.2m/s` is the best overall speed-control setting from these runs.
- `1.0m/s` can reduce peak collision, but it pays too much in episode length and final score.

Recommended next experiment:

- Promote the Experiment 13 best checkpoint for fixed-seed visual evaluation:
  `navigation/artifacts/speed_control_ent005_max12_25m_checkpoints/field_nav/1777302613777/0000000021884928.bin`
- Compare it against the fixed-speed reset-clearance best checkpoint on the same held-out seeds.
- If failures are mostly late obstacle contacts, add a reward term that penalizes high speed near obstacles instead of globally lowering max speed.

## Fixed-Seed Eval - Speed Control vs Fixed Speed

Purpose:

- Compare the best speed-control checkpoint against the best fixed-speed reset-clearance checkpoint on identical held-out seeds.
- Use deterministic Python env rollouts with reset clearance enabled.

Setup:

- episodes: `250`
- seeds: `300000..300249`
- max steps: `400`
- reset start clearance: `1.0m`
- reset goal clearance: `1.0m`
- reset forward clearance: `1.5m`
- reset forward margin: `0.25m`

Checkpoints:

- speed control: `navigation/artifacts/speed_control_ent005_max12_25m_checkpoints/field_nav/1777302613777/0000000021884928.bin`
- fixed speed: `navigation/artifacts/polar_reset_clearance_25m_checkpoints/field_nav/1777285806862/0000000015986688.bin`

Artifacts:

- speed control render/eval: `navigation/artifacts/eval_speed_control_max12_best/`
- fixed speed render/eval: `navigation/artifacts/eval_fixed_speed_reset_clearance_best/`

Results:

| policy | success | collision | timeout | mean return | mean steps | collision kinds |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| speed control max `1.2m/s` | `96.0%` | `3.6%` | `0.4%` | `32.27` | `66.1` | `7` tree, `1` wall, `1` person |
| fixed speed `1.0m/s` | `97.6%` | `1.6%` | `0.8%` | `32.75` | `100.0` | `1` tree, `3` wall |

Interpretation:

- The fixed-speed checkpoint still wins on aggregate reliability: higher success and lower collision.
- Speed control is much faster: `66` mean steps versus `100` for fixed speed.
- Mean return is close despite the extra collisions because the speed-control policy reaches goals sooner.
- Speed-control failures are mostly tree collisions, with one person collision; fixed-speed failures are mostly wall collisions plus one tree.
- This suggests braking/acceleration is useful for speed, but the current reward does not sufficiently discourage fast approaches near hard obstacles.

Next recommendation:

- Keep max speed at `1.2m/s`.
- Add a speed-aware clearance penalty instead of globally lowering speed:
  penalize `speed * max(0, clearance_threshold - nearest_margin)` for hard obstacles.
- Start from the Experiment 13 speed-control checkpoint and run `25M`.
- Promote by fixed-seed eval, prioritizing collision rate first and mean return second.
