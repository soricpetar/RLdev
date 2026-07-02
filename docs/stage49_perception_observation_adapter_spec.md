# Stage49 Real-Camera Perception → Observation Adapter Product Spec

## Goal
Build a real-time adapter that converts real robot camera perception into the exact observation vector expected by the current Stage49/Stage58/Stage59 `field_nav_camera_native_hard_speed` policies, so the frozen RL policy can be run in shadow mode and later used as a steering/navigation prior under a safety controller.

This spec is grounded in the current codebase:
- Env observation construction: `navigation/nav_env/env.py`
- Costmap/channel generation: `navigation/nav_env/maps.py`
- Policy encoder input contract: `pufferlib/models.py::FieldNavPolarEncoder`
- Config: `config/field_nav_camera_native_hard_speed.ini`
- Stage49 checkpoint: `navigation/artifacts/front_camera_native_stage49_stage45final_accelpenalty025_safety20_ent012_lr5e5_60m_checkpoints/field_nav_camera_native_hard_speed/1779384641942/0000000051658752.bin`

## Current Policy Observation Contract

Stage49 is a 15-action speed-control policy with a flattened float32 observation vector of length `2055`:

- `costmap`: 4 x 32 x 16 = 2048 values
- `goal`: 3 values
- `state`: 4 values

Flatten order is exactly:

1. `costmap.reshape(-1)`
2. `goal.reshape(-1)`
3. `state.reshape(-1)`

This is determined by `FieldNavEnv.observation_space` keys: `['costmap', 'goal', 'state']`, and `pufferlib/emulation.py::_flatten_obs()`.

The Stage49 checkpoint confirms:

- map encoder input: 2048
- vector encoder input: 7
- action logits: 15

Policy input tensor shape at inference should be `(1, 2055)` float32.

## Costmap Semantics

The costmap shape is:

`(channels=4, angle_bins=32, distance_bins=16)`

The first three channels are behavior-level obstacle channels from `BEHAVIOR_CHANNELS`:

0. `static_obstacle`
   - simulator kinds: `tree`, `wall`, and unknown/default hard obstacles
   - cost: tree/wall = `1.0`

1. `moving_obstacle`
   - simulator kind: `person`
   - cost: `1.0`

2. `soft_hazard`
   - simulator kinds: `bush`, `pothole`
   - bush cost: `0.42`
   - pothole cost: `0.65`

3. `not_visible`
   - front-camera mask channel
   - value `1.0` outside the camera FOV, `0.0` inside the FOV

Real-world class mapping should be conservative:

- person / cyclist / animal / moving human-sized object → channel 1, cost 1.0
- wall / fence / post / tree trunk / furniture / vehicle / unknown solid obstacle → channel 0, cost 1.0
- bush / grass clump / soft vegetation → channel 2, cost 0.42 unless collision risk is unknown, then channel 0
- hole / curb drop / stair / ditch / pothole → channel 2, cost 0.65, or channel 0 if the robot cannot traverse it
- segmentation uncertainty + close depth blob → channel 0, cost 1.0

For first deployment, false obstacles are acceptable; missed near obstacles are not.

## Polar Geometry Contract

Current hard-speed config uses:

- angle bins: `32`
- distance bins: `16`
- max distance: `24.0 m`
- front camera FOV: `100°`
- robot radius: `0.35 m`
- inflation radius: `1.0 m`
- max steps: `400`
- dt: `0.2 s`
- speed range: `0.0–1.2 m/s`

The simulator builds a full 360° polar map, then masks observations outside the front-camera FOV. Angle bin centers are:

`angle = -pi + (bin + 0.5) * 2pi / 32`

Only bins whose center angle satisfies:

`abs(angle) <= 50°`

are visible. All obstacle channels outside that visible sector are zeroed, and channel 3 is set to 1.0 outside that sector.

Implication: even though the robot has a 100° forward camera, the policy input remains a 360° polar tensor with a visibility mask. With 32 bins, bin width is 11.25°. A 100° FOV makes approximately 8 central bins visible, depending on bin-center thresholding.

Distance bins are linear from 0 to 24 m. Bin width is 1.5 m. The simulator marks objects across a radial span based on object radius/inflation. For real perception, the adapter should approximate this by painting obstacle cost into bins covered by the measured obstacle extent plus a 1.0 m safety inflation.

## Real Adapter Pipeline

Inputs per frame:

- RGB image
- metric depth map, preferably from RGB-D/stereo; monocular depth acceptable for prototype only
- semantic segmentation/detection output
- camera intrinsics
- camera-to-robot extrinsics
- current robot speed, yaw rate, previous steering command
- goal bearing and distance from navigation layer, beacon, clicked target, GPS/SLAM, or operator-provided waypoint

Processing:

1. Align RGB, depth, and segmentation in the same image coordinate frame.
2. For pixels/segments with valid depth, back-project to camera 3D.
3. Transform points into robot frame: x forward, y left/right, heading-relative.
4. Compute polar angle `atan2(y, x)` and range `sqrt(x^2 + y^2)`.
5. Ignore points outside 24 m or behind the robot for obstacle marking; keep the 360° output shape but mark non-front bins as not visible.
6. Map segmentation class to behavior channel.
7. Paint the channel/bin with the appropriate cost, inflated by approximately 1.0 m around the object footprint.
8. Set channel 3 to 1.0 outside the 100° visible FOV and 0.0 inside.
9. Append goal and state features.
10. Flatten to length 2055 float32.

For obstacle painting, first implementation can be conservative and pixel/point based:

- For each class and angle bin, keep nearest occupied distance.
- Paint that distance bin plus neighboring radial bins corresponding to 1.0 m inflation.
- Optionally fill angular neighbors if the segment spans multiple pixels or appears wide.

Later implementation can fit object blobs/segments and paint using approximate object radius, closer to `maps.py::polar_costmap()`.

## Goal and State Features

`goal` is a 3-vector from `env.py::_goal_features()`:

1. `dist_norm = clip(goal_distance_m / 25.0, 0.0, 1.0)`
2. `sin(heading_error)`
3. `cos(heading_error)`

`heading_error = wrap_to_pi(goal_bearing_world_or_local - robot_heading)`.

In real deployment, the adapter must receive or estimate goal distance and heading error. If only bearing is known, use a conservative distance proxy; but this is a product risk because Stage49 was trained with true distance normalized to 25 m.

`state` is a 4-vector from `env.py::_state_features()`:

1. `clip(robot_speed_mps / 3.0, -1.0, 1.0)`
2. `clip(yaw_rate_rps / max_turn_rate_rps, -1.0, 1.0)`, with `max_turn_rate_rps = 1.0`
3. `clip(previous_steering_command, -1.0, 1.0)`
4. constant `1.0`

If the robot lacks yaw-rate sensing, estimate from wheel odometry/IMU; do not leave it arbitrary. If previous steering is unavailable on first frame, initialize to 0.0.

## Policy Output Contract

Action space has 15 discrete actions. The env decodes action by `divmod(action, 3)`:

- steering values: `[-1.0, -0.5, 0.0, 0.5, 1.0]`
- throttle values: `[-1.0, 0.0, 1.0]`

Mapping:

- 0=(-1.0, brake), 1=(-1.0, coast), 2=(-1.0, accel)
- 3=(-0.5, brake), 4=(-0.5, coast), 5=(-0.5, accel)
- 6=(0.0, brake), 7=(0.0, coast), 8=(0.0, accel)
- 9=(0.5, brake), 10=(0.5, coast), 11=(0.5, accel)
- 12=(1.0, brake), 13=(1.0, coast), 14=(1.0, accel)

Stage49 is known to prefer accelerate almost always. Initial real-world use should expose steering as a recommendation and route throttle through a separate safety/speed controller.

## Product Milestones and Acceptance Criteria

M1: Offline adapter library
- Function: `rgb/depth/segmentation + robot/goal state -> np.float32[2055]`
- Includes debug visualization of RGB, depth, segmentation, polar bins, and policy action.
- Unit test verifies vector length/order and value ranges.

M2: Synthetic parity test
- Generate sim scenes with known `FieldNavEnv._observation()`.
- Reconstruct an observation from synthetic object/depth/segmentation data.
- Acceptance: visible FOV mask exactly matches; channel/bin obstacle IoU is high enough for visible obstacles; goal/state vectors match numerically.

M3: Real data logging
- Record synchronized RGB, depth, segmentation, adapter output, robot state, goal state, policy logits/action, and manual command.
- Acceptance: replay script can reproduce policy actions deterministically from logs.

M4: Shadow mode
- Run on robot without control authority.
- Acceptance: no adapter crashes; latency measured; unsafe action rate reported; action distribution compared for Stage49 vs Stage58/59.

M5: Safety-gated steering test
- Policy controls steering only.
- Speed controller uses depth/free-space independent of policy throttle.
- Acceptance: low-speed, geofenced test with emergency stop; no raw Stage49 acceleration authority.

## Open Risks

- Real perception has occlusion, depth holes, lighting, blur, and camera pitch not represented in sim.
- 32 angle bins over 360° gives coarse front FOV resolution; only about 8 bins represent the actual 100° camera sector.
- Simulator object painting uses ideal object geometry and inflation, not raw pixels; the real adapter must approximate object extent conservatively.
- Goal distance/bearing source is external to camera perception and must be specified for IRL use.
- Stage49 throttle collapse makes direct velocity control unsafe; use a safety controller until Stage58/59-style brake/coast policies are validated.
