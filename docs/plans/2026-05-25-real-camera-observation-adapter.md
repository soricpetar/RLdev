# Real-Camera Observation Adapter Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build the first deterministic real-camera/depth/segmentation to Stage49 observation adapter, with tests and debug tools, so the policy can run in offline replay and shadow mode.

**Architecture:** Implement a pure Python/Numpy adapter library under `navigation/real_adapter/`. The first version does not run neural depth or segmentation itself; it consumes already-aligned depth and class maps plus robot/goal state, then emits the exact 2055-float policy observation. Perception models are pluggable later. Tests first verify parity with the current Stage49 observation contract.

**Tech Stack:** Python, NumPy, pytest, existing `navigation.nav_env` code, optional Matplotlib/Pillow for debug visualization.

---

## Phase 1: Build the observation contract as code

### Task 1: Create adapter package skeleton

**Objective:** Add a dedicated package for real-camera adapter code.

**Files:**
- Create: `navigation/real_adapter/__init__.py`
- Create: `navigation/real_adapter/spec.py`
- Test: `tests/test_real_adapter_spec.py`

**Step 1: Write failing test**

Create `tests/test_real_adapter_spec.py`:

```python
import numpy as np

from navigation.real_adapter.spec import Stage49ObservationSpec


def test_stage49_spec_matches_policy_contract():
    spec = Stage49ObservationSpec()
    assert spec.channels == 4
    assert spec.angle_bins == 32
    assert spec.distance_bins == 16
    assert spec.costmap_size == 2048
    assert spec.vector_size == 7
    assert spec.flat_size == 2055
    assert spec.distance_bin_width_m == 1.5
    assert spec.visible_angle_mask().shape == (32,)
    assert spec.visible_angle_mask().dtype == np.bool_
```

**Step 2: Verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_real_adapter_spec.py -v
```

Expected: FAIL because `navigation.real_adapter.spec` does not exist.

**Step 3: Implement minimal code**

Create `navigation/real_adapter/__init__.py`:

```python
from .spec import Stage49ObservationSpec
```

Create `navigation/real_adapter/spec.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Stage49ObservationSpec:
    channels: int = 4
    angle_bins: int = 32
    distance_bins: int = 16
    max_distance_m: float = 24.0
    front_camera_fov_deg: float = 100.0
    inflation_radius_m: float = 1.0
    goal_norm_distance_m: float = 25.0
    speed_norm_mps: float = 3.0
    max_turn_rate_rps: float = 1.0

    @property
    def costmap_size(self) -> int:
        return self.channels * self.angle_bins * self.distance_bins

    @property
    def vector_size(self) -> int:
        return 7

    @property
    def flat_size(self) -> int:
        return self.costmap_size + self.vector_size

    @property
    def distance_bin_width_m(self) -> float:
        return self.max_distance_m / self.distance_bins

    def angle_centers_rad(self) -> np.ndarray:
        return -np.pi + (np.arange(self.angle_bins, dtype=np.float32) + 0.5) * (2.0 * np.pi / self.angle_bins)

    def visible_angle_mask(self) -> np.ndarray:
        half_fov = np.deg2rad(self.front_camera_fov_deg) * 0.5
        return np.abs(self.angle_centers_rad()) <= half_fov
```

**Step 4: Verify GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_real_adapter_spec.py -v
```

Expected: PASS.

---

## Phase 2: Implement costmap construction from robot-frame points

### Task 2: Add class mapping constants

**Objective:** Encode the simulator behavior channels and real-world class mapping explicitly.

**Files:**
- Modify: `navigation/real_adapter/spec.py`
- Test: `tests/test_real_adapter_spec.py`

**Step 1: Write failing test**

Add:

```python
def test_class_mapping_is_conservative():
    spec = Stage49ObservationSpec()
    assert spec.class_to_channel_and_cost("person") == (1, 1.0)
    assert spec.class_to_channel_and_cost("wall") == (0, 1.0)
    assert spec.class_to_channel_and_cost("tree") == (0, 1.0)
    assert spec.class_to_channel_and_cost("bush") == (2, 0.42)
    assert spec.class_to_channel_and_cost("pothole") == (2, 0.65)
    assert spec.class_to_channel_and_cost("unknown") == (0, 1.0)
```

**Step 2: Verify RED**

Run the spec test; expect missing method failure.

**Step 3: Implement**

Add to `Stage49ObservationSpec`:

```python
    def class_to_channel_and_cost(self, class_name: str) -> tuple[int, float]:
        key = str(class_name).lower()
        if key in {"person", "human", "pedestrian", "cyclist", "animal"}:
            return 1, 1.0
        if key in {"bush", "grass", "vegetation", "soft_vegetation"}:
            return 2, 0.42
        if key in {"pothole", "hole", "curb", "ditch", "stair", "drop"}:
            return 2, 0.65
        return 0, 1.0
```

**Step 4: Verify GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_real_adapter_spec.py -v
```

---

### Task 3: Build point-to-polar costmap adapter

**Objective:** Convert robot-frame obstacle points/classes into the 4x32x16 Stage49 costmap.

**Files:**
- Create: `navigation/real_adapter/costmap.py`
- Test: `tests/test_real_adapter_costmap.py`

**Step 1: Write failing tests**

Create `tests/test_real_adapter_costmap.py`:

```python
import numpy as np

from navigation.real_adapter.costmap import points_to_costmap
from navigation.real_adapter.spec import Stage49ObservationSpec


def test_costmap_has_visibility_mask_and_shape():
    spec = Stage49ObservationSpec()
    xyz = np.zeros((0, 3), dtype=np.float32)
    classes = np.asarray([], dtype=object)
    costmap = points_to_costmap(xyz, classes, spec)
    assert costmap.shape == (4, 32, 16)
    visible = spec.visible_angle_mask()
    assert np.all(costmap[3, visible, :] == 0.0)
    assert np.all(costmap[3, ~visible, :] == 1.0)


def test_forward_static_obstacle_marks_static_channel():
    spec = Stage49ObservationSpec()
    xyz = np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32)
    classes = np.asarray(["tree"], dtype=object)
    costmap = points_to_costmap(xyz, classes, spec)
    visible = spec.visible_angle_mask()
    assert costmap[0, visible, :].max() == 1.0
    assert costmap[1].max() == 0.0


def test_person_marks_moving_channel():
    spec = Stage49ObservationSpec()
    xyz = np.asarray([[4.0, 0.0, 0.0]], dtype=np.float32)
    classes = np.asarray(["person"], dtype=object)
    costmap = points_to_costmap(xyz, classes, spec)
    assert costmap[1].max() == 1.0
    assert costmap[0].max() == 0.0
```

**Step 2: Verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_real_adapter_costmap.py -v
```

Expected: FAIL because module does not exist.

**Step 3: Implement**

Create `navigation/real_adapter/costmap.py`:

```python
from __future__ import annotations

import numpy as np

from .spec import Stage49ObservationSpec


def _angle_to_bin(angle_rad: np.ndarray, spec: Stage49ObservationSpec) -> np.ndarray:
    return np.floor((angle_rad + np.pi) / (2.0 * np.pi) * spec.angle_bins).astype(np.int64)


def _distance_to_bin(distance_m: np.ndarray, spec: Stage49ObservationSpec) -> np.ndarray:
    return np.floor(distance_m / spec.max_distance_m * spec.distance_bins).astype(np.int64)


def points_to_costmap(
    points_robot_xyz: np.ndarray,
    class_names: np.ndarray,
    spec: Stage49ObservationSpec | None = None,
) -> np.ndarray:
    spec = spec or Stage49ObservationSpec()
    costmap = np.zeros((spec.channels, spec.angle_bins, spec.distance_bins), dtype=np.float32)

    visible = spec.visible_angle_mask()
    costmap[3, ~visible, :] = 1.0

    points = np.asarray(points_robot_xyz, dtype=np.float32)
    if points.size == 0:
        return costmap
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_robot_xyz must have shape (N, 3)")
    if len(class_names) != len(points):
        raise ValueError("class_names length must match points")

    x = points[:, 0]
    y = points[:, 1]
    dist = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)

    valid = (x > 0.0) & (dist > 0.0) & (dist <= spec.max_distance_m)
    if not np.any(valid):
        return costmap

    a_bins = _angle_to_bin(angle[valid], spec)
    d_bins = _distance_to_bin(dist[valid], spec)
    valid_indices = np.nonzero(valid)[0]

    inflate_bins = max(0, int(np.ceil(spec.inflation_radius_m / spec.distance_bin_width_m)))

    for src_idx, ab, db in zip(valid_indices, a_bins, d_bins, strict=False):
        if ab < 0 or ab >= spec.angle_bins or db < 0 or db >= spec.distance_bins:
            continue
        if not visible[ab]:
            continue
        channel, cost = spec.class_to_channel_and_cost(class_names[src_idx])
        lo = max(0, db - inflate_bins)
        hi = min(spec.distance_bins - 1, db + inflate_bins)
        costmap[channel, ab, lo : hi + 1] = np.maximum(costmap[channel, ab, lo : hi + 1], cost)

    return costmap
```

**Step 4: Verify GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_real_adapter_costmap.py -v
```

---

## Phase 3: Implement camera projection and observation assembly

### Task 4: Add depth/class image to robot-frame points projection

**Objective:** Convert aligned depth and class maps into robot-frame points with class names.

**Files:**
- Create: `navigation/real_adapter/projection.py`
- Test: `tests/test_real_adapter_projection.py`

**Step 1: Write failing test**

Create a simple pinhole-camera test where the center pixel projects forward.

```python
import numpy as np

from navigation.real_adapter.projection import depth_classes_to_robot_points


def test_center_pixel_projects_forward_with_identity_extrinsic():
    depth = np.asarray([[2.0]], dtype=np.float32)
    class_map = np.asarray([["tree"]], dtype=object)
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    camera_to_robot = np.eye(4, dtype=np.float32)
    points, classes = depth_classes_to_robot_points(depth, class_map, intrinsics, camera_to_robot)
    assert points.shape == (1, 3)
    assert classes.tolist() == ["tree"]
    np.testing.assert_allclose(points[0], [2.0, 0.0, 0.0], atol=1e-6)
```

**Step 2: Verify RED**

Run test; expect missing module.

**Step 3: Implement**

Important convention: internal robot frame is x-forward, y-left, z-up. The projection function should accept a `camera_to_robot` 4x4 matrix so we can calibrate real camera frames explicitly. For the first test, identity assumes input camera points are already x-forward/y-left/z-up.

**Step 4: Verify GREEN**

Run projection tests.

---

### Task 5: Assemble full 2055-vector observation

**Objective:** Build `make_stage49_observation()` that combines costmap, goal, and state.

**Files:**
- Create: `navigation/real_adapter/observation.py`
- Test: `tests/test_real_adapter_observation.py`

**Step 1: Write failing tests**

Tests should verify:

- output shape `(2055,)`
- dtype float32
- costmap occupies first 2048 values
- goal vector uses `[distance/25, sin(error), cos(error)]`
- state vector uses `[speed/3, yaw_rate/1.0, previous_steering, 1.0]`

**Step 2: Verify RED**

Run observation tests; expect failure.

**Step 3: Implement**

Function signature:

```python
def make_stage49_observation(
    costmap: np.ndarray,
    goal_distance_m: float,
    goal_heading_error_rad: float,
    speed_mps: float,
    yaw_rate_rps: float,
    previous_steering: float,
    spec: Stage49ObservationSpec | None = None,
) -> np.ndarray:
    ...
```

**Step 4: Verify GREEN**

Run all real-adapter tests.

---

## Phase 4: Add synthetic parity against the simulator

### Task 6: Compare adapter output to `front_camera_polar_costmap()`

**Objective:** Prove the adapter’s angle bins, distance bins, visible mask, and flattening match the simulator.

**Files:**
- Create: `tests/test_real_adapter_sim_parity.py`

**Step 1: Write parity test**

Use a few synthetic points at known locations and compare with expected visible bins. Do not require exact parity with simulator object-radius painting yet; use this to test core binning and visibility.

**Step 2: Run RED/GREEN as needed**

Fix adapter binning if mismatches are found.

---

## Phase 5: Add debug/replay tooling

### Task 7: Add debug visualization script

**Objective:** Produce a PNG panel for one adapter input/output: RGB, depth, class map, costmap channel summaries, and policy action/logits placeholder.

**Files:**
- Create: `navigation/scripts/debug_real_adapter_frame.py`

CLI shape:

```bash
./.venv/bin/python navigation/scripts/debug_real_adapter_frame.py \
  --rgb frame.png \
  --depth depth.npy \
  --classes classes.npy \
  --output debug.png
```

Initial version can omit policy inference and only show adapter outputs.

---

### Task 8: Add offline sample data format

**Objective:** Define and support a simple `.npz` frame bundle for real/synthetic adapter testing.

Required `.npz` keys:

- `rgb`: HxWx3 uint8, optional for adapter but useful for debug
- `depth_m`: HxW float32
- `class_map`: HxW object/string or integer labels with label map
- `camera_to_robot`: 4x4 float32
- `fx`, `fy`, `cx`, `cy`
- `goal_distance_m`
- `goal_heading_error_rad`
- `speed_mps`
- `yaw_rate_rps`
- `previous_steering`

Create a loader that returns a 2055 observation plus debug metadata.

---

## Phase 6: Policy shadow-mode wrapper

### Task 9: Add policy inference wrapper for adapter observations

**Objective:** Load a Stage49/Stage58 checkpoint and run action inference from a 2055-vector observation.

**Files:**
- Create: `navigation/real_adapter/policy.py`
- Create: `navigation/scripts/run_policy_on_adapter_frame.py`

Acceptance:

- Loads Stage49 checkpoint.
- Accepts `.npz` frame bundle.
- Prints action id, steering, throttle, logits/probs.
- Does not command any robot hardware.

---

## Execution Order Recommendation

Implement now in this order:

1. `spec.py` + tests
2. class mapping tests
3. `costmap.py` + tests
4. `observation.py` + tests
5. synthetic parity test
6. projection from depth/class image
7. debug visualization script
8. `.npz` frame loader
9. checkpoint inference wrapper

Do not start with neural depth or segmentation models. For the first adapter, use already-produced depth/class maps. That keeps the boundary clean and lets us validate the hard part: matching the RL policy’s observation contract.

## First Real-World Prototype Stack

For shadow mode, use:

- depth source: RGB-D/depth camera if available; otherwise monocular depth saved as `depth_m`
- segmentation source: pretrained model output reduced to our class names
- adapter: this package
- policy: Stage49 and Stage58/59 run side-by-side
- output: log action suggestions only

## Verification Commands

Run adapter tests:

```bash
cd /Users/petar/dev/RLdev
./.venv/bin/python -m pytest tests/test_real_adapter_*.py -v
```

Run full existing tests only after adapter tests pass:

```bash
cd /Users/petar/dev/RLdev
./.venv/bin/python -m pytest tests -q
```
