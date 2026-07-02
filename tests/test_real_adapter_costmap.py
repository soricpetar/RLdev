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


def test_points_outside_front_fov_are_not_visible_not_obstacles():
    spec = Stage49ObservationSpec()
    xyz = np.asarray([[0.0, 3.0, 0.0]], dtype=np.float32)
    classes = np.asarray(["tree"], dtype=object)
    costmap = points_to_costmap(xyz, classes, spec)
    assert costmap[:3].max() == 0.0
