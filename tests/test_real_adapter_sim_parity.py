import numpy as np

from navigation.nav_env.maps import CircularObstacle, front_camera_polar_costmap
from navigation.real_adapter.sim_parity import objects_to_adapter_points
from navigation.real_adapter.costmap import points_to_costmap
from navigation.real_adapter.spec import Stage49ObservationSpec


def test_visible_mask_matches_front_camera_polar_costmap():
    spec = Stage49ObservationSpec()
    sim_map = front_camera_polar_costmap(
        (0.0, 0.0, 0.0),
        [],
        angle_bins=spec.angle_bins,
        distance_bins=spec.distance_bins,
        max_distance_m=spec.max_distance_m,
        inflation_radius_m=spec.inflation_radius_m,
        fov_deg=spec.front_camera_fov_deg,
    )
    adapter_map = points_to_costmap(np.zeros((0, 3), dtype=np.float32), np.asarray([], dtype=object), spec)
    np.testing.assert_array_equal(adapter_map[3], sim_map[3])
    np.testing.assert_allclose(adapter_map[:3], sim_map[:3])


def test_adapter_matches_sim_for_point_like_forward_objects():
    spec = Stage49ObservationSpec(inflation_radius_m=0.0)
    objects = [
        CircularObstacle(3.0, 0.0, 0.0, kind="tree"),
        CircularObstacle(5.0, 1.0, 0.0, kind="person"),
        CircularObstacle(7.0, -1.0, 0.0, kind="pothole"),
    ]
    points, classes = objects_to_adapter_points((0.0, 0.0, 0.0), objects)
    adapter_map = points_to_costmap(points, classes, spec)
    sim_map = front_camera_polar_costmap(
        (0.0, 0.0, 0.0),
        objects,
        angle_bins=spec.angle_bins,
        distance_bins=spec.distance_bins,
        max_distance_m=spec.max_distance_m,
        inflation_radius_m=spec.inflation_radius_m,
        fov_deg=spec.front_camera_fov_deg,
    )
    np.testing.assert_allclose(adapter_map, sim_map)


def test_object_points_transform_into_robot_frame_with_heading():
    obj = CircularObstacle(0.0, 3.0, 0.0, kind="tree")
    points, classes = objects_to_adapter_points((0.0, 0.0, np.pi / 2), [obj])
    np.testing.assert_allclose(points, [[3.0, 0.0, 0.0]], atol=1e-6)
    assert classes.tolist() == ["tree"]
