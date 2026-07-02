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


def test_invalid_and_too_far_depth_pixels_are_filtered():
    depth = np.asarray([[0.0, np.nan, 25.0, 3.0]], dtype=np.float32)
    class_map = np.asarray([["tree", "tree", "tree", "person"]], dtype=object)
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    points, classes = depth_classes_to_robot_points(depth, class_map, intrinsics, np.eye(4, dtype=np.float32), max_distance_m=24.0)
    assert points.shape == (1, 3)
    assert classes.tolist() == ["person"]
