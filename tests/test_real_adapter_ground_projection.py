import numpy as np

from navigation.real_adapter.ground_projection import GroundPlaneCalibration, apply_homography


def test_default_calibration_maps_correspondences_exactly():
    calibration = GroundPlaneCalibration.default_for_image(
        width=101,
        height=101,
        near_distance_m=1.5,
        far_distance_m=12.0,
        near_half_width_m=1.0,
        far_half_width_m=5.0,
        horizon_fraction=0.4,
        bottom_fraction=1.0,
    )

    projected = calibration.project_pixels(calibration.image_points_px)
    np.testing.assert_allclose(projected, calibration.robot_points_m, atol=1e-5)


def test_default_calibration_centerline_maps_to_forward_axis():
    calibration = GroundPlaneCalibration.default_for_image(
        width=101,
        height=101,
        near_distance_m=1.5,
        far_distance_m=12.0,
        near_half_width_m=1.0,
        far_half_width_m=5.0,
        horizon_fraction=0.4,
        bottom_fraction=1.0,
    )

    projected = calibration.project_pixels(np.asarray([[50.0, 100.0], [50.0, 40.0]], dtype=np.float32))

    np.testing.assert_allclose(projected[0], [1.5, 0.0], atol=1e-5)
    np.testing.assert_allclose(projected[1], [12.0, 0.0], atol=1e-5)


def test_apply_homography_handles_empty_input():
    out = apply_homography(np.zeros((0, 2), dtype=np.float32), np.eye(3, dtype=np.float32))
    assert out.shape == (0, 2)
    assert out.dtype == np.float32
