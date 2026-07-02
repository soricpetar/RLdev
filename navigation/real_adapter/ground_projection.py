from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundPlaneCalibration:
    """Pixel-to-ground calibration for a fixed forward camera.

    Robot-frame convention matches the real adapter: x is forward, y is left.
    The calibration stores four or more image-to-ground correspondences and
    solves a homography from image pixels to ground-plane robot meters.
    """

    image_points_px: np.ndarray
    robot_points_m: np.ndarray

    @classmethod
    def default_for_image(
        cls,
        width: int,
        height: int,
        *,
        near_distance_m: float = 1.0,
        far_distance_m: float = 10.0,
        near_half_width_m: float = 0.8,
        far_half_width_m: float = 4.0,
        horizon_fraction: float = 0.45,
        bottom_fraction: float = 0.98,
    ) -> "GroundPlaneCalibration":
        """Return a conservative starter calibration for a low forward camera.

        This is only a bootstrap. Replace it with measured correspondences once
        camera height/pitch and row spacing are known.
        """

        if width <= 1 or height <= 1:
            raise ValueError("width and height must be greater than 1")
        top_v = float(np.clip(horizon_fraction, 0.0, 1.0) * (height - 1))
        bottom_v = float(np.clip(bottom_fraction, 0.0, 1.0) * (height - 1))
        image_points = np.asarray(
            [
                [0.0, bottom_v],
                [float(width - 1), bottom_v],
                [0.0, top_v],
                [float(width - 1), top_v],
            ],
            dtype=np.float32,
        )
        robot_points = np.asarray(
            [
                [near_distance_m, near_half_width_m],
                [near_distance_m, -near_half_width_m],
                [far_distance_m, far_half_width_m],
                [far_distance_m, -far_half_width_m],
            ],
            dtype=np.float32,
        )
        return cls(image_points_px=image_points, robot_points_m=robot_points)

    def homography(self) -> np.ndarray:
        return homography_from_points(self.image_points_px, self.robot_points_m)

    def project_pixels(self, pixels_uv: np.ndarray) -> np.ndarray:
        return apply_homography(pixels_uv, self.homography())


def homography_from_points(image_points_px: np.ndarray, robot_points_m: np.ndarray) -> np.ndarray:
    src = np.asarray(image_points_px, dtype=np.float64)
    dst = np.asarray(robot_points_m, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("image_points_px must have shape (N, 2)")
    if dst.shape != src.shape:
        raise ValueError("robot_points_m must have the same shape as image_points_px")
    if len(src) < 4:
        raise ValueError("at least four correspondences are required")

    rows = []
    rhs = []
    for (u, v), (x, y) in zip(src, dst):
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v])
        rhs.append(x)
        rows.append([0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v])
        rhs.append(y)

    coeffs, *_ = np.linalg.lstsq(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64), rcond=None)
    return np.asarray(
        [
            [coeffs[0], coeffs[1], coeffs[2]],
            [coeffs[3], coeffs[4], coeffs[5]],
            [coeffs[6], coeffs[7], 1.0],
        ],
        dtype=np.float32,
    )


def apply_homography(pixels_uv: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float32)
    h = np.asarray(homography, dtype=np.float32)
    if pixels.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_uv must have shape (N, 2)")
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")

    ones = np.ones((pixels.shape[0], 1), dtype=np.float32)
    projected = np.concatenate([pixels, ones], axis=1) @ h.T
    denom = projected[:, 2:3]
    valid = np.abs(denom) > 1e-8
    out = np.full((pixels.shape[0], 2), np.nan, dtype=np.float32)
    out[valid[:, 0]] = projected[valid[:, 0], :2] / denom[valid[:, 0]]
    return out
