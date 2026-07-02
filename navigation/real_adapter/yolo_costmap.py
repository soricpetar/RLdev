from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .costmap import points_to_costmap
from .ground_projection import GroundPlaneCalibration
from .spec import Stage49ObservationSpec


TREE_ROW_CLASS = "tree_row"
FREE_CORRIDOR_CLASS = "free_corridor"


@dataclass(frozen=True)
class YoloMaskCostmapConfig:
    mask_sample_stride_px: int = 8
    row_column_stride_px: int = 6
    min_confidence: float = 0.25
    unknown_inside_fov_by_default: bool = True


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    src = np.asarray(mask)
    if src.ndim != 2:
        raise ValueError("mask must be 2D")
    out_h, out_w = shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError("shape must be positive")
    if src.shape == (out_h, out_w):
        return src.astype(bool, copy=False)

    y_idx = np.clip((np.arange(out_h) * src.shape[0] / out_h).astype(np.int64), 0, src.shape[0] - 1)
    x_idx = np.clip((np.arange(out_w) * src.shape[1] / out_w).astype(np.int64), 0, src.shape[1] - 1)
    return src[np.ix_(y_idx, x_idx)].astype(bool, copy=False)


def mask_sample_pixels(mask: np.ndarray, stride_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    stride = max(1, int(stride_px))
    sampled = binary[::stride, ::stride]
    rows, cols = np.nonzero(sampled)
    if len(rows) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([cols * stride, rows * stride], axis=1).astype(np.float32)


def mask_bottom_edge_pixels(mask: np.ndarray, column_stride_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    stride = max(1, int(column_stride_px))
    cols = np.arange(0, binary.shape[1], stride, dtype=np.int64)
    points: list[tuple[int, int]] = []
    for col in cols:
        ys = np.flatnonzero(binary[:, col])
        if ys.size:
            points.append((int(col), int(ys[-1])))
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _valid_robot_points(points_xy: np.ndarray, spec: Stage49ObservationSpec) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    x = points[:, 0]
    y = points[:, 1]
    dist = np.sqrt(x * x + y * y)
    valid = finite & (x > 0.0) & (dist > 0.0) & (dist <= spec.max_distance_m)
    return points[valid]


def _clear_known_free_bins(costmap: np.ndarray, points_xy: np.ndarray, spec: Stage49ObservationSpec) -> None:
    points = _valid_robot_points(points_xy, spec)
    if points.size == 0:
        return
    x = points[:, 0]
    y = points[:, 1]
    dist = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)
    half_fov = np.deg2rad(spec.front_camera_fov_deg) * 0.5
    visible = np.abs(angle) <= half_fov
    if not np.any(visible):
        return

    angle_bins = ((angle[visible] + np.pi) / (2.0 * np.pi) * spec.angle_bins).astype(np.int64)
    distance_bins = (dist[visible] / spec.max_distance_m * spec.distance_bins).astype(np.int64)
    angle_bins = np.clip(angle_bins, 0, spec.angle_bins - 1)
    distance_bins = np.clip(distance_bins, 0, spec.distance_bins - 1)
    costmap[3, angle_bins, distance_bins] = 0.0


def masks_to_costmap(
    class_masks: Mapping[str, np.ndarray],
    calibration: GroundPlaneCalibration,
    spec: Stage49ObservationSpec | None = None,
    config: YoloMaskCostmapConfig | None = None,
) -> np.ndarray:
    """Convert YOLO segmentation masks into the Stage49 polar costmap.

    `free_corridor` clears unknown bins. `tree_row` contributes static obstacle
    cost from the lower mask edge, which is the closest monocular proxy for the
    row contact line on the ground.
    """

    spec = spec or Stage49ObservationSpec()
    cfg = config or YoloMaskCostmapConfig()
    costmap = np.zeros((spec.channels, spec.angle_bins, spec.distance_bins), dtype=np.float32)
    visible = spec.visible_angle_mask()
    costmap[3, ~visible, :] = 1.0
    if cfg.unknown_inside_fov_by_default:
        costmap[3, visible, :] = 1.0

    corridor = class_masks.get(FREE_CORRIDOR_CLASS)
    if corridor is not None:
        corridor_pixels = mask_sample_pixels(corridor, cfg.mask_sample_stride_px)
        corridor_xy = calibration.project_pixels(corridor_pixels)
        _clear_known_free_bins(costmap, corridor_xy, spec)

    row = class_masks.get(TREE_ROW_CLASS)
    if row is not None:
        row_pixels = mask_bottom_edge_pixels(row, cfg.row_column_stride_px)
        row_xy = calibration.project_pixels(row_pixels)
        row_xy = _valid_robot_points(row_xy, spec)
        if row_xy.size:
            row_points = np.concatenate([row_xy, np.zeros((row_xy.shape[0], 1), dtype=np.float32)], axis=1)
            row_classes = np.full(row_xy.shape[0], "tree", dtype=object)
            row_costmap = points_to_costmap(row_points, row_classes, spec)
            costmap[:3] = np.maximum(costmap[:3], row_costmap[:3])

    return np.clip(costmap, 0.0, 1.0).astype(np.float32, copy=False)


def class_map_from_masks(class_masks: Mapping[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, "unknown", dtype=object)
    corridor = class_masks.get(FREE_CORRIDOR_CLASS)
    if corridor is not None:
        out[resize_mask_nearest(corridor, shape)] = FREE_CORRIDOR_CLASS
    row = class_masks.get(TREE_ROW_CLASS)
    if row is not None:
        out[resize_mask_nearest(row, shape)] = TREE_ROW_CLASS
    return out


class YoloSegmentationCostmap:
    """Lazy Ultralytics wrapper that produces Stage49 costmaps from RGB frames."""

    def __init__(
        self,
        weights: str | Path,
        *,
        spec: Stage49ObservationSpec | None = None,
        calibration: GroundPlaneCalibration | None = None,
        device: str = "cpu",
        imgsz: int = 640,
        config: YoloMaskCostmapConfig | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.spec = spec or Stage49ObservationSpec()
        self.calibration = calibration
        self.device = device
        self.imgsz = imgsz
        self.config = config or YoloMaskCostmapConfig()

    def predict_masks(self, source: str | Path | np.ndarray) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
        results = self.model.predict(
            source=source,
            imgsz=self.imgsz,
            conf=self.config.min_confidence,
            device=self.device,
            verbose=False,
        )
        if not results:
            raise RuntimeError("YOLO returned no results")
        result = results[0]
        orig_shape = tuple(int(v) for v in result.orig_shape)
        masks: dict[str, np.ndarray] = {}
        if result.masks is None or result.boxes is None:
            return masks, orig_shape

        names: Mapping[int, str] | dict[int, str] = getattr(result, "names", None) or self.model.names
        mask_data = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        confidences = result.boxes.conf.detach().cpu().numpy()
        for mask, class_id, confidence in zip(mask_data, classes, confidences):
            if float(confidence) < self.config.min_confidence:
                continue
            class_name = str(names[int(class_id)])
            if class_name not in {FREE_CORRIDOR_CLASS, TREE_ROW_CLASS}:
                continue
            resized = resize_mask_nearest(mask > 0.5, orig_shape)
            if class_name in masks:
                masks[class_name] = masks[class_name] | resized
            else:
                masks[class_name] = resized
        return masks, orig_shape

    def predict_costmap(self, source: str | Path | np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        masks, image_shape = self.predict_masks(source)
        calibration = self.calibration or GroundPlaneCalibration.default_for_image(
            width=image_shape[1],
            height=image_shape[0],
        )
        costmap = masks_to_costmap(masks, calibration, self.spec, self.config)
        return costmap, masks
