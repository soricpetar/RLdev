#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigation.real_adapter.debug_viz import save_adapter_debug_panel
from navigation.real_adapter.ground_projection import GroundPlaneCalibration
from navigation.real_adapter.yolo_costmap import (
    YoloMaskCostmapConfig,
    YoloSegmentationCostmap,
    class_map_from_masks,
)


def _default_weights() -> Path:
    return DEV_ROOT / "field-row-yolo-share" / "models" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one RGB frame into a YOLO-derived Stage49 costmap.")
    parser.add_argument("--image", required=True, type=Path, help="Input RGB image")
    parser.add_argument("--weights", type=Path, default=_default_weights())
    parser.add_argument("--output", required=True, type=Path, help="Output debug panel PNG")
    parser.add_argument("--costmap-output", type=Path, default=None, help="Optional output .npy costmap path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--near-distance-m", type=float, default=1.0)
    parser.add_argument("--far-distance-m", type=float, default=10.0)
    parser.add_argument("--near-half-width-m", type=float, default=0.8)
    parser.add_argument("--far-half-width-m", type=float, default=4.0)
    parser.add_argument("--horizon-fraction", type=float, default=0.45)
    parser.add_argument("--bottom-fraction", type=float, default=0.98)
    args = parser.parse_args()

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = rgb.shape[:2]
    calibration = GroundPlaneCalibration.default_for_image(
        width=width,
        height=height,
        near_distance_m=args.near_distance_m,
        far_distance_m=args.far_distance_m,
        near_half_width_m=args.near_half_width_m,
        far_half_width_m=args.far_half_width_m,
        horizon_fraction=args.horizon_fraction,
        bottom_fraction=args.bottom_fraction,
    )
    adapter = YoloSegmentationCostmap(
        args.weights,
        calibration=calibration,
        device=args.device,
        imgsz=args.imgsz,
        config=YoloMaskCostmapConfig(min_confidence=args.conf),
    )
    costmap, masks = adapter.predict_costmap(str(args.image))
    class_map = class_map_from_masks(masks, (height, width))

    if args.costmap_output:
        args.costmap_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.costmap_output, costmap)

    out = save_adapter_debug_panel(
        args.output,
        rgb=rgb,
        depth_m=None,
        class_map=class_map,
        points_robot_xyz=None,
        point_classes=None,
        costmap=costmap,
        action_text=f"YOLO masks: {', '.join(sorted(masks)) or 'none'}",
    )
    print(out)
    if args.costmap_output:
        print(args.costmap_output)


if __name__ == "__main__":
    main()
