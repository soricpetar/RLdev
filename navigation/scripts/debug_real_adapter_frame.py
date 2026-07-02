#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigation.real_adapter.costmap import points_to_costmap
from navigation.real_adapter.debug_viz import save_adapter_debug_panel
from navigation.real_adapter.projection import depth_classes_to_robot_points
from navigation.real_adapter.spec import Stage49ObservationSpec


def _load_rgb(path: str | None):
    if not path:
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a debug panel for one real-adapter frame.")
    parser.add_argument("--rgb", type=str, default=None, help="Optional RGB image path")
    parser.add_argument("--depth", type=str, required=True, help="Depth .npy path in meters")
    parser.add_argument("--classes", type=str, required=True, help="Class map .npy path")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--camera-to-robot", type=str, default=None, help="Optional 4x4 .npy transform; defaults to identity")
    parser.add_argument("--action-text", type=str, default="")
    args = parser.parse_args()

    rgb = _load_rgb(args.rgb)
    depth = np.load(args.depth).astype(np.float32)
    class_map = np.load(args.classes, allow_pickle=True)
    transform = np.load(args.camera_to_robot).astype(np.float32) if args.camera_to_robot else np.eye(4, dtype=np.float32)
    spec = Stage49ObservationSpec()

    points, point_classes = depth_classes_to_robot_points(
        depth,
        class_map,
        {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy},
        transform,
        max_distance_m=spec.max_distance_m,
    )
    costmap = points_to_costmap(points, point_classes, spec)
    out = save_adapter_debug_panel(
        Path(args.output),
        rgb=rgb,
        depth_m=depth,
        class_map=class_map,
        points_robot_xyz=points,
        point_classes=point_classes,
        costmap=costmap,
        action_text=args.action_text,
    )
    print(out)


if __name__ == "__main__":
    main()
