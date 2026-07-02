import numpy as np
from PIL import Image

from navigation.real_adapter.debug_viz import polar_costmap_to_cartesian, save_adapter_debug_panel
from navigation.real_adapter.costmap import points_to_costmap
from navigation.real_adapter.spec import Stage49ObservationSpec


def test_save_adapter_debug_panel_writes_png(tmp_path):
    spec = Stage49ObservationSpec()
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 128
    depth = np.linspace(1.0, 4.0, 64, dtype=np.float32).reshape(8, 8)
    class_map = np.full((8, 8), "tree", dtype=object)
    points = np.asarray([[3.0, 0.0, 0.0], [4.0, 1.0, 0.0]], dtype=np.float32)
    classes = np.asarray(["tree", "person"], dtype=object)
    costmap = points_to_costmap(points, classes, spec)
    out = tmp_path / "debug.png"

    result = save_adapter_debug_panel(
        out,
        rgb=rgb,
        depth_m=depth,
        class_map=class_map,
        points_robot_xyz=points,
        point_classes=classes,
        costmap=costmap,
        action_text="action=8 steering=0 throttle=+1",
    )

    assert result == out
    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.size[0] > 100
        assert img.size[1] > 100


def test_polar_costmap_to_cartesian_projects_forward_obstacle():
    spec = Stage49ObservationSpec()
    points = np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32)
    classes = np.asarray(["tree"], dtype=object)
    costmap = points_to_costmap(points, classes, spec)

    cartesian, extent = polar_costmap_to_cartesian(costmap, spec, x_bins=48, y_bins=48, lateral_extent_m=6.0)

    assert cartesian.shape == (4, 48, 48)
    assert cartesian.dtype == np.float32
    assert extent == (-6.0, 6.0, 0.0, 24.0)
    assert cartesian[0].max() == 1.0
    assert cartesian[1].max() == 0.0


def test_polar_costmap_to_cartesian_marks_out_of_range_as_unknown():
    spec = Stage49ObservationSpec()
    costmap = np.zeros((4, 32, 16), dtype=np.float32)

    cartesian, _ = polar_costmap_to_cartesian(costmap, spec, x_bins=24, y_bins=24, lateral_extent_m=24.0)

    assert cartesian[3].max() == 1.0
