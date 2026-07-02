import numpy as np

from navigation.real_adapter.ground_projection import GroundPlaneCalibration
from navigation.real_adapter.spec import Stage49ObservationSpec
from navigation.real_adapter.yolo_costmap import (
    FREE_CORRIDOR_CLASS,
    TREE_ROW_CLASS,
    YoloMaskCostmapConfig,
    class_map_from_masks,
    masks_to_costmap,
)


def _calibration() -> GroundPlaneCalibration:
    return GroundPlaneCalibration.default_for_image(
        width=160,
        height=120,
        near_distance_m=1.0,
        far_distance_m=12.0,
        near_half_width_m=1.2,
        far_half_width_m=6.0,
        horizon_fraction=0.35,
        bottom_fraction=0.98,
    )


def test_yolo_masks_mark_tree_rows_and_clear_free_corridor_unknowns():
    spec = Stage49ObservationSpec()
    free = np.zeros((120, 160), dtype=bool)
    free[45:118, 60:100] = True
    rows = np.zeros_like(free)
    rows[30:118, 12:42] = True
    rows[30:118, 118:148] = True

    costmap = masks_to_costmap(
        {
            FREE_CORRIDOR_CLASS: free,
            TREE_ROW_CLASS: rows,
        },
        _calibration(),
        spec,
        YoloMaskCostmapConfig(mask_sample_stride_px=6, row_column_stride_px=5),
    )

    assert costmap.shape == (4, 32, 16)
    assert costmap.dtype == np.float32
    assert costmap.min() >= 0.0
    assert costmap.max() <= 1.0
    assert costmap[0].max() == 1.0
    assert costmap[1].max() == 0.0
    assert costmap[3, spec.visible_angle_mask(), :].min() == 0.0
    assert np.all(costmap[3, ~spec.visible_angle_mask(), :] == 1.0)


def test_empty_yolo_masks_leave_visible_area_unknown_by_default():
    spec = Stage49ObservationSpec()
    costmap = masks_to_costmap({}, _calibration(), spec)

    assert costmap[:3].max() == 0.0
    assert np.all(costmap[3] == 1.0)


def test_class_map_prefers_tree_rows_over_free_corridor_overlap():
    free = np.zeros((4, 4), dtype=bool)
    free[1:3, 1:3] = True
    rows = np.zeros_like(free)
    rows[2, 2] = True

    class_map = class_map_from_masks(
        {
            FREE_CORRIDOR_CLASS: free,
            TREE_ROW_CLASS: rows,
        },
        (4, 4),
    )

    assert class_map[0, 0] == "unknown"
    assert class_map[1, 1] == FREE_CORRIDOR_CLASS
    assert class_map[2, 2] == TREE_ROW_CLASS
