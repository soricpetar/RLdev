import numpy as np

from navigation.real_adapter.spec import Stage49ObservationSpec


def test_stage49_spec_matches_policy_contract():
    spec = Stage49ObservationSpec()
    assert spec.channels == 4
    assert spec.angle_bins == 32
    assert spec.distance_bins == 16
    assert spec.costmap_size == 2048
    assert spec.vector_size == 7
    assert spec.flat_size == 2055
    assert spec.distance_bin_width_m == 1.5
    assert spec.visible_angle_mask().shape == (32,)
    assert spec.visible_angle_mask().dtype == np.bool_
    assert int(spec.visible_angle_mask().sum()) == 8


def test_class_mapping_is_conservative():
    spec = Stage49ObservationSpec()
    assert spec.class_to_channel_and_cost("person") == (1, 1.0)
    assert spec.class_to_channel_and_cost("wall") == (0, 1.0)
    assert spec.class_to_channel_and_cost("tree") == (0, 1.0)
    assert spec.class_to_channel_and_cost("bush") == (2, 0.42)
    assert spec.class_to_channel_and_cost("pothole") == (2, 0.65)
    assert spec.class_to_channel_and_cost("unknown") == (0, 1.0)
