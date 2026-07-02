import numpy as np

from navigation.real_adapter.observation import make_stage49_observation
from navigation.real_adapter.spec import Stage49ObservationSpec


def test_make_stage49_observation_shape_dtype_and_order():
    spec = Stage49ObservationSpec()
    costmap = np.zeros((4, 32, 16), dtype=np.float32)
    costmap[0, 16, 2] = 1.0
    obs = make_stage49_observation(
        costmap,
        goal_distance_m=12.5,
        goal_heading_error_rad=np.pi / 2,
        speed_mps=1.5,
        yaw_rate_rps=0.25,
        previous_steering=-0.5,
        spec=spec,
    )
    assert obs.shape == (2055,)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs[:2048], costmap.reshape(-1))
    np.testing.assert_allclose(obs[2048:2051], [0.5, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(obs[2051:], [0.5, 0.25, -0.5, 1.0], atol=1e-6)


def test_make_stage49_observation_clips_goal_and_state():
    costmap = np.zeros((4, 32, 16), dtype=np.float32)
    obs = make_stage49_observation(
        costmap,
        goal_distance_m=999.0,
        goal_heading_error_rad=0.0,
        speed_mps=99.0,
        yaw_rate_rps=-99.0,
        previous_steering=99.0,
    )
    np.testing.assert_allclose(obs[2048:2051], [1.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs[2051:], [1.0, -1.0, 1.0, 1.0], atol=1e-6)
