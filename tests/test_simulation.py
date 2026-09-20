import numpy as np
import pandas as pd

from powergrid_security.simulation import ATTACK_CLASSES, SimulationConfig, generate_dataset


def test_generation_is_deterministic_and_has_expected_shape() -> None:
    config = SimulationConfig(episodes_per_scenario=5, steps_per_episode=48, seed=17)
    first = generate_dataset(config)
    second = generate_dataset(config)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5 * 4 * 48
    assert first["episode_id"].nunique() == 20
    assert set(first["scenario"]) == set(ATTACK_CLASSES)


def test_attacks_are_contiguous_and_normal_episodes_are_clean() -> None:
    data = generate_dataset(
        SimulationConfig(episodes_per_scenario=5, steps_per_episode=48, seed=23)
    )
    for _, episode in data.groupby("episode_id"):
        scenario = episode["scenario"].iloc[0]
        attacked = episode["attack_type"] != "normal"
        if scenario == "normal":
            assert not attacked.any()
        else:
            indices = np.flatnonzero(attacked.to_numpy())
            assert len(indices) >= 10
            assert np.all(np.diff(indices) == 1)
            assert set(episode.loc[attacked, "attack_type"]) == {scenario}


def test_normal_measurements_obey_approximate_three_phase_balance() -> None:
    data = generate_dataset(
        SimulationConfig(episodes_per_scenario=5, steps_per_episode=48, seed=31)
    )
    normal = data[data["attack_type"] == "normal"].dropna()
    expected_current = normal["load_mw"] * 1e6 / (
        np.sqrt(3) * 33e3 * normal["voltage_pu"] * normal["power_factor"]
    )
    relative_error = (normal["current_a"] - expected_current).abs() / expected_current
    assert relative_error.median() < 0.02
