import numpy as np
import pandas as pd

from powergrid_security.features import build_features
from powergrid_security.simulation import SimulationConfig, generate_dataset


def test_features_exclude_labels_and_metadata() -> None:
    data = generate_dataset(
        SimulationConfig(episodes_per_scenario=5, steps_per_episode=48, seed=11)
    )
    features = build_features(data)
    forbidden = {"attack_type", "scenario", "event_id", "attack_severity", "episode_id"}
    assert forbidden.isdisjoint(features.columns)
    assert len(features) == len(data)


def test_temporal_features_do_not_look_into_the_future() -> None:
    data = generate_dataset(
        SimulationConfig(episodes_per_scenario=5, steps_per_episode=48, seed=19)
    )
    episode_id = data["episode_id"].iloc[0]
    episode = data[data["episode_id"] == episode_id].copy().reset_index(drop=True)
    baseline = build_features(episode)
    modified = episode.copy()
    last = modified.index[-1]
    modified.loc[last, ["load_mw", "voltage_pu", "current_a"]] = [999.0, 0.5, 9999.0]
    changed = build_features(modified)

    pd.testing.assert_frame_equal(
        baseline.iloc[:-1], changed.iloc[:-1], check_dtype=False, check_exact=False
    )


def test_ablation_feature_sets_are_distinct() -> None:
    data = generate_dataset(
        SimulationConfig(episodes_per_scenario=5, steps_per_episode=48, seed=13)
    )
    full = build_features(data, "full")
    raw = build_features(data, "raw_only")
    no_physics = build_features(data, "without_physics")
    no_network = build_features(data, "without_network")
    assert raw.shape[1] < no_physics.shape[1] < full.shape[1]
    assert no_network.shape[1] < full.shape[1]
    assert np.isfinite(full["hour_sin"]).all()
