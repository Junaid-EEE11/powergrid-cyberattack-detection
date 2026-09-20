from powergrid_security.simulation import SimulationConfig, generate_dataset
from train_cyberattack_detector import episode_stratified_split


def test_episode_holdout_has_no_group_overlap_and_all_scenarios() -> None:
    data = generate_dataset(
        SimulationConfig(episodes_per_scenario=8, steps_per_episode=48, seed=7)
    )
    train_ids, test_ids = episode_stratified_split(data, seed=7)
    assert set(train_ids).isdisjoint(test_ids)
    train_scenarios = set(data[data["episode_id"].isin(train_ids)]["scenario"])
    test_scenarios = set(data[data["episode_id"].isin(test_ids)]["scenario"])
    assert train_scenarios == test_scenarios == set(data["scenario"])
