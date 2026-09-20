import numpy as np
import pandas as pd

from powergrid_security.evaluation import (
    classification_metrics,
    grouped_bootstrap_intervals,
    operational_metrics,
)


def test_metrics_and_group_bootstrap_have_valid_ranges() -> None:
    labels = ["normal", "attack"]
    truth = np.array(["normal", "normal", "attack", "attack"] * 3)
    prediction = np.array(["normal", "attack", "attack", "attack"] * 3)
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8]] * 3)
    groups = np.repeat(["E1", "E2", "E3"], 4)

    metrics = classification_metrics(truth, prediction, probabilities, labels)
    assert all(0 <= metrics[name] <= 1 for name in ["accuracy", "macro_f1"])
    intervals = grouped_bootstrap_intervals(
        truth, prediction, groups, labels, iterations=25, seed=3
    )
    assert set(intervals["metric"]) == {"accuracy", "balanced_accuracy", "macro_f1"}
    assert (intervals["ci_2.5%"] <= intervals["ci_97.5%"] ).all()


def test_operational_latency_includes_confirmation_delay() -> None:
    metadata = pd.DataFrame(
        {
            "episode_id": ["E1"] * 7,
            "timestep": range(7),
            "attack_type": ["normal", "normal", "attack", "attack", "attack", "attack", "normal"],
            "event_id": ["none", "none", "A1", "A1", "A1", "A1", "none"],
        }
    )
    predictions = ["normal", "normal", "attack", "attack", "attack", "attack", "normal"]
    metrics = operational_metrics(metadata, predictions, sampling_minutes=15, confirmation_steps=3)
    assert metrics["event_detection_rate"] == 1.0
    assert metrics["median_detection_latency_minutes"] == 30.0
