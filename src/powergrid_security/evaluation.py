"""Metrics and uncertainty estimates for grouped cyberattack evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: Sequence[str],
    labels: Sequence[str],
) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(model_classes)}
    return np.column_stack([probabilities[:, lookup[label]] for label in labels])


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: Sequence[str],
    bins: int = 10,
) -> float:
    label_array = np.asarray(labels)
    confidence = probabilities.max(axis=1)
    predicted = label_array[probabilities.argmax(axis=1)]
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        include = (confidence > left) & (confidence <= right)
        if include.any():
            error += include.mean() * abs(correct[include].mean() - confidence[include].mean())
    return float(error)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
) -> dict[str, float]:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    one_hot = np.column_stack([(truth == label).astype(float) for label in labels])
    recalls = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0
    )[1]
    aucs = [
        roc_auc_score((truth == label).astype(int), probabilities[:, index])
        for index, label in enumerate(labels)
    ]
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, prediction, labels=labels, average="weighted", zero_division=0)
        ),
        "matthews_correlation": float(matthews_corrcoef(truth, prediction)),
        "macro_ovr_auroc": float(np.mean(aucs)),
        "multiclass_brier": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
        "expected_calibration_error": expected_calibration_error(
            truth, probabilities, labels
        ),
    }


def per_class_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )


def grouped_bootstrap_intervals(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    groups: Sequence[str],
    labels: Sequence[str],
    iterations: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Episode bootstrap confidence intervals without pretending rows are IID."""

    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    group_array = np.asarray(groups)
    unique_groups = np.unique(group_array)
    group_indices = {group: np.flatnonzero(group_array == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }
    for _ in range(iterations):
        selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in selected_groups])
        sample_truth = truth[indices]
        sample_prediction = prediction[indices]
        recall = precision_recall_fscore_support(
            sample_truth,
            sample_prediction,
            labels=labels,
            zero_division=0,
        )[1]
        samples["accuracy"].append(float(accuracy_score(sample_truth, sample_prediction)))
        samples["balanced_accuracy"].append(float(np.mean(recall)))
        samples["macro_f1"].append(
            float(
                f1_score(
                    sample_truth,
                    sample_prediction,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
        )

    records = []
    for metric, values in samples.items():
        lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
        records.append(
            {
                "metric": metric,
                "median": median,
                "ci_2.5%": lower,
                "ci_97.5%": upper,
                "bootstrap_iterations": iterations,
            }
        )
    return pd.DataFrame(records)


def operational_metrics(
    metadata: pd.DataFrame,
    predictions: Sequence[str],
    sampling_minutes: int,
    confirmation_steps: int = 3,
) -> dict[str, float | int]:
    """Calculate event detection and false-alert burden for chronological episodes."""

    working = metadata.loc[:, ["episode_id", "timestep", "attack_type", "event_id"]].copy()
    working["prediction"] = np.asarray(predictions)
    latencies: list[float] = []
    attacked_events = 0
    false_alerts = 0
    normal_steps = 0

    for _, episode in working.sort_values(["episode_id", "timestep"]).groupby("episode_id"):
        predicted_attack = (episode["prediction"] != "normal").to_numpy(dtype=bool)
        confirmed = (
            pd.Series(predicted_attack)
            .rolling(confirmation_steps, min_periods=confirmation_steps)
            .sum()
            .eq(confirmation_steps)
            .to_numpy()
        )
        confirmed_start = confirmed & ~np.r_[False, confirmed[:-1]]
        normal = episode["attack_type"].eq("normal").to_numpy()
        false_alerts += int(np.sum(confirmed_start & normal))
        normal_steps += int(normal.sum())

        for event_id, event in episode[episode["event_id"] != "none"].groupby("event_id"):
            del event_id
            attacked_events += 1
            event_positions = np.flatnonzero(episode.index.isin(event.index))
            detections = event_positions[confirmed[event_positions]]
            if detections.size:
                onset = int(event_positions[0])
                # Latency is measured when the alert is actually confirmed, not at
                # the first contributing positive prediction.
                latency_steps = max(0, int(detections[0]) - onset)
                latencies.append(float(latency_steps * sampling_minutes))

    observed_normal_hours = normal_steps * sampling_minutes / 60.0
    return {
        "events": attacked_events,
        "events_detected": len(latencies),
        "event_detection_rate": len(latencies) / attacked_events if attacked_events else 0.0,
        "median_detection_latency_minutes": float(np.median(latencies)) if latencies else float("nan"),
        "false_alerts_per_24h": false_alerts * 24.0 / observed_normal_hours
        if observed_normal_hours
        else float("nan"),
        "confirmation_steps": confirmation_steps,
    }
