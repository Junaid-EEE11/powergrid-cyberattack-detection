"""Causal temporal and electrical-consistency features."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

RAW_FEATURES: Final[tuple[str, ...]] = (
    "hour",
    "dayofweek",
    "ambient_temperature_c",
    "load_mw",
    "voltage_pu",
    "frequency_hz",
    "current_a",
    "power_factor",
    "thd_percent",
    "breaker_status",
    "missing_rate",
    "packet_delay_ms",
)

NETWORK_FEATURES: Final[tuple[str, ...]] = ("missing_rate", "packet_delay_ms")
PHYSICS_FEATURES: Final[tuple[str, ...]] = (
    "frequency_deviation_hz",
    "expected_current_a",
    "current_balance_error_ratio",
    "voltage_load_stress",
)
TEMPORAL_BASE: Final[tuple[str, ...]] = (
    "load_mw",
    "voltage_pu",
    "frequency_hz",
    "current_a",
    "missing_rate",
    "packet_delay_ms",
)

FEATURE_GROUPS: Final[dict[str, str]] = {
    **{name: "raw" for name in RAW_FEATURES},
    "hour_sin": "calendar",
    "hour_cos": "calendar",
    "day_sin": "calendar",
    "day_cos": "calendar",
    "frequency_deviation_hz": "physics",
    "expected_current_a": "physics",
    "current_balance_error_ratio": "physics",
    "voltage_load_stress": "physics",
    "row_missing_fraction": "quality",
    **{f"delta_{name}": "temporal" for name in TEMPORAL_BASE},
    **{f"innovation_{name}": "temporal" for name in TEMPORAL_BASE},
}


def _validate(frame: pd.DataFrame) -> None:
    required = {"episode_id", "timestep", *RAW_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.duplicated(["episode_id", "timestep"]).any():
        raise ValueError("episode_id/timestep pairs must be unique")


def build_features(frame: pd.DataFrame, feature_set: str = "full") -> pd.DataFrame:
    """Build features using only present and past observations within episodes.

    ``feature_set`` supports pre-declared ablations used by the experiment:
    ``full``, ``raw_only``, ``without_physics``, and ``without_network``.
    """

    _validate(frame)
    if feature_set not in {"full", "raw_only", "without_physics", "without_network"}:
        raise ValueError(f"Unknown feature set: {feature_set}")

    ordered = frame.sort_values(["episode_id", "timestep"])
    features = ordered.loc[:, RAW_FEATURES].copy()
    if feature_set == "raw_only":
        return features.reindex(frame.index)

    features["hour_sin"] = np.sin(2.0 * np.pi * ordered["hour"] / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * ordered["hour"] / 24.0)
    features["day_sin"] = np.sin(2.0 * np.pi * ordered["dayofweek"] / 7.0)
    features["day_cos"] = np.cos(2.0 * np.pi * ordered["dayofweek"] / 7.0)
    features["frequency_deviation_hz"] = (ordered["frequency_hz"] - 50.0).abs()
    expected_current = ordered["load_mw"] * 1e6 / (
        np.sqrt(3.0)
        * 33e3
        * ordered["voltage_pu"].clip(lower=0.1)
        * ordered["power_factor"].clip(lower=0.1)
    )
    features["expected_current_a"] = expected_current
    features["current_balance_error_ratio"] = (
        (ordered["current_a"] - expected_current).abs() / expected_current.clip(lower=1.0)
    )
    features["voltage_load_stress"] = ordered["load_mw"] / ordered["voltage_pu"].clip(
        lower=0.1
    )
    features["row_missing_fraction"] = ordered.loc[:, RAW_FEATURES].isna().mean(axis=1)

    grouped = ordered.groupby("episode_id", sort=False)
    for column in TEMPORAL_BASE:
        features[f"delta_{column}"] = grouped[column].diff()
        trailing_median = grouped[column].transform(
            lambda series: series.shift(1).rolling(window=4, min_periods=2).median()
        )
        features[f"innovation_{column}"] = ordered[column] - trailing_median

    if feature_set == "without_physics":
        features = features.drop(columns=list(PHYSICS_FEATURES))
    elif feature_set == "without_network":
        network_derived = [
            *NETWORK_FEATURES,
            *[f"delta_{name}" for name in NETWORK_FEATURES],
            *[f"innovation_{name}" for name in NETWORK_FEATURES],
        ]
        features = features.drop(columns=network_derived)

    return features.reindex(frame.index)
