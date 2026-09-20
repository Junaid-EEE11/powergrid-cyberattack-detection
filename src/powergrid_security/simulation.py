"""Physics-informed, episode-based synthetic smart-grid telemetry.

This is a transparent research simulator, not a power-flow solver.  It creates
correlated operating trajectories and then applies measurement-channel attack
operators.  Keeping the clean trajectory and attack operators conceptually
separate makes assumptions auditable and prevents the synthetic data from being
presented as field evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

ATTACK_CLASSES: Final[tuple[str, ...]] = (
    "normal",
    "false_data_injection",
    "dos_missing_data",
    "replay_attack",
)

SENSOR_COLUMNS: Final[tuple[str, ...]] = (
    "load_mw",
    "voltage_pu",
    "frequency_hz",
    "current_a",
    "power_factor",
    "thd_percent",
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a balanced set of normal and attacked episodes."""

    episodes_per_scenario: int = 24
    steps_per_episode: int = 96
    sampling_minutes: int = 15
    seed: int = 42

    def validate(self) -> None:
        if self.episodes_per_scenario < 5:
            raise ValueError("episodes_per_scenario must be at least 5 for grouped CV")
        if self.steps_per_episode < 48:
            raise ValueError("steps_per_episode must be at least 48")
        if self.sampling_minutes <= 0:
            raise ValueError("sampling_minutes must be positive")


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    values = np.empty(n, dtype=float)
    values[0] = rng.normal(0.0, sigma)
    innovations = rng.normal(0.0, sigma, n)
    for index in range(1, n):
        values[index] = phi * values[index - 1] + innovations[index]
    return values


def _clean_episode(
    episode_number: int,
    scenario: str,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = config.steps_per_episode
    start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=episode_number)
    timestamp = pd.date_range(start, periods=n, freq=f"{config.sampling_minutes}min")
    hour = timestamp.hour.to_numpy() + timestamp.minute.to_numpy() / 60.0
    day_of_week = timestamp.dayofweek.to_numpy()

    episode_load_scale = rng.normal(1.0, 0.055)
    temperature_offset = rng.normal(0.0, 1.7)
    ambient_temperature = (
        27.0
        + temperature_offset
        + 5.2 * np.sin(2.0 * np.pi * (hour - 14.0) / 24.0)
        + _ar1(rng, n, 0.90, 0.25)
    )
    morning_peak = 13.0 * np.exp(-0.5 * ((hour - 9.0) / 2.2) ** 2)
    evening_peak = 24.0 * np.exp(-0.5 * ((hour - 19.0) / 2.8) ** 2)
    weekday_effect = np.where(day_of_week < 5, 6.0, -3.0)
    cooling_load = 0.75 * np.maximum(ambient_temperature - 27.0, 0.0)
    load_mw = episode_load_scale * (
        58.0 + morning_peak + evening_peak + weekday_effect + cooling_load
    ) + _ar1(rng, n, 0.82, 1.15)

    power_factor = np.clip(
        0.955 - 0.00035 * (load_mw - 70.0) + _ar1(rng, n, 0.70, 0.0035),
        0.86,
        0.99,
    )
    voltage_pu = np.clip(
        1.018 - 0.00072 * (load_mw - 65.0) + _ar1(rng, n, 0.76, 0.0028),
        0.90,
        1.08,
    )
    load_ramp = np.diff(load_mw, prepend=load_mw[0])
    frequency_hz = 50.0 - 0.0012 * load_ramp + _ar1(rng, n, 0.58, 0.0065)
    expected_current = load_mw * 1e6 / (
        np.sqrt(3.0) * 33e3 * voltage_pu * power_factor
    )
    current_a = expected_current * (1.0 + _ar1(rng, n, 0.62, 0.0055))
    thd_percent = np.clip(
        1.7 + 0.018 * np.maximum(load_mw - 50.0, 0.0) + _ar1(rng, n, 0.55, 0.12),
        0.4,
        7.0,
    )

    return pd.DataFrame(
        {
            "episode_id": f"E{episode_number:04d}",
            "timestep": np.arange(n, dtype=int),
            "timestamp": timestamp,
            "hour": hour,
            "dayofweek": day_of_week,
            "ambient_temperature_c": ambient_temperature,
            "load_mw": load_mw,
            "voltage_pu": voltage_pu,
            "frequency_hz": frequency_hz,
            "current_a": current_a,
            "power_factor": power_factor,
            "thd_percent": thd_percent,
            "breaker_status": np.ones(n, dtype=int),
            "missing_rate": np.clip(rng.beta(1.2, 90.0, n), 0.0, 0.08),
            "packet_delay_ms": rng.lognormal(mean=2.35, sigma=0.30, size=n),
            "scenario": scenario,
            "event_id": "none",
            "attack_severity": 0.0,
            "attack_type": "normal",
        }
    )


def _event_bounds(config: SimulationConfig, rng: np.random.Generator) -> tuple[int, int]:
    n = config.steps_per_episode
    start_low = max(16, int(0.45 * n))
    start_high = max(start_low + 1, int(0.62 * n))
    start = int(rng.integers(start_low, start_high))
    min_length = max(10, int(0.20 * n))
    max_length = max(min_length + 1, min(int(0.36 * n), n - start))
    length = int(rng.integers(min_length, max_length + 1))
    return start, min(start + length, n)


def _apply_fdi(
    frame: pd.DataFrame,
    start: int,
    end: int,
    severity: float,
    rng: np.random.Generator,
) -> None:
    length = end - start
    direction = float(rng.choice([-1.0, 1.0]))
    envelope = np.sin(np.linspace(0.12, np.pi - 0.12, length))
    load_bias = direction * (0.035 + 0.075 * severity) * envelope
    frame.loc[start : end - 1, "load_mw"] *= 1.0 + load_bias
    frame.loc[start : end - 1, "voltage_pu"] += (
        direction * (0.006 + 0.018 * severity) * envelope
    )
    # An independently forged current channel creates a subtle physical residual.
    current_bias = -direction * (0.025 + 0.055 * severity) * envelope
    frame.loc[start : end - 1, "current_a"] *= 1.0 + current_bias
    frame.loc[start : end - 1, "frequency_hz"] += (
        direction * 0.008 * severity * envelope
    )


def _apply_dos(
    frame: pd.DataFrame,
    start: int,
    end: int,
    severity: float,
    rng: np.random.Generator,
) -> None:
    event_index = np.arange(start, end)
    frame.loc[event_index, "missing_rate"] = rng.uniform(
        0.20 + 0.20 * severity,
        0.50 + 0.35 * severity,
        len(event_index),
    )
    frame.loc[event_index, "packet_delay_ms"] *= rng.uniform(
        2.5,
        5.0 + 4.0 * severity,
        len(event_index),
    )
    dropout_probability = 0.14 + 0.30 * severity
    for column in SENSOR_COLUMNS:
        missing = rng.random(len(event_index)) < dropout_probability
        frame.loc[event_index[missing], column] = np.nan


def _apply_replay(
    frame: pd.DataFrame,
    start: int,
    end: int,
    severity: float,
    rng: np.random.Generator,
) -> None:
    max_lag = min(start - 1, max(8, int(0.38 * len(frame))))
    min_lag = max(6, int(0.16 * len(frame)))
    lag = int(rng.integers(min_lag, max_lag + 1))
    target = np.arange(start, end)
    source = target - lag
    replay_columns = [*SENSOR_COLUMNS, "breaker_status", "missing_rate"]
    frame.loc[target, replay_columns] = frame.loc[source, replay_columns].to_numpy()
    # Higher severity means less jitter and therefore more exact duplication.
    jitter_scale = 0.004 * (1.0 - severity)
    frame.loc[target, "load_mw"] *= 1.0 + rng.normal(0.0, jitter_scale, len(target))


def generate_dataset(config: SimulationConfig | None = None) -> pd.DataFrame:
    """Generate deterministic telemetry with isolated attack episodes."""

    config = config or SimulationConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)
    scenarios = np.repeat(ATTACK_CLASSES, config.episodes_per_scenario)
    scenarios = scenarios[rng.permutation(len(scenarios))]
    episodes: list[pd.DataFrame] = []

    for episode_number, scenario_value in enumerate(scenarios):
        scenario = str(scenario_value)
        frame = _clean_episode(episode_number, scenario, config, rng)
        if scenario != "normal":
            start, end = _event_bounds(config, rng)
            severity = float(rng.uniform(0.35, 1.0))
            if scenario == "false_data_injection":
                _apply_fdi(frame, start, end, severity, rng)
            elif scenario == "dos_missing_data":
                _apply_dos(frame, start, end, severity, rng)
            elif scenario == "replay_attack":
                _apply_replay(frame, start, end, severity, rng)
            frame.loc[start : end - 1, "attack_type"] = scenario
            frame.loc[start : end - 1, "event_id"] = f"A{episode_number:04d}"
            frame.loc[start : end - 1, "attack_severity"] = severity
        episodes.append(frame)

    data = pd.concat(episodes, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return data
