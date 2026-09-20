"""Command-line entry point for the episode-based telemetry simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

from powergrid_security.simulation import SimulationConfig, generate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "synthetic_grid_cyberattack_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-per-scenario", type=int, default=24)
    parser.add_argument("--steps-per-episode", type=int, default=96)
    parser.add_argument("--sampling-minutes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        episodes_per_scenario=args.episodes_per_scenario,
        steps_per_episode=args.steps_per_episode,
        sampling_minutes=args.sampling_minutes,
        seed=args.seed,
    )
    data = generate_dataset(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(
        f"Saved {len(data):,} observations from "
        f"{data['episode_id'].nunique()} episodes to {args.output}"
    )
    print(data["attack_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
