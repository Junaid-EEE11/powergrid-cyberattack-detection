# Episode-Aware Power Grid Cyberattack Detection

A reproducible Python project for benchmarking machine learning methods on synthetic smart-grid telemetry with episode-aware cyberattack scenarios.

This repository generates realistic operating episodes for a transmission/distribution-style grid and trains detectors to distinguish between normal operation and three attack behaviors:

- false data injection
- DoS / missing-data attacks
- replay attacks

The project is designed for research and experimentation rather than production deployment, with emphasis on transparent feature engineering, episode-level validation, and leakage-aware evaluation.

## Why this project exists

Power-grid cyberattack detection is often evaluated on row-level data without preserving the temporal grouping structure of real operating events. This benchmark instead models episodes as continuous operating trajectories and keeps all rows from the same episode in the same split.

The result is a cleaner evaluation setup for:

- causal and temporal feature design
- scenario-aware model comparison
- stress testing under sensor noise and telemetry loss
- attack detection using sequence-level and physics-informed signals

## Project highlights

- Synthetic smart-grid telemetry generator
- Episode-based data schema with `episode_id`, `timestep`, `scenario`, and `attack_type`
- Physics-informed and temporal feature engineering
- Baseline and comparison model training
- Holdout evaluation and robustness studies
- Artifact export to `reports/`, `figures/`, and `models/`

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── notebooks/
├── src/
│   ├── generate_synthetic_grid_attack_data.py
│   ├── train_cyberattack_detector.py
│   └── powergrid_security/
│       ├── __init__.py
│       ├── simulation.py
│       ├── features.py
│       └── evaluation.py
├── tests/
│   ├── test_simulation.py
│   ├── test_features.py
│   ├── test_evaluation.py
│   └── test_splitting.py
├── data/
├── reports/
├── figures/
├── models/
└── .gitignore
```

## Installation

Using a virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For development and notebook support:

```bash
pip install -r requirements-dev.txt
pip install -e .[dev,notebook]
```

## Quick start

### 1) Generate a synthetic dataset

```bash
grid-simulate --episodes-per-scenario 24 --steps-per-episode 96 --seed 42
```

This creates a dataset under `data/` by default.

### 2) Train and evaluate detectors

```bash
grid-evaluate --data data/synthetic_grid_cyberattack_data.csv --seed 42
```

This runs:

- episode-stratified train/test splitting
- cross-validation on development episodes
- model selection using macro F1
- holdout evaluation and reporting
- artifact generation in `reports/`, `figures/`, and `models/`

## Main entry points

The project exposes command-line scripts defined in `pyproject.toml`:

- `grid-simulate` → generate synthetic smart-grid attack data
- `grid-evaluate` → train and benchmark detection models

The source modules include:

- `src/generate_synthetic_grid_attack_data.py` — CLI for data generation
- `src/train_cyberattack_detector.py` — experiment pipeline and research reporting
- `src/powergrid_security/simulation.py` — synthetic telemetry simulator and attack templates
- `src/powergrid_security/features.py` — temporal, physics, and network-quality features
- `src/powergrid_security/evaluation.py` — metrics and uncertainty analysis

## Data generation design

The simulator creates a balanced mix of synthetic episodes with labeled operating conditions. Each episode includes time-indexed measurements such as:

- load and power demand
- voltage, frequency, current, and power factor
- harmonic distortion
- communication-quality indicators like missing telemetry and packet delay

Attack windows are applied to contiguous segments of each episode, which allows realistic sequential attack behavior and guards against label leakage from per-row randomization.

## Evaluation design

The training flow follows an episode-aware protocol:

- split by `episode_id`, not by individual rows
- stratify the split by scenario type
- use cross-validation only on the training episodes
- hold out a separate locked test set
- compare model performance with robust metrics and calibration checks

This is especially important when normal telemetry dominates and attack classes are imbalanced.

## Outputs

After running the evaluation pipeline, the project generates:

- `reports/` — CSV summaries, JSON provenance, and a research report
- `figures/` — confusion matrix, model comparison, importance plots, and calibration figures
- `models/` — serialized trained model artifacts
- `data/` — generated benchmark datasets

## Testing

Run the automated test suite with:

```bash
pytest
```

The tests cover:

- synthetic data generation behavior
- feature engineering correctness
- evaluation metric consistency
- split integrity and leakage prevention

## Citation

If you use this work in research, please cite the project metadata in `CITATION.cff`.

## Notes

This repository is a transparent research benchmark and synthetic pilot dataset. It is useful for experimenting with detection pipelines and benchmark design, but it does not represent field-collected utility telemetry or live operational system data.

## License

This repository does not currently include a project license file. If you plan to reuse or redistribute it, confirm the intended licensing terms before publication or commercial use.
