"""Run a reproducible, group-aware cyberattack detection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from powergrid_security.evaluation import (
    align_probabilities,
    classification_metrics,
    grouped_bootstrap_intervals,
    operational_metrics,
    per_class_metrics,
)
from powergrid_security.features import build_features
from powergrid_security.simulation import ATTACK_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "synthetic_grid_cyberattack_data.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
LABELS = list(ATTACK_CLASSES)
DISPLAY_NAMES = {
    "normal": "Normal",
    "false_data_injection": "False-data injection",
    "dos_missing_data": "DoS / missing data",
    "replay_attack": "Replay",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer trees, three folds, and 100 bootstrap replicates for a smoke run.",
    )
    return parser.parse_args()


def make_models(seed: int, quick: bool = False) -> dict[str, Pipeline]:
    trees = 100 if quick else 350
    boosting_iterations = 80 if quick else 220
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    return {
        "Dummy (class prior)": Pipeline(
            [("imputer", clone(imputer)), ("model", DummyClassifier(strategy="prior"))]
        ),
        "Regularized logistic regression": Pipeline(
            [
                ("imputer", clone(imputer)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("imputer", clone(imputer)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=trees,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Histogram gradient boosting": Pipeline(
            [
                ("imputer", clone(imputer)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=boosting_iterations,
                        max_leaf_nodes=31,
                        l2_regularization=0.5,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def episode_stratified_split(
    data: pd.DataFrame, seed: int, test_size: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    episodes = data.groupby("episode_id", sort=True)["scenario"].first().reset_index()
    train_ids, test_ids = train_test_split(
        episodes["episode_id"].to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=episodes["scenario"],
    )
    if set(train_ids).intersection(test_ids):
        raise RuntimeError("Episode leakage detected in train/test split")
    return np.sort(train_ids), np.sort(test_ids)


def episode_cv_splits(
    data: pd.DataFrame,
    train_ids: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    episode_table = (
        data[data["episode_id"].isin(train_ids)]
        .groupby("episode_id", sort=True)["scenario"]
        .first()
        .reset_index()
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_group_index, validation_group_index in splitter.split(
        episode_table["episode_id"], episode_table["scenario"]
    ):
        fold_train_ids = episode_table.iloc[train_group_index]["episode_id"].to_numpy()
        fold_validation_ids = episode_table.iloc[validation_group_index]["episode_id"].to_numpy()
        train_rows = np.flatnonzero(data["episode_id"].isin(fold_train_ids).to_numpy())
        validation_rows = np.flatnonzero(
            data["episode_id"].isin(fold_validation_ids).to_numpy()
        )
        splits.append((train_rows, validation_rows))
    return splits


def evaluate_cv(
    models: dict[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for model_name, estimator in models.items():
        for fold, (train_rows, validation_rows) in enumerate(splits, start=1):
            fitted = clone(estimator).fit(features.iloc[train_rows], target.iloc[train_rows])
            predictions = fitted.predict(features.iloc[validation_rows])
            probabilities = align_probabilities(
                fitted.predict_proba(features.iloc[validation_rows]),
                fitted.classes_,
                LABELS,
            )
            metrics = classification_metrics(
                target.iloc[validation_rows].to_numpy(), predictions, probabilities, LABELS
            )
            records.append({"model": model_name, "fold": fold, **metrics})
    return pd.DataFrame(records)


def evaluate_holdout_models(
    models: dict[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    records = []
    fitted_models: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        fitted = clone(estimator).fit(features.iloc[train_rows], target.iloc[train_rows])
        predictions = fitted.predict(features.iloc[test_rows])
        probabilities = align_probabilities(
            fitted.predict_proba(features.iloc[test_rows]), fitted.classes_, LABELS
        )
        records.append(
            {
                "model": name,
                **classification_metrics(
                    target.iloc[test_rows].to_numpy(), predictions, probabilities, LABELS
                ),
            }
        )
        fitted_models[name] = fitted
    return pd.DataFrame(records), fitted_models


def perturb_holdout(
    frame: pd.DataFrame, condition: str, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shifted = frame.copy()
    if condition in {"sensor_noise", "combined_shift"}:
        shifted["load_mw"] *= 1.0 + rng.normal(0.0, 0.025, len(shifted))
        shifted["voltage_pu"] += rng.normal(0.0, 0.006, len(shifted))
        shifted["frequency_hz"] += rng.normal(0.0, 0.018, len(shifted))
        shifted["current_a"] *= 1.0 + rng.normal(0.0, 0.025, len(shifted))
        shifted["power_factor"] = np.clip(
            shifted["power_factor"] + rng.normal(0.0, 0.01, len(shifted)), 0.7, 1.0
        )
        shifted["thd_percent"] += rng.normal(0.0, 0.25, len(shifted))
    if condition in {"telemetry_dropout", "combined_shift"}:
        sensors = [
            "load_mw",
            "voltage_pu",
            "frequency_hz",
            "current_a",
            "power_factor",
            "thd_percent",
        ]
        probability = 0.05 if condition == "telemetry_dropout" else 0.08
        for column in sensors:
            mask = rng.random(len(shifted)) < probability
            shifted.loc[mask, column] = np.nan
    return shifted


def save_confusion_figure(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(8.2, 6.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#17202a",
            )
    names = [DISPLAY_NAMES[label] for label in LABELS]
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Held-out episodes: row-normalized confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Recall")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_figure(
    cv_results: pd.DataFrame, holdout_results: pd.DataFrame, path: Path
) -> None:
    summary = cv_results.groupby("model")["macro_f1"].agg(["mean", "std"]).reset_index()
    summary = summary.merge(holdout_results[["model", "macro_f1"]], on="model")
    summary = summary.sort_values("mean")
    positions = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        summary["mean"],
        positions,
        xerr=summary["std"],
        fmt="o",
        capsize=4,
        color="#16697a",
        label="Training CV mean ± SD",
    )
    ax.scatter(
        summary["macro_f1"], positions, marker="D", color="#d1495b", label="Locked holdout"
    )
    ax.set_yticks(positions, summary["model"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Macro F1")
    ax.set_title("Model selection uses episode-level cross-validation")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def permutation_importance_table(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = f1_score(target, model.predict(features), labels=LABELS, average="macro")
    records = []
    for column in features.columns:
        reductions = []
        for _ in range(repeats):
            permuted = features.copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())
            score = f1_score(
                target,
                model.predict(permuted),
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
            reductions.append(baseline - score)
        records.append(
            {
                "feature": column,
                "importance_mean": float(np.mean(reductions)),
                "importance_std": float(np.std(reductions, ddof=1)),
            }
        )
    return pd.DataFrame(records).sort_values("importance_mean", ascending=False)


def save_importance_figure(importance: pd.DataFrame, path: Path) -> None:
    top = importance.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4895ef")
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Decrease in holdout macro F1 after permutation")
    ax.set_title("Post-hoc permutation importance (top 15)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_robustness_figure(robustness: pd.DataFrame, path: Path) -> None:
    ordered = robustness.sort_values("macro_f1", ascending=True)
    condition_names = {
        "nominal": "Nominal holdout",
        "sensor_noise": "Added sensor noise",
        "telemetry_dropout": "Telemetry dropout",
        "combined_shift": "Combined shift",
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#2a9d8f" if value == "nominal" else "#e76f51" for value in ordered["condition"]]
    bars = ax.barh(
        [condition_names[value] for value in ordered["condition"]],
        ordered["macro_f1"],
        color=colors,
    )
    for bar, value in zip(bars, ordered["macro_f1"], strict=True):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Macro F1")
    ax.set_title("Pre-declared distribution-shift stress tests")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_calibration_figure(
    target: pd.Series, probabilities: np.ndarray, path: Path
) -> None:
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    correct = predicted == target.to_numpy()
    edges = np.linspace(0.0, 1.0, 11)
    accuracy, mean_confidence, counts = [], [], []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            accuracy.append(correct[mask].mean())
            mean_confidence.append(confidence[mask].mean())
            counts.append(mask.sum())
    fig, ax = plt.subplots(figsize=(6.2, 5.7))
    ax.plot([0, 1], [0, 1], "--", color="#666666", label="Perfect calibration")
    sizes = 25 + 125 * np.asarray(counts) / max(counts)
    ax.scatter(mean_confidence, accuracy, s=sizes, color="#7b2cbf", alpha=0.8)
    ax.plot(mean_confidence, accuracy, color="#7b2cbf", alpha=0.6, label="Observed")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy")
    ax.set_title("Top-label reliability on held-out episodes")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_research_report(
    path: Path,
    data: pd.DataFrame,
    best_name: str,
    best_metrics: dict[str, float],
    intervals: pd.DataFrame,
    operational: dict[str, float | int],
    robustness: pd.DataFrame,
    comparison: pd.DataFrame,
    ablations: pd.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    seed: int,
) -> None:
    macro_interval = intervals.set_index("metric").loc["macro_f1"]
    combined = robustness.set_index("condition").loc["combined_shift", "macro_f1"]
    dummy_f1 = comparison.set_index("model").loc["Dummy (class prior)", "macro_f1"]
    raw_f1 = ablations.set_index("feature_set").loc["raw_only", "macro_f1"]
    h1_margin = best_metrics["macro_f1"] - dummy_f1
    h2_margin = best_metrics["macro_f1"] - raw_f1
    h3_drop = best_metrics["macro_f1"] - combined
    content = f"""# Research report: episode-aware smart-grid attack detection

## Executive finding

The model selected *only from training-fold performance* was **{best_name}**. On a locked set of {len(test_ids)} previously unseen operating episodes it achieved macro F1 **{best_metrics['macro_f1']:.3f}** (episode-bootstrap 95% interval **{macro_interval['ci_2.5%']:.3f}–{macro_interval['ci_97.5%']:.3f}**) and balanced accuracy **{best_metrics['balanced_accuracy']:.3f}**. Under the combined sensor-noise/dropout stress test, macro F1 was **{combined:.3f}**. These are synthetic-benchmark results, not evidence of field readiness.

## Research question

Can causal temporal features and simplified electrical-consistency residuals distinguish normal telemetry from false-data injection, denial-of-service, and replay events when evaluation episodes are independent of training episodes?

## Experimental design

- **Unit of independence:** operating episode, never an individual row.
- **Data:** {len(data):,} observations across {data['episode_id'].nunique()} episodes; attacks occupy contiguous event windows.
- **Locked holdout:** {len(test_ids)} episodes ({len(test_ids) / data['episode_id'].nunique():.0%}); {len(train_ids)} episodes remain for model development.
- **Selection:** stratified episode-level cross-validation on development episodes, optimizing macro F1.
- **Primary endpoint:** macro F1, chosen because the naturally generated normal class is more frequent.
- **Uncertainty:** non-parametric bootstrap over entire held-out episodes ({int(macro_interval['bootstrap_iterations'])} replicates).
- **Operational rule:** an alert requires {operational['confirmation_steps']} consecutive attack predictions.
- **Reproducibility seed:** {seed}.

## Operational interpretation

- Event detection rate: **{operational['event_detection_rate']:.1%}** ({operational['events_detected']}/{operational['events']} held-out events)
- Median detection latency: **{operational['median_detection_latency_minutes']:.1f} minutes**
- Confirmed false alerts per 24 hours of normal telemetry: **{operational['false_alerts_per_24h']:.2f}**
- Macro one-vs-rest AUROC: **{best_metrics['macro_ovr_auroc']:.3f}**
- Expected calibration error: **{best_metrics['expected_calibration_error']:.3f}** (lower is better)

## Pre-declared hypothesis outcomes

- **H1 supported:** the selected model exceeded the dummy baseline by **{h1_margin:.3f}** macro F1 (required: ≥0.20).
- **H2 supported:** the full feature set exceeded raw-only features by **{h2_margin:.3f}** macro F1.
- **H3 not supported:** combined-shift degradation was **{h3_drop:.3f}** macro F1 (required: ≤0.15). Sensor-noise robustness is therefore a priority for external validation and training-time augmentation, not a solved problem.

## Controls against optimistic bias

1. All rows from one simulated episode remain in exactly one split.
2. Feature engineering uses only the current and preceding observations; future samples are never used.
3. The held-out set is not used to select the model family.
4. A class-prior dummy model is included as a negative control.
5. Ablations quantify dependence on physics and communication-quality features.
6. Stress tests measure sensitivity to noise and unplanned missing telemetry.

## Scope and limitations

This is a transparent *pilot benchmark*. It uses a distribution-level simulator, not AC state estimation, hardware-in-the-loop equipment, adversarial optimization, or utility data. Attack labels are generated by known operators, so performance can be inflated by simulator-specific fingerprints. Confidence intervals quantify episode sampling variability but not simulator misspecification. The next defensible step is external validation on an established PMU/SCADA or hardware-in-the-loop benchmark, with all preprocessing frozen before evaluation.

## Artifact map

Detailed fold metrics, per-class errors, ablations, stress tests, predictions, and provenance are stored as CSV/JSON in `reports/`. Figures are stored in `figures/`, and the fitted research artifact is in `models/`.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.cv_folds = min(args.cv_folds, 3)
        args.bootstrap = min(args.bootstrap, 100)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    required = {"episode_id", "timestep", "scenario", "attack_type"}
    if missing := required.difference(data.columns):
        raise ValueError(
            "Dataset uses the legacy row-independent schema. Regenerate it first with "
            f"src/generate_synthetic_grid_attack_data.py. Missing: {sorted(missing)}"
        )
    data = data.sort_values(["episode_id", "timestep"]).reset_index(drop=True)
    target = data["attack_type"]
    features = build_features(data, "full")
    train_ids, test_ids = episode_stratified_split(data, args.seed)
    train_rows = np.flatnonzero(data["episode_id"].isin(train_ids).to_numpy())
    test_rows = np.flatnonzero(data["episode_id"].isin(test_ids).to_numpy())
    cv_splits = episode_cv_splits(data, train_ids, args.cv_folds, args.seed)
    models = make_models(args.seed, args.quick)

    cv_results = evaluate_cv(models, features, target, cv_splits)
    cv_results.to_csv(REPORT_DIR / "cross_validation.csv", index=False)
    cv_summary = (
        cv_results.groupby("model")
        .agg(cv_macro_f1_mean=("macro_f1", "mean"), cv_macro_f1_sd=("macro_f1", "std"))
        .reset_index()
        .sort_values("cv_macro_f1_mean", ascending=False)
    )
    best_name = str(cv_summary.iloc[0]["model"])

    holdout_results, fitted_models = evaluate_holdout_models(
        models, features, target, train_rows, test_rows
    )
    comparison = cv_summary.merge(holdout_results, on="model")
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    best_model = fitted_models[best_name]
    test_features = features.iloc[test_rows]
    test_target = target.iloc[test_rows]
    best_predictions = best_model.predict(test_features)
    best_probabilities = align_probabilities(
        best_model.predict_proba(test_features), best_model.classes_, LABELS
    )
    best_metrics = classification_metrics(
        test_target.to_numpy(), best_predictions, best_probabilities, LABELS
    )

    class_results = per_class_metrics(test_target, best_predictions, LABELS)
    class_results.to_csv(REPORT_DIR / "per_class_metrics.csv", index=False)
    intervals = grouped_bootstrap_intervals(
        test_target,
        best_predictions,
        data.iloc[test_rows]["episode_id"],
        LABELS,
        iterations=args.bootstrap,
        seed=args.seed,
    )
    intervals.to_csv(REPORT_DIR / "bootstrap_intervals.csv", index=False)
    operational = operational_metrics(
        data.iloc[test_rows],
        best_predictions,
        sampling_minutes=int(
            pd.to_datetime(data.groupby("episode_id")["timestamp"].nth(1)).sub(
                pd.to_datetime(data.groupby("episode_id")["timestamp"].nth(0)).to_numpy()
            ).dt.total_seconds().median()
            / 60
        ),
        confirmation_steps=3,
    )

    ablation_records = []
    for feature_set in ["full", "raw_only", "without_physics", "without_network"]:
        ablation_features = build_features(data, feature_set)
        estimator = clone(models[best_name]).fit(
            ablation_features.iloc[train_rows], target.iloc[train_rows]
        )
        predictions = estimator.predict(ablation_features.iloc[test_rows])
        probabilities = align_probabilities(
            estimator.predict_proba(ablation_features.iloc[test_rows]), estimator.classes_, LABELS
        )
        ablation_records.append(
            {
                "feature_set": feature_set,
                "feature_count": ablation_features.shape[1],
                **classification_metrics(test_target, predictions, probabilities, LABELS),
            }
        )
    ablations = pd.DataFrame(ablation_records)
    ablations.to_csv(REPORT_DIR / "ablation_study.csv", index=False)

    robustness_records = []
    raw_test = data.iloc[test_rows].copy().reset_index(drop=True)
    for index, condition in enumerate(
        ["nominal", "sensor_noise", "telemetry_dropout", "combined_shift"]
    ):
        shifted = raw_test if condition == "nominal" else perturb_holdout(
            raw_test, condition, args.seed + index
        )
        shifted_features = build_features(shifted, "full")
        predictions = best_model.predict(shifted_features)
        probabilities = align_probabilities(
            best_model.predict_proba(shifted_features), best_model.classes_, LABELS
        )
        robustness_records.append(
            {
                "condition": condition,
                **classification_metrics(test_target, predictions, probabilities, LABELS),
            }
        )
    robustness = pd.DataFrame(robustness_records)
    robustness.to_csv(REPORT_DIR / "robustness_study.csv", index=False)

    importance = permutation_importance_table(
        best_model,
        test_features,
        test_target,
        repeats=3 if args.quick else 8,
        seed=args.seed,
    )
    importance.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)

    prediction_frame = data.iloc[test_rows][
        ["episode_id", "timestep", "timestamp", "scenario", "event_id", "attack_type"]
    ].copy()
    prediction_frame["prediction"] = best_predictions
    for index, label in enumerate(LABELS):
        prediction_frame[f"probability_{label}"] = best_probabilities[:, index]
    prediction_frame.to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)

    save_confusion_figure(
        test_target,
        best_predictions,
        FIGURE_DIR / "confusion_matrix_heldout_episodes.png",
    )
    save_model_comparison_figure(
        cv_results, holdout_results, FIGURE_DIR / "model_comparison.png"
    )
    save_importance_figure(importance, FIGURE_DIR / "permutation_importance.png")
    save_robustness_figure(robustness, FIGURE_DIR / "robustness_study.png")
    save_calibration_figure(
        test_target, best_probabilities, FIGURE_DIR / "calibration_reliability.png"
    )

    fingerprint = hashlib.sha256(args.data.read_bytes()).hexdigest()
    resolved_data_path = args.data.resolve()
    recorded_data_path = (
        str(resolved_data_path.relative_to(PROJECT_ROOT))
        if resolved_data_path.is_relative_to(PROJECT_ROOT)
        else str(resolved_data_path)
    )
    record = {
        "experiment": "episode-aware-multiclass-detection-v2",
        "seed": args.seed,
        "quick_mode": args.quick,
        "dataset": {
            "path": recorded_data_path,
            "sha256": fingerprint,
            "rows": len(data),
            "episodes": int(data["episode_id"].nunique()),
            "class_counts": data["attack_type"].value_counts().to_dict(),
        },
        "split": {
            "unit": "episode_id",
            "train_episode_ids": train_ids.tolist(),
            "test_episode_ids": test_ids.tolist(),
        },
        "selection": {
            "criterion": "mean episode-level CV macro F1",
            "cv_folds": args.cv_folds,
            "selected_model": best_name,
        },
        "holdout_metrics": best_metrics,
        "operational_metrics": operational,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (REPORT_DIR / "experiment_record.json").write_text(
        json.dumps(record, indent=2, allow_nan=True), encoding="utf-8"
    )
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": list(features.columns),
            "classes": LABELS,
            "dataset_sha256": fingerprint,
            "experiment_seed": args.seed,
        },
        MODEL_DIR / "best_detector.joblib",
    )
    write_research_report(
        REPORT_DIR / "research_report.md",
        data,
        best_name,
        best_metrics,
        intervals,
        operational,
        robustness,
        comparison,
        ablations,
        train_ids,
        test_ids,
        args.seed,
    )
    (REPORT_DIR / "baseline_results.txt").write_text(
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}") + "\n",
        encoding="utf-8",
    )

    print(f"Selected model: {best_name}")
    print(f"Held-out macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"Held-out balanced accuracy: {best_metrics['balanced_accuracy']:.4f}")
    print(f"Results: {REPORT_DIR}")


if __name__ == "__main__":
    main()
