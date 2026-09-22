#!/usr/bin/env python3
"""Automated Predictive Analysis for Context Health Session Replay (Milestone 12B).

Investigates whether relevant_context_ratio contains predictive information
about future observable degradation beyond context_utilization and task_complexity.

Important Disclaimers
---------------------
This is EXPERIMENTAL INFRASTRUCTURE for offline analysis.

It does NOT:
- Validate the Context Health hypothesis on real coding-agent sessions.
- Prove that context signals predict degradation in production.
- Modify the Context Health scoring engine.

The synthetic session generator intentionally encodes relationships between
signals and future outcomes. The purpose of this script is to verify that the
experimental analysis pipeline can recover those relationships and perform
controlled comparisons without temporal leakage.

Standard Library Only
---------------------
No external libraries (no numpy, scipy, pandas, sklearn, statsmodels, matplotlib).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


OUTCOME_NAMES = [
    "next_turn_correction",
    "next_turn_tool_retry",
    "next_turn_contradiction",
    "next_turn_rework",
    "next_turn_abandoned",
]

SIGNAL_NAMES = [
    "context_utilization",
    "relevant_context_ratio",
    "task_complexity",
    "health_score",
]


@dataclass(frozen=True)
class PredictorOutcomeRecord:
    """Temporal predictor→outcome record loaded from session replay dataset."""

    session_id: str
    predictor_turn: int
    outcome_turn: int

    # Predictors (turn N)
    context_utilization: float
    relevant_context_ratio: float
    task_complexity: float
    contradiction_count: int
    correction_count: int
    retry_count: int
    health_score: float
    health_status: str

    # Outcomes (turn N+1)
    next_turn_correction: bool
    next_turn_tool_retry: bool
    next_turn_contradiction: bool
    next_turn_rework: bool
    next_turn_abandoned: bool


# ---------------------------------------------------------------------------
# Step 1: Load and validate dataset
# ---------------------------------------------------------------------------


def validate_record_dict(data: dict[str, Any], line_num: int) -> PredictorOutcomeRecord:
    """Validate a raw dictionary from JSONL against schema requirements.

    Fails loudly with ValueError on any malformed record or value out of bounds.
    Does not silently repair invalid data.
    """
    # Required field existence check
    required_fields = [
        "session_id",
        "predictor_turn",
        "outcome_turn",
        "context_utilization",
        "relevant_context_ratio",
        "task_complexity",
        "contradiction_count",
        "correction_count",
        "retry_count",
        "health_score",
        "health_status",
        "next_turn_correction",
        "next_turn_tool_retry",
        "next_turn_contradiction",
        "next_turn_rework",
        "next_turn_abandoned",
    ]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Line {line_num}: missing required field '{field}'")

    session_id = data["session_id"]
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError(f"Line {line_num}: session_id must be a non-empty string")

    # In Python bool is subclass of int, so check bool first where relevant
    predictor_turn = data["predictor_turn"]
    if isinstance(predictor_turn, bool) or not isinstance(predictor_turn, int) or predictor_turn < 0:
        raise ValueError(f"Line {line_num}: predictor_turn must be an integer >= 0, got {predictor_turn}")

    outcome_turn = data["outcome_turn"]
    if isinstance(outcome_turn, bool) or not isinstance(outcome_turn, int) or outcome_turn != predictor_turn + 1:
        raise ValueError(
            f"Line {line_num}: outcome_turn must be predictor_turn + 1, "
            f"got outcome_turn={outcome_turn}, predictor_turn={predictor_turn}"
        )

    # Float signal validations
    for sig_name in ["context_utilization", "relevant_context_ratio", "task_complexity"]:
        val = data[sig_name]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
            raise ValueError(f"Line {line_num}: {sig_name} must be float in [0.0, 1.0], got {val}")

    health_score = data["health_score"]
    if isinstance(health_score, bool) or not isinstance(health_score, (int, float)) or not (0.0 <= health_score <= 100.0):
        raise ValueError(f"Line {line_num}: health_score must be float in [0.0, 100.0], got {health_score}")

    health_status = data["health_status"]
    if not isinstance(health_status, str) or not health_status.strip():
        raise ValueError(f"Line {line_num}: health_status must be a non-empty string")

    # Count validations
    for count_name in ["contradiction_count", "correction_count", "retry_count"]:
        c = data[count_name]
        if isinstance(c, bool) or not isinstance(c, int) or c < 0:
            raise ValueError(f"Line {line_num}: {count_name} must be an integer >= 0, got {c}")

    # Boolean outcome validations
    for out_name in OUTCOME_NAMES:
        b = data[out_name]
        if not isinstance(b, bool):
            raise ValueError(f"Line {line_num}: {out_name} must be a boolean, got {type(b).__name__}")

    return PredictorOutcomeRecord(
        session_id=session_id,
        predictor_turn=predictor_turn,
        outcome_turn=outcome_turn,
        context_utilization=float(data["context_utilization"]),
        relevant_context_ratio=float(data["relevant_context_ratio"]),
        task_complexity=float(data["task_complexity"]),
        contradiction_count=data["contradiction_count"],
        correction_count=data["correction_count"],
        retry_count=data["retry_count"],
        health_score=float(health_score),
        health_status=health_status,
        next_turn_correction=bool(data["next_turn_correction"]),
        next_turn_tool_retry=bool(data["next_turn_tool_retry"]),
        next_turn_contradiction=bool(data["next_turn_contradiction"]),
        next_turn_rework=bool(data["next_turn_rework"]),
        next_turn_abandoned=bool(data["next_turn_abandoned"]),
    )


def load_dataset(input_path: Path) -> list[PredictorOutcomeRecord]:
    """Read and validate a JSONL session replay dataset.

    Fails loudly on non-existent files, empty files, malformed JSON, or invalid schemas.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records: list[PredictorOutcomeRecord] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as err:
                raise ValueError(f"Line {line_num}: invalid JSON syntax: {err}") from err

            if not isinstance(data, dict):
                raise ValueError(f"Line {line_num}: each line must be a JSON object, got {type(data).__name__}")

            records.append(validate_record_dict(data, line_num))

    if not records:
        raise ValueError(f"Dataset at {input_path} contains zero valid records")

    return records


# ---------------------------------------------------------------------------
# Step 2: Dataset summary statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalStats:
    min: float
    max: float
    mean: float


@dataclass(frozen=True)
class OutcomeRateStat:
    count: int
    total: int
    rate: float


@dataclass(frozen=True)
class DatasetSummary:
    total_records: int
    total_sessions: int
    records_per_session: float
    utilization: SignalStats
    relevance: SignalStats
    complexity: SignalStats
    health: SignalStats
    outcome_rates: dict[str, OutcomeRateStat]


def compute_dataset_summary(records: list[PredictorOutcomeRecord]) -> DatasetSummary:
    """Compute overall descriptive statistics for the loaded dataset."""
    n = len(records)
    sessions = set(r.session_id for r in records)
    num_sessions = len(sessions)
    rps = n / num_sessions if num_sessions > 0 else 0.0

    def get_stats(values: list[float]) -> SignalStats:
        return SignalStats(
            min=round(min(values), 4),
            max=round(max(values), 4),
            mean=round(sum(values) / len(values), 4),
        )

    util_stats = get_stats([r.context_utilization for r in records])
    rel_stats = get_stats([r.relevant_context_ratio for r in records])
    comp_stats = get_stats([r.task_complexity for r in records])
    health_stats = get_stats([r.health_score for r in records])

    outcome_rates: dict[str, OutcomeRateStat] = {}
    for name in OUTCOME_NAMES:
        c = sum(1 for r in records if getattr(r, name))
        rate = round(c / n, 4) if n > 0 else 0.0
        outcome_rates[name] = OutcomeRateStat(count=c, total=n, rate=rate)

    return DatasetSummary(
        total_records=n,
        total_sessions=num_sessions,
        records_per_session=round(rps, 2),
        utilization=util_stats,
        relevance=rel_stats,
        complexity=comp_stats,
        health=health_stats,
        outcome_rates=outcome_rates,
    )


# ---------------------------------------------------------------------------
# Step 3: Simple univariate analysis (Bucketing)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketStat:
    bucket_label: str
    count: int
    outcome_count: int
    outcome_rate: float


SIGNAL_BUCKET_BINS = [
    ("0.0–0.2", 0.0, 0.2),
    ("0.2–0.4", 0.2, 0.4),
    ("0.4–0.6", 0.4, 0.6),
    ("0.6–0.8", 0.6, 0.8),
    ("0.8–1.0", 0.8, 1.000001),
]

HEALTH_BUCKET_BINS = [
    ("80–100", 80.0, 100.000001),
    ("60–80", 60.0, 80.0),
    ("40–60", 40.0, 60.0),
    ("20–40", 20.0, 40.0),
    ("0–20", 0.0, 20.0),
]


def assign_bucket(val: float, bins: list[tuple[str, float, float]]) -> str:
    """Assign a scalar value to a named bucket interval."""
    for label, low, high in bins:
        if low <= val < high:
            return label
    # Boundary edge case fallback
    return bins[-1][0]


def compute_univariate_buckets(
    records: list[PredictorOutcomeRecord],
) -> dict[str, dict[str, list[BucketStat]]]:
    """Compute outcome counts and rates across fixed bins for each signal and outcome."""
    results: dict[str, dict[str, list[BucketStat]]] = {}

    signals_to_analyze = [
        ("context_utilization", SIGNAL_BUCKET_BINS),
        ("relevant_context_ratio", SIGNAL_BUCKET_BINS),
        ("task_complexity", SIGNAL_BUCKET_BINS),
        ("health_score", HEALTH_BUCKET_BINS),
    ]

    for outcome in OUTCOME_NAMES:
        results[outcome] = {}
        for sig_name, bin_defs in signals_to_analyze:
            bucket_counts: dict[str, int] = {label: 0 for label, _, _ in bin_defs}
            bucket_positives: dict[str, int] = {label: 0 for label, _, _ in bin_defs}

            for r in records:
                sig_val = getattr(r, sig_name)
                b_label = assign_bucket(sig_val, bin_defs)
                bucket_counts[b_label] += 1
                if getattr(r, outcome):
                    bucket_positives[b_label] += 1

            bucket_stats: list[BucketStat] = []
            for label, _, _ in bin_defs:
                cnt = bucket_counts[label]
                pos = bucket_positives[label]
                rate = round(pos / cnt, 4) if cnt > 0 else 0.0
                bucket_stats.append(
                    BucketStat(
                        bucket_label=label,
                        count=cnt,
                        outcome_count=pos,
                        outcome_rate=rate,
                    )
                )
            results[outcome][sig_name] = bucket_stats

    return results


# ---------------------------------------------------------------------------
# Step 4: Correlation analysis
# ---------------------------------------------------------------------------


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Calculate Pearson correlation coefficient r between two series.

    Returns 0.0 if either series has zero variance.
    Clamped to [-1.0, 1.0].
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0

    r = cov / denom
    return round(max(-1.0, min(1.0, r)), 4)


@dataclass(frozen=True)
class CorrelationResults:
    outcome_correlations: dict[str, dict[str, float]]
    signal_health_correlations: dict[str, float]


def compute_correlations(records: list[PredictorOutcomeRecord]) -> CorrelationResults:
    """Compute Pearson correlations between all predictor signals and outcomes."""
    util = [r.context_utilization for r in records]
    rel = [r.relevant_context_ratio for r in records]
    comp = [r.task_complexity for r in records]
    health = [r.health_score for r in records]

    outcome_corrs: dict[str, dict[str, float]] = {}
    for outcome in OUTCOME_NAMES:
        y = [1.0 if getattr(r, outcome) else 0.0 for r in records]
        outcome_corrs[outcome] = {
            "context_utilization": pearson_correlation(util, y),
            "relevant_context_ratio": pearson_correlation(rel, y),
            "task_complexity": pearson_correlation(comp, y),
            "health_score": pearson_correlation(health, y),
        }

    signal_health = {
        "context_utilization": pearson_correlation(util, health),
        "relevant_context_ratio": pearson_correlation(rel, health),
        "task_complexity": pearson_correlation(comp, health),
    }

    return CorrelationResults(
        outcome_correlations=outcome_corrs,
        signal_health_correlations=signal_health,
    )


# ---------------------------------------------------------------------------
# Step 6: Session-level train / test split
# ---------------------------------------------------------------------------


def split_by_session(
    records: list[PredictorOutcomeRecord],
    train_ratio: float = 0.8,
) -> tuple[list[PredictorOutcomeRecord], list[PredictorOutcomeRecord], list[str], list[str]]:
    """Perform a deterministic session-level train/test split.

    Uses sorted session IDs to partition sessions into disjoint train and test sets.
    Guarantees no session appears in both sets.
    """
    sorted_sessions = sorted(list(set(r.session_id for r in records)))
    n_train_sessions = int(len(sorted_sessions) * train_ratio)

    train_session_ids = sorted_sessions[:n_train_sessions]
    test_session_ids = sorted_sessions[n_train_sessions:]

    train_set = set(train_session_ids)
    test_set = set(test_session_ids)

    train_records = [r for r in records if r.session_id in train_set]
    test_records = [r for r in records if r.session_id in test_set]

    return train_records, test_records, train_session_ids, test_session_ids


# ---------------------------------------------------------------------------
# Step 7 & 8: Standardization & Logistic Regression from scratch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalerParams:
    means: list[float]
    stddevs: list[float]


def fit_scaler(X_train: list[list[float]]) -> ScalerParams:
    """Calculate feature means and population standard deviations from training data only."""
    if not X_train:
        return ScalerParams([], [])
    n = len(X_train)
    d = len(X_train[0])
    means: list[float] = []
    stddevs: list[float] = []

    for j in range(d):
        col = [row[j] for row in X_train]
        mean_j = sum(col) / n
        var_j = sum((val - mean_j) ** 2 for val in col) / n
        std_j = math.sqrt(var_j)
        if std_j < 1e-12:
            std_j = 1.0  # Safe fallback for constant features
        means.append(mean_j)
        stddevs.append(std_j)

    return ScalerParams(means, stddevs)


def transform_features(X: list[list[float]], scaler: ScalerParams) -> list[list[float]]:
    """Standardize feature matrix using precomputed scaler parameters."""
    return [
        [(val - m) / s for val, m, s in zip(row, scaler.means, scaler.stddevs)]
        for row in X
    ]


def sigmoid(z: float) -> float:
    """Numerically stable sigmoid function."""
    if z >= 35.0:
        return 1.0
    elif z <= -35.0:
        return 0.0
    elif z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


@dataclass(frozen=True)
class LogisticRegressionModel:
    weights: list[float]
    bias: float
    feature_names: list[str]
    scaler: ScalerParams
    learning_rate: float
    iterations: int


def train_logistic_regression(
    X_train: list[list[float]],
    y_train: list[int],
    feature_names: list[str],
    learning_rate: float = 0.05,
    iterations: int = 2000,
) -> LogisticRegressionModel:
    """Fit a binary logistic regression model using deterministic gradient descent."""
    n = len(X_train)
    d = len(feature_names)
    scaler = fit_scaler(X_train)
    X_norm = transform_features(X_train, scaler)

    # Deterministic zero initialization
    weights = [0.0] * d
    bias = 0.0

    inv_n = 1.0 / n if n > 0 else 0.0

    for _ in range(iterations):
        grad_w = [0.0] * d
        grad_b = 0.0

        for i in range(n):
            row = X_norm[i]
            z = bias + sum(w * x for w, x in zip(weights, row))
            p = sigmoid(z)
            diff = p - y_train[i]
            for j in range(d):
                grad_w[j] += diff * row[j]
            grad_b += diff

        for j in range(d):
            weights[j] -= learning_rate * (grad_w[j] * inv_n)
        bias -= learning_rate * (grad_b * inv_n)

    return LogisticRegressionModel(
        weights=[round(w, 6) for w in weights],
        bias=round(bias, 6),
        feature_names=list(feature_names),
        scaler=scaler,
        learning_rate=learning_rate,
        iterations=iterations,
    )


def predict_probabilities(model: LogisticRegressionModel, X: list[list[float]]) -> list[float]:
    """Compute predicted probabilities for feature matrix X."""
    X_norm = transform_features(X, model.scaler)
    probs: list[float] = []
    for row in X_norm:
        z = model.bias + sum(w * x for w, x in zip(model.weights, row))
        probs.append(sigmoid(z))
    return probs


# ---------------------------------------------------------------------------
# Step 9: Evaluation metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationMetrics:
    log_loss: float
    accuracy: float
    precision: float
    recall: float
    sample_count: int


def evaluate_predictions(
    y_true: list[int],
    y_prob: list[float],
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Compute binary log loss, accuracy, precision, and recall on predictions."""
    n = len(y_true)
    if n == 0:
        return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0)

    eps = 1e-15
    total_loss = 0.0
    correct = 0
    tp = 0
    fp = 0
    fn = 0

    for y, p in zip(y_true, y_prob):
        p_c = max(eps, min(1.0 - eps, p))
        total_loss += -(y * math.log(p_c) + (1.0 - y) * math.log(1.0 - p_c))

        pred = 1 if p >= threshold else 0
        if pred == y:
            correct += 1
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1

    return EvaluationMetrics(
        log_loss=round(total_loss / n, 4),
        accuracy=round(correct / n, 4),
        precision=round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
        recall=round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
        sample_count=n,
    )


# ---------------------------------------------------------------------------
# Step 11: Shuffled relevance helper
# ---------------------------------------------------------------------------


def shuffle_feature_values(values: list[float], seed: int) -> list[float]:
    """Deterministically permute a list of scalar values using a fixed seed."""
    shuffled = list(values)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


# ---------------------------------------------------------------------------
# Step 5, 9, 10, 11: Controlled comparison execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelComparisonDeltas:
    delta_log_loss: float
    delta_accuracy: float
    delta_precision: float
    delta_recall: float


@dataclass(frozen=True)
class OutcomeComparisonResult:
    outcome: str
    model_a: EvaluationMetrics
    model_b: EvaluationMetrics
    model_c: EvaluationMetrics
    model_b_shuffled: EvaluationMetrics
    deltas_b_minus_a: ModelComparisonDeltas
    weights_model_a: dict[str, float]
    weights_model_b: dict[str, float]
    weights_model_c: dict[str, float]
    weights_model_b_shuffled: dict[str, float]


def run_controlled_comparison(
    train_records: list[PredictorOutcomeRecord],
    test_records: list[PredictorOutcomeRecord],
    learning_rate: float = 0.05,
    iterations: int = 2000,
    shuffle_seed: int = 42,
) -> dict[str, OutcomeComparisonResult]:
    """Train and evaluate Model A, Model B, Model C, and Model B_shuffled across all outcomes."""
    results: dict[str, OutcomeComparisonResult] = {}

    # Extract base predictor features for train and test
    train_util = [r.context_utilization for r in train_records]
    train_comp = [r.task_complexity for r in train_records]
    train_rel = [r.relevant_context_ratio for r in train_records]
    train_health = [r.health_score for r in train_records]

    test_util = [r.context_utilization for r in test_records]
    test_comp = [r.task_complexity for r in test_records]
    test_rel = [r.relevant_context_ratio for r in test_records]
    test_health = [r.health_score for r in test_records]

    # Create shuffled relevance within train and test splits independently (no cross-split leakage)
    shuffled_train_rel = shuffle_feature_values(train_rel, seed=shuffle_seed)
    shuffled_test_rel = shuffle_feature_values(test_rel, seed=shuffle_seed + 1)

    # Build feature matrices
    # Model A: utilization + complexity
    X_train_a = [[u, c] for u, c in zip(train_util, train_comp)]
    X_test_a = [[u, c] for u, c in zip(test_util, test_comp)]
    names_a = ["context_utilization", "task_complexity"]

    # Model B: utilization + complexity + relevance
    X_train_b = [[u, c, r] for u, c, r in zip(train_util, train_comp, train_rel)]
    X_test_b = [[u, c, r] for u, c, r in zip(test_util, test_comp, test_rel)]
    names_b = ["context_utilization", "task_complexity", "relevant_context_ratio"]

    # Model C: health_score
    X_train_c = [[h] for h in train_health]
    X_test_c = [[h] for h in test_health]
    names_c = ["health_score"]

    # Model B_shuffled: utilization + complexity + shuffled_relevance
    X_train_b_shuf = [[u, c, sr] for u, c, sr in zip(train_util, train_comp, shuffled_train_rel)]
    X_test_b_shuf = [[u, c, sr] for u, c, sr in zip(test_util, test_comp, shuffled_test_rel)]
    names_b_shuf = ["context_utilization", "task_complexity", "shuffled_relevance"]

    for outcome in OUTCOME_NAMES:
        y_train = [1 if getattr(r, outcome) else 0 for r in train_records]
        y_test = [1 if getattr(r, outcome) else 0 for r in test_records]

        # Train models
        model_a = train_logistic_regression(
            X_train_a, y_train, names_a, learning_rate=learning_rate, iterations=iterations
        )
        model_b = train_logistic_regression(
            X_train_b, y_train, names_b, learning_rate=learning_rate, iterations=iterations
        )
        model_c = train_logistic_regression(
            X_train_c, y_train, names_c, learning_rate=learning_rate, iterations=iterations
        )
        model_b_shuf = train_logistic_regression(
            X_train_b_shuf, y_train, names_b_shuf, learning_rate=learning_rate, iterations=iterations
        )

        # Evaluate on test set
        probs_a = predict_probabilities(model_a, X_test_a)
        probs_b = predict_probabilities(model_b, X_test_b)
        probs_c = predict_probabilities(model_c, X_test_c)
        probs_b_shuf = predict_probabilities(model_b_shuf, X_test_b_shuf)

        metrics_a = evaluate_predictions(y_test, probs_a)
        metrics_b = evaluate_predictions(y_test, probs_b)
        metrics_c = evaluate_predictions(y_test, probs_c)
        metrics_b_shuf = evaluate_predictions(y_test, probs_b_shuf)

        deltas = ModelComparisonDeltas(
            delta_log_loss=round(metrics_b.log_loss - metrics_a.log_loss, 4),
            delta_accuracy=round(metrics_b.accuracy - metrics_a.accuracy, 4),
            delta_precision=round(metrics_b.precision - metrics_a.precision, 4),
            delta_recall=round(metrics_b.recall - metrics_a.recall, 4),
        )

        results[outcome] = OutcomeComparisonResult(
            outcome=outcome,
            model_a=metrics_a,
            model_b=metrics_b,
            model_c=metrics_c,
            model_b_shuffled=metrics_b_shuf,
            deltas_b_minus_a=deltas,
            weights_model_a=dict(zip(names_a, model_a.weights)) | {"bias": model_a.bias},
            weights_model_b=dict(zip(names_b, model_b.weights)) | {"bias": model_b.bias},
            weights_model_c=dict(zip(names_c, model_c.weights)) | {"bias": model_c.bias},
            weights_model_b_shuffled=dict(zip(names_b_shuf, model_b_shuf.weights)) | {"bias": model_b_shuf.bias},
        )

    return results


# ---------------------------------------------------------------------------
# Permutation sanity check for relevance
# ---------------------------------------------------------------------------


def compute_median(values: list[float]) -> float:
    """Calculate the median of a list of floats using standard library."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return round(sorted_vals[mid], 4)
    else:
        return round((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0, 4)


@dataclass(frozen=True)
class PermutationSanityCheckResult:
    """Outcome of relevance permutation sanity check across multiple seeds."""

    outcome: str
    num_permutations: int
    real_relevance_log_loss: float
    permutation_log_losses: list[float]
    min_log_loss: float
    max_log_loss: float
    mean_log_loss: float
    median_log_loss: float


def run_permutation_sanity_check(
    train_records: list[PredictorOutcomeRecord],
    test_records: list[PredictorOutcomeRecord],
    model_b_results: dict[str, OutcomeComparisonResult],
    num_permutations: int = 10,
    learning_rate: float = 0.05,
    iterations: int = 2000,
) -> dict[str, PermutationSanityCheckResult]:
    """Run deterministic permutations of relevance to characterize test log loss variation.

    Preserves marginal distribution of relevance, keeps utilization and complexity
    unchanged, uses the same train/test session split, and applies training-only
    standardization.

    Important:
    This is an experimental sanity check.
    It does NOT compute p-values or claim statistical significance.
    """
    train_util = [r.context_utilization for r in train_records]
    train_comp = [r.task_complexity for r in train_records]
    train_rel = [r.relevant_context_ratio for r in train_records]

    test_util = [r.context_utilization for r in test_records]
    test_comp = [r.task_complexity for r in test_records]
    test_rel = [r.relevant_context_ratio for r in test_records]

    names_perm = ["context_utilization", "task_complexity", "shuffled_relevance"]

    perm_losses: dict[str, list[float]] = {out: [] for out in OUTCOME_NAMES}

    # Run deterministic relevance permutations using seeds 1 .. num_permutations
    for seed_idx in range(1, num_permutations + 1):
        shuffled_train_rel = shuffle_feature_values(train_rel, seed=seed_idx)
        shuffled_test_rel = shuffle_feature_values(test_rel, seed=seed_idx + 1000)

        X_train_perm = [[u, c, sr] for u, c, sr in zip(train_util, train_comp, shuffled_train_rel)]
        X_test_perm = [[u, c, sr] for u, c, sr in zip(test_util, test_comp, shuffled_test_rel)]

        for outcome in OUTCOME_NAMES:
            y_train = [1 if getattr(r, outcome) else 0 for r in train_records]
            y_test = [1 if getattr(r, outcome) else 0 for r in test_records]

            model_perm = train_logistic_regression(
                X_train_perm, y_train, names_perm, learning_rate=learning_rate, iterations=iterations
            )
            probs_perm = predict_probabilities(model_perm, X_test_perm)
            metrics_perm = evaluate_predictions(y_test, probs_perm)
            perm_losses[outcome].append(metrics_perm.log_loss)

    results: dict[str, PermutationSanityCheckResult] = {}
    for outcome in OUTCOME_NAMES:
        real_loss = model_b_results[outcome].model_b.log_loss
        losses = perm_losses[outcome]
        min_loss = min(losses) if losses else 0.0
        max_loss = max(losses) if losses else 0.0
        mean_loss = round(sum(losses) / len(losses), 4) if losses else 0.0
        median_loss = compute_median(losses)

        results[outcome] = PermutationSanityCheckResult(
            outcome=outcome,
            num_permutations=num_permutations,
            real_relevance_log_loss=real_loss,
            permutation_log_losses=losses,
            min_log_loss=min_loss,
            max_log_loss=max_loss,
            mean_log_loss=mean_loss,
            median_log_loss=median_loss,
        )

    return results


# ---------------------------------------------------------------------------
# Complete analysis report structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisReport:
    summary: DatasetSummary
    univariate_buckets: dict[str, dict[str, list[BucketStat]]]
    correlations: CorrelationResults
    train_session_ids: list[str]
    test_session_ids: list[str]
    model_comparisons: dict[str, OutcomeComparisonResult]
    permutation_sanity_check: dict[str, PermutationSanityCheckResult]
    configuration: dict[str, Any]


def run_full_analysis(
    records: list[PredictorOutcomeRecord],
    train_ratio: float = 0.8,
    learning_rate: float = 0.05,
    iterations: int = 2000,
    shuffle_seed: int = 42,
    num_permutations: int = 10,
) -> AnalysisReport:
    """Execute complete end-to-end analysis pipeline."""
    summary = compute_dataset_summary(records)
    buckets = compute_univariate_buckets(records)
    corrs = compute_correlations(records)

    train_rec, test_rec, train_sids, test_sids = split_by_session(records, train_ratio)

    comparisons = run_controlled_comparison(
        train_rec,
        test_rec,
        learning_rate=learning_rate,
        iterations=iterations,
        shuffle_seed=shuffle_seed,
    )

    permutations = run_permutation_sanity_check(
        train_rec,
        test_rec,
        model_b_results=comparisons,
        num_permutations=num_permutations,
        learning_rate=learning_rate,
        iterations=iterations,
    )

    config = {
        "train_ratio": train_ratio,
        "learning_rate": learning_rate,
        "iterations": iterations,
        "shuffle_seed": shuffle_seed,
        "num_permutations": num_permutations,
    }

    return AnalysisReport(
        summary=summary,
        univariate_buckets=buckets,
        correlations=corrs,
        train_session_ids=train_sids,
        test_session_ids=test_sids,
        model_comparisons=comparisons,
        permutation_sanity_check=permutations,
        configuration=config,
    )


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------


def format_terminal_report(report: AnalysisReport) -> str:
    """Format the analysis report into a human-readable terminal string."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("CONTEXT HEALTH — MILESTONE 12B EXPERIMENTAL ANALYSIS")
    lines.append("=" * 60)
    lines.append("")

    # Dataset
    lines.append("Dataset")
    lines.append("-------")
    lines.append(f"Sessions:       {report.summary.total_sessions}")
    lines.append(f"Records:        {report.summary.total_records}")
    lines.append(f"Train sessions: {len(report.train_session_ids)}")
    lines.append(f"Test sessions:  {len(report.test_session_ids)}")
    lines.append("")

    # Outcome rates
    lines.append("Outcome rates")
    lines.append("-------------")
    for name in OUTCOME_NAMES:
        stat = report.summary.outcome_rates[name]
        pct = stat.rate * 100.0
        lines.append(f"{name:<25}: {stat.count:4d} / {stat.total} ({pct:5.1f}%)")
    lines.append("")

    # Univariate correlations
    lines.append("Univariate correlations")
    lines.append("-----------------------")
    lines.append(f"{'Outcome':<26} {'Utilization':>12} {'Relevance':>12} {'Complexity':>12} {'Health':>12}")
    for name in OUTCOME_NAMES:
        c = report.correlations.outcome_correlations[name]
        lines.append(
            f"{name:<26} {c['context_utilization']:12.4f} {c['relevant_context_ratio']:12.4f} "
            f"{c['task_complexity']:12.4f} {c['health_score']:12.4f}"
        )
    lines.append("")

    lines.append("Signal correlations with Health:")
    for sig, r_val in report.correlations.signal_health_correlations.items():
        lines.append(f"  {sig:<25}: {r_val:7.4f}")
    lines.append("")

    # Predictive comparison
    lines.append("Predictive comparison")
    lines.append("---------------------")

    for name in OUTCOME_NAMES:
        res = report.model_comparisons[name]
        lines.append(f"Outcome: {name}")
        lines.append("")

        lines.append("Model A (utilization + complexity)")
        lines.append(
            f"  Log loss:  {res.model_a.log_loss:7.4f}  "
            f"Accuracy:  {res.model_a.accuracy:7.4f}  "
            f"Precision: {res.model_a.precision:7.4f}  "
            f"Recall:    {res.model_a.recall:7.4f}"
        )

        lines.append("Model B (utilization + complexity + relevance)")
        lines.append(
            f"  Log loss:  {res.model_b.log_loss:7.4f}  "
            f"Accuracy:  {res.model_b.accuracy:7.4f}  "
            f"Precision: {res.model_b.precision:7.4f}  "
            f"Recall:    {res.model_b.recall:7.4f}"
        )

        lines.append("Model C (health_score)")
        lines.append(
            f"  Log loss:  {res.model_c.log_loss:7.4f}  "
            f"Accuracy:  {res.model_c.accuracy:7.4f}  "
            f"Precision: {res.model_c.precision:7.4f}  "
            f"Recall:    {res.model_c.recall:7.4f}"
        )

        lines.append("Model B shuffled (utilization + complexity + shuffled relevance)")
        lines.append(
            f"  Log loss:  {res.model_b_shuffled.log_loss:7.4f}  "
            f"Accuracy:  {res.model_b_shuffled.accuracy:7.4f}  "
            f"Precision: {res.model_b_shuffled.precision:7.4f}  "
            f"Recall:    {res.model_b_shuffled.recall:7.4f}"
        )

        d = res.deltas_b_minus_a
        lines.append("Deltas (Model B - Model A):")
        lines.append(
            f"  Δ Log loss: {d.delta_log_loss:+7.4f}  "
            f"Δ Accuracy: {d.delta_accuracy:+7.4f}  "
            f"Δ Precision: {d.delta_precision:+7.4f}  "
            f"Δ Recall: {d.delta_recall:+7.4f}"
        )
        lines.append("")

    # Relevance permutation sanity check
    lines.append("Relevance permutation sanity check")
    lines.append("----------------------------------")
    for name in OUTCOME_NAMES:
        p_res = report.permutation_sanity_check[name]
        lines.append(f"Outcome: {name}")
        lines.append(f"  Real relevance test log loss: {p_res.real_relevance_log_loss:7.4f}")
        losses_formatted = [f"{loss:.4f}" for loss in p_res.permutation_log_losses]
        lines.append(f"  Permutation log losses:       [{', '.join(losses_formatted)}]")
        lines.append(f"  Minimum:                      {p_res.min_log_loss:7.4f}")
        lines.append(f"  Maximum:                      {p_res.max_log_loss:7.4f}")
        lines.append(f"  Mean:                         {p_res.mean_log_loss:7.4f}")
        lines.append(f"  Median:                       {p_res.median_log_loss:7.4f}")
        lines.append("")

    # Interpretation
    lines.append("=" * 60)
    lines.append("INTERPRETATION")
    lines.append("=" * 60)
    lines.append("Descriptive observations on synthetic dataset:")
    for name in OUTCOME_NAMES:
        res = report.model_comparisons[name]
        d_loss = res.deltas_b_minus_a.delta_log_loss
        shuf_loss = res.model_b_shuffled.log_loss - res.model_a.log_loss
        lines.append(
            f"- {name}: adding relevance changed test log loss from "
            f"{res.model_a.log_loss:.4f} to {res.model_b.log_loss:.4f} (delta: {d_loss:+.4f})."
        )
        lines.append(
            f"  shuffling relevance changed test log loss from "
            f"{res.model_a.log_loss:.4f} to {res.model_b_shuffled.log_loss:.4f} (delta: {shuf_loss:+.4f})."
        )
    lines.append("")
    lines.append("Relevance permutation sanity check observations:")
    for name in OUTCOME_NAMES:
        p_res = report.permutation_sanity_check[name]
        lines.append(
            f"- {name}: real log loss is {p_res.real_relevance_log_loss:.4f}; "
            f"across {p_res.num_permutations} permutations, log loss ranged from "
            f"{p_res.min_log_loss:.4f} to {p_res.max_log_loss:.4f} "
            f"(mean: {p_res.mean_log_loss:.4f}, median: {p_res.median_log_loss:.4f})."
        )
    lines.append("")
    lines.append("Important Disclaimers:")
    lines.append("- This is an OFFLINE EXPERIMENTAL ANALYSIS on synthetic data.")
    lines.append("- The permutation sanity check characterizes variation; it does NOT claim statistical significance or report p-values.")
    lines.append("- These measurements do NOT validate Context Health on real sessions.")
    lines.append("- These measurements do NOT establish causality or predictive validity.")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_analysis_json(report: AnalysisReport, output_path: Path) -> None:
    """Save complete analysis report as deterministic JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = asdict(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline predictive analysis for Context Health session replay dataset.",
        epilog=(
            "This is EXPERIMENTAL INFRASTRUCTURE for offline analysis.\n"
            "It does NOT validate the Context Health hypothesis on real sessions."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/session_replay.jsonl",
        help="Input JSONL session replay dataset path (default: data/session_replay.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/session_replay_analysis.json",
        help="Output analysis JSON file path (default: data/session_replay_analysis.json)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of sessions to assign to training set (default: 0.8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Gradient descent learning rate for logistic regression (default: 0.05)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Gradient descent iteration count (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffled-relevance control (default: 42)",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=10,
        help="Number of permutations for relevance permutation sanity check (default: 10)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1. Load and validate dataset
    records = load_dataset(input_path)

    # 2. Run complete analysis
    report = run_full_analysis(
        records=records,
        train_ratio=args.train_ratio,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        shuffle_seed=args.seed,
        num_permutations=args.num_permutations,
    )

    # 3. Print terminal report
    print(format_terminal_report(report))

    # 4. Save results to JSON
    save_analysis_json(report, output_path)
    print(f"\nAnalysis results written to {output_path}")


if __name__ == "__main__":
    main()
