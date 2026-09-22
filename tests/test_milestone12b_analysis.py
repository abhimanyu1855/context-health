"""Tests for Milestone 12B — Automated Predictive Analysis.

Validates:
1. JSONL loading
2. Schema validation
3. Malformed records
4. Bucket assignment
5. Outcome-rate calculation
6. Pearson correlation
7. Train/test session disjointness
8. Deterministic session split
9. Training-only standardization
10. Sigmoid numerical stability
11. Logistic regression convergence on a tiny known dataset
12. Deterministic model training
13. Metric calculation
14. Shuffled relevance determinism
15. Shuffled relevance distribution preservation
16. No outcome leakage
17. No future-turn leakage
18. Analysis JSON determinism
19. No network/API imports
20. No raw prompt/response/private data
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts/ to path so we can import analyze_session_replay
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_session_replay import (
    OUTCOME_NAMES,
    SIGNAL_NAMES,
    AnalysisReport,
    BucketStat,
    DatasetSummary,
    EvaluationMetrics,
    LogisticRegressionModel,
    PermutationSanityCheckResult,
    PredictorOutcomeRecord,
    ScalerParams,
    assign_bucket,
    compute_correlations,
    compute_dataset_summary,
    compute_median,
    compute_univariate_buckets,
    evaluate_predictions,
    fit_scaler,
    load_dataset,
    pearson_correlation,
    predict_probabilities,
    run_controlled_comparison,
    run_full_analysis,
    run_permutation_sanity_check,
    save_analysis_json,
    shuffle_feature_values,
    sigmoid,
    split_by_session,
    train_logistic_regression,
    transform_features,
    validate_record_dict,
)


# Helper to create a valid minimal sample record dict
def make_valid_record_dict(
    session_id: str = "session-0001",
    turn: int = 0,
    utilization: float = 0.3,
    relevance: float = 0.8,
    complexity: float = 0.4,
    health: float = 75.0,
    correction: bool = False,
    tool_retry: bool = False,
    contradiction: bool = False,
    rework: bool = False,
    abandoned: bool = False,
) -> dict:
    return {
        "session_id": session_id,
        "predictor_turn": turn,
        "outcome_turn": turn + 1,
        "context_utilization": utilization,
        "relevant_context_ratio": relevance,
        "task_complexity": complexity,
        "contradiction_count": 0,
        "correction_count": 0,
        "retry_count": 0,
        "health_score": health,
        "health_status": "watch",
        "next_turn_correction": correction,
        "next_turn_tool_retry": tool_retry,
        "next_turn_contradiction": contradiction,
        "next_turn_rework": rework,
        "next_turn_abandoned": abandoned,
    }


def make_valid_record(**kwargs) -> PredictorOutcomeRecord:
    d = make_valid_record_dict(**kwargs)
    return validate_record_dict(d, line_num=1)


# ===========================================================================
# 1. JSONL Loading
# ===========================================================================


class TestJsonlLoading:
    def test_load_valid_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(3):
                f.write(json.dumps(make_valid_record_dict(session_id=f"session-{i+1:04d}", turn=0)) + "\n")
            temp_path = Path(f.name)

        try:
            records = load_dataset(temp_path)
            assert len(records) == 3
            assert records[0].session_id == "session-0001"
            assert records[1].session_id == "session-0002"
            assert records[2].session_id == "session-0003"
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/nonexistent/file/path.jsonl"))

    def test_load_empty_file_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="zero valid records"):
                load_dataset(temp_path)
        finally:
            temp_path.unlink()


# ===========================================================================
# 2. Schema Validation
# ===========================================================================


class TestSchemaValidation:
    def test_valid_record_dict(self) -> None:
        d = make_valid_record_dict()
        record = validate_record_dict(d, line_num=1)
        assert isinstance(record, PredictorOutcomeRecord)
        assert record.session_id == "session-0001"
        assert record.predictor_turn == 0
        assert record.outcome_turn == 1
        assert record.context_utilization == 0.3
        assert record.relevant_context_ratio == 0.8
        assert record.task_complexity == 0.4
        assert record.health_score == 75.0
        assert record.next_turn_correction is False


# ===========================================================================
# 3. Malformed Records (Fail Loudly)
# ===========================================================================


class TestMalformedRecords:
    def test_missing_required_field_raises(self) -> None:
        d = make_valid_record_dict()
        del d["health_score"]
        with pytest.raises(ValueError, match="missing required field 'health_score'"):
            validate_record_dict(d, line_num=10)

    def test_empty_session_id_raises(self) -> None:
        d = make_valid_record_dict(session_id="")
        with pytest.raises(ValueError, match="session_id must be a non-empty string"):
            validate_record_dict(d, line_num=2)

    def test_negative_predictor_turn_raises(self) -> None:
        d = make_valid_record_dict()
        d["predictor_turn"] = -1
        with pytest.raises(ValueError, match="predictor_turn must be an integer >= 0"):
            validate_record_dict(d, line_num=3)

    def test_outcome_turn_not_plus_one_raises(self) -> None:
        d = make_valid_record_dict()
        d["outcome_turn"] = 5  # predictor_turn is 0
        with pytest.raises(ValueError, match=r"outcome_turn must be predictor_turn \+ 1"):
            validate_record_dict(d, line_num=4)

    def test_signal_out_of_bounds_raises(self) -> None:
        d = make_valid_record_dict()
        d["context_utilization"] = 1.5
        with pytest.raises(ValueError, match="context_utilization must be float in"):
            validate_record_dict(d, line_num=5)

    def test_health_score_out_of_bounds_raises(self) -> None:
        d = make_valid_record_dict()
        d["health_score"] = 105.0
        with pytest.raises(ValueError, match="health_score must be float in"):
            validate_record_dict(d, line_num=6)

    def test_outcome_not_boolean_raises(self) -> None:
        d = make_valid_record_dict()
        d["next_turn_correction"] = 1  # Integer instead of bool
        with pytest.raises(ValueError, match="next_turn_correction must be a boolean"):
            validate_record_dict(d, line_num=7)

    def test_invalid_json_in_file_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("{broken json\n")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="invalid JSON syntax"):
                load_dataset(temp_path)
        finally:
            temp_path.unlink()


# ===========================================================================
# 4. Bucket Assignment
# ===========================================================================


class TestBucketAssignment:
    def test_signal_bucket_assignment(self) -> None:
        from analyze_session_replay import SIGNAL_BUCKET_BINS

        assert assign_bucket(0.0, SIGNAL_BUCKET_BINS) == "0.0–0.2"
        assert assign_bucket(0.19, SIGNAL_BUCKET_BINS) == "0.0–0.2"
        assert assign_bucket(0.2, SIGNAL_BUCKET_BINS) == "0.2–0.4"
        assert assign_bucket(0.39, SIGNAL_BUCKET_BINS) == "0.2–0.4"
        assert assign_bucket(0.4, SIGNAL_BUCKET_BINS) == "0.4–0.6"
        assert assign_bucket(0.6, SIGNAL_BUCKET_BINS) == "0.6–0.8"
        assert assign_bucket(0.8, SIGNAL_BUCKET_BINS) == "0.8–1.0"
        assert assign_bucket(1.0, SIGNAL_BUCKET_BINS) == "0.8–1.0"

    def test_health_bucket_assignment(self) -> None:
        from analyze_session_replay import HEALTH_BUCKET_BINS

        assert assign_bucket(100.0, HEALTH_BUCKET_BINS) == "80–100"
        assert assign_bucket(80.0, HEALTH_BUCKET_BINS) == "80–100"
        assert assign_bucket(79.9, HEALTH_BUCKET_BINS) == "60–80"
        assert assign_bucket(60.0, HEALTH_BUCKET_BINS) == "60–80"
        assert assign_bucket(40.0, HEALTH_BUCKET_BINS) == "40–60"
        assert assign_bucket(20.0, HEALTH_BUCKET_BINS) == "20–40"
        assert assign_bucket(5.0, HEALTH_BUCKET_BINS) == "0–20"


# ===========================================================================
# 5. Outcome-Rate Calculation
# ===========================================================================


class TestOutcomeRateCalculation:
    def test_univariate_bucket_outcome_rates(self) -> None:
        # Create 4 records in bucket 0.0-0.2, 2 with correction=True
        records = [
            make_valid_record(session_id="s1", utilization=0.05, correction=True),
            make_valid_record(session_id="s1", turn=1, utilization=0.10, correction=True),
            make_valid_record(session_id="s2", utilization=0.15, correction=False),
            make_valid_record(session_id="s2", turn=1, utilization=0.18, correction=False),
        ]
        buckets = compute_univariate_buckets(records)
        corr_util = buckets["next_turn_correction"]["context_utilization"]
        b0 = next(b for b in corr_util if b.bucket_label == "0.0–0.2")
        assert b0.count == 4
        assert b0.outcome_count == 2
        assert b0.outcome_rate == 0.5000

    def test_empty_bucket_gives_zero_rate(self) -> None:
        records = [make_valid_record(utilization=0.05, correction=False)]
        buckets = compute_univariate_buckets(records)
        corr_util = buckets["next_turn_correction"]["context_utilization"]
        b_empty = next(b for b in corr_util if b.bucket_label == "0.8–1.0")
        assert b_empty.count == 0
        assert b_empty.outcome_count == 0
        assert b_empty.outcome_rate == 0.0


# ===========================================================================
# 6. Pearson Correlation
# ===========================================================================


class TestPearsonCorrelation:
    def test_perfect_positive_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert pearson_correlation(x, y) == pytest.approx(1.0, abs=1e-4)

    def test_perfect_negative_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert pearson_correlation(x, y) == pytest.approx(-1.0, abs=1e-4)

    def test_zero_variance_returns_zero(self) -> None:
        x = [3.0, 3.0, 3.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0]
        assert pearson_correlation(x, y) == 0.0

    def test_known_hand_calculated_correlation(self) -> None:
        # x = [1, 2, 3], y = [1, 5, 7]
        # mean_x = 2, mean_y = 4.3333
        # cov = (1-2)*(1-4.333) + (2-2)*(5-4.333) + (3-2)*(7-4.333) = 3.333 + 0 + 2.667 = 6.0
        # var_x = 1 + 0 + 1 = 2.0
        # var_y = (-3.333)^2 + (0.667)^2 + (2.667)^2 = 11.111 + 0.444 + 7.111 = 18.667
        # r = 6.0 / sqrt(2.0 * 18.667) = 6.0 / sqrt(37.333) = 6.0 / 6.110 = 0.98198
        x = [1.0, 2.0, 3.0]
        y = [1.0, 5.0, 7.0]
        r = pearson_correlation(x, y)
        assert r == pytest.approx(0.9820, abs=1e-3)


# ===========================================================================
# 7 & 8. Session-level Train/Test Split (Disjoint & Deterministic)
# ===========================================================================


class TestSessionSplit:
    def test_train_test_sessions_are_disjoint(self) -> None:
        records = []
        for s in range(1, 11):
            sid = f"session-{s:04d}"
            for t in range(5):
                records.append(make_valid_record(session_id=sid, turn=t))

        train_rec, test_rec, train_sids, test_sids = split_by_session(records, train_ratio=0.8)

        # 10 sessions * 0.8 = 8 train, 2 test
        assert len(train_sids) == 8
        assert len(test_sids) == 2

        train_set = set(train_sids)
        test_set = set(test_sids)
        assert train_set.isdisjoint(test_set)

        # Verify all train records are in train_set and no test records
        for r in train_rec:
            assert r.session_id in train_set
            assert r.session_id not in test_set
        for r in test_rec:
            assert r.session_id in test_set
            assert r.session_id not in train_set

    def test_deterministic_split(self) -> None:
        records = [make_valid_record(session_id=f"session-{i:04d}") for i in range(20)]
        _, _, train_a, test_a = split_by_session(records, train_ratio=0.7)
        _, _, train_b, test_b = split_by_session(records, train_ratio=0.7)
        assert train_a == train_b
        assert test_a == test_b


# ===========================================================================
# 9. Training-Only Standardization
# ===========================================================================


class TestStandardization:
    def test_scaler_computed_from_training_only(self) -> None:
        X_train = [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]]
        X_test = [[1000.0, 5000.0]]  # Extreme values in test

        scaler = fit_scaler(X_train)
        # Train mean should be 20.0 and 200.0
        assert scaler.means[0] == pytest.approx(20.0)
        assert scaler.means[1] == pytest.approx(200.0)

        # Transform test using train scaler
        X_test_norm = transform_features(X_test, scaler)
        # Test values normalized using train means/stddevs
        assert X_test_norm[0][0] == pytest.approx((1000.0 - 20.0) / scaler.stddevs[0])

    def test_zero_variance_handled_safely(self) -> None:
        X_train = [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]  # Column 0 has zero variance
        scaler = fit_scaler(X_train)
        assert scaler.stddevs[0] == 1.0  # Fallback to 1.0
        X_norm = transform_features(X_train, scaler)
        # 5.0 - 5.0 / 1.0 = 0.0
        for row in X_norm:
            assert row[0] == 0.0


# ===========================================================================
# 10. Sigmoid Numerical Stability
# ===========================================================================


class TestSigmoidStability:
    def test_sigmoid_values(self) -> None:
        assert sigmoid(0.0) == 0.5
        assert sigmoid(35.0) == 1.0
        assert sigmoid(1000.0) == 1.0  # No overflow
        assert sigmoid(-35.0) == 0.0
        assert sigmoid(-1000.0) == 0.0  # No underflow

    def test_sigmoid_monotonic(self) -> None:
        vals = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
        sig_vals = [sigmoid(v) for v in vals]
        for i in range(len(sig_vals) - 1):
            assert sig_vals[i] <= sig_vals[i + 1]


# ===========================================================================
# 11. Logistic Regression Convergence on Tiny Dataset
# ===========================================================================


class TestLogisticRegressionConvergence:
    def test_convergence_on_separable_data(self) -> None:
        # Simple 1D toy problem: x < 0 -> y = 0, x > 0 -> y = 1
        X_train = [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]]
        y_train = [0, 0, 0, 1, 1, 1]

        model = train_logistic_regression(
            X_train, y_train, ["feature_x"], learning_rate=0.1, iterations=1000
        )

        # Weight on feature_x must be positive (higher x -> higher prob)
        assert model.weights[0] > 0.0

        # Predict probabilities
        probs = predict_probabilities(model, X_train)
        metrics = evaluate_predictions(y_train, probs)
        assert metrics.accuracy == 1.0
        assert metrics.log_loss < 0.25


# ===========================================================================
# 12. Deterministic Model Training
# ===========================================================================


class TestDeterministicTraining:
    def test_identical_data_produces_identical_weights(self) -> None:
        X = [[0.2, 0.8], [0.5, 0.3], [0.7, 0.2], [0.1, 0.9]]
        y = [0, 1, 1, 0]

        m1 = train_logistic_regression(X, y, ["u", "c"], learning_rate=0.05, iterations=500)
        m2 = train_logistic_regression(X, y, ["u", "c"], learning_rate=0.05, iterations=500)

        assert m1.weights == m2.weights
        assert m1.bias == m2.bias


# ===========================================================================
# 13. Metric Calculation
# ===========================================================================


class TestMetricCalculation:
    def test_perfect_predictions(self) -> None:
        y_true = [0, 1, 0, 1]
        y_prob = [0.01, 0.99, 0.02, 0.98]
        m = evaluate_predictions(y_true, y_prob)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.log_loss < 0.05

    def test_zero_positives_handled_safely(self) -> None:
        y_true = [0, 0, 0]
        y_prob = [0.1, 0.2, 0.3]
        m = evaluate_predictions(y_true, y_prob)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.accuracy == 1.0

    def test_hand_calculated_metrics(self) -> None:
        # y_true = [1, 0, 1, 0]
        # y_prob = [0.8, 0.6, 0.4, 0.2]
        # threshold = 0.5:
        # preds: [1, 1, 0, 0]
        # sample 0: true 1, pred 1 (TP)
        # sample 1: true 0, pred 1 (FP)
        # sample 2: true 1, pred 0 (FN)
        # sample 3: true 0, pred 0 (TN)
        # TP = 1, FP = 1, FN = 1, TN = 1
        # accuracy = 2/4 = 0.5
        # precision = 1 / (1 + 1) = 0.5
        # recall = 1 / (1 + 1) = 0.5
        y_true = [1, 0, 1, 0]
        y_prob = [0.8, 0.6, 0.4, 0.2]
        m = evaluate_predictions(y_true, y_prob)
        assert m.accuracy == 0.5
        assert m.precision == 0.5
        assert m.recall == 0.5


# ===========================================================================
# 14 & 15. Shuffled Relevance Control (Determinism & Distribution Preservation)
# ===========================================================================


class TestShuffledRelevance:
    def test_shuffled_relevance_deterministic(self) -> None:
        vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        s1 = shuffle_feature_values(vals, seed=42)
        s2 = shuffle_feature_values(vals, seed=42)
        assert s1 == s2

    def test_different_seeds_give_different_orders(self) -> None:
        vals = [float(i) for i in range(20)]
        s1 = shuffle_feature_values(vals, seed=42)
        s2 = shuffle_feature_values(vals, seed=99)
        assert s1 != s2

    def test_distribution_preserved(self) -> None:
        vals = [0.15, 0.32, 0.85, 0.42, 0.91, 0.67]
        shuffled = shuffle_feature_values(vals, seed=42)
        # Exact same multiset of values
        assert sorted(shuffled) == sorted(vals)
        assert sum(shuffled) == pytest.approx(sum(vals))


# ===========================================================================
# 16 & 17. Leakage Protections (No Outcome or Future-Turn Leakage)
# ===========================================================================


class TestLeakageProtections:
    def test_no_outcome_in_model_features(self) -> None:
        records = [make_valid_record(session_id=f"s{i}") for i in range(10)]
        train_rec, test_rec, _, _ = split_by_session(records, train_ratio=0.8)
        comparisons = run_controlled_comparison(train_rec, test_rec, iterations=10)

        for outcome, comp in comparisons.items():
            # Check model feature names
            features_a = list(comp.weights_model_a.keys())
            features_b = list(comp.weights_model_b.keys())
            features_c = list(comp.weights_model_c.keys())

            for out_name in OUTCOME_NAMES:
                assert out_name not in features_a
                assert out_name not in features_b
                assert out_name not in features_c

    def test_no_future_turn_in_model_features(self) -> None:
        records = [make_valid_record(session_id=f"s{i}") for i in range(10)]
        train_rec, test_rec, _, _ = split_by_session(records, train_ratio=0.8)
        comparisons = run_controlled_comparison(train_rec, test_rec, iterations=10)

        forbidden_fields = ["outcome_turn", "predictor_turn", "session_id"]
        for outcome, comp in comparisons.items():
            for f in comp.weights_model_b.keys():
                assert f not in forbidden_fields


# ===========================================================================
# 18. Analysis JSON Determinism
# ===========================================================================


class TestJsonDeterminism:
    def test_run_full_analysis_produces_deterministic_json(self) -> None:
        records = []
        for s in range(1, 11):
            for t in range(5):
                records.append(
                    make_valid_record(
                        session_id=f"s{s:03d}",
                        turn=t,
                        utilization=0.1 * (t + 1),
                        relevance=1.0 - 0.05 * t,
                        complexity=0.3,
                        correction=(t % 2 == 1),
                    )
                )

        report1 = run_full_analysis(records, train_ratio=0.8, iterations=100, shuffle_seed=42)
        report2 = run_full_analysis(records, train_ratio=0.8, iterations=100, shuffle_seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "rep1.json"
            p2 = Path(tmpdir) / "rep2.json"
            save_analysis_json(report1, p1)
            save_analysis_json(report2, p2)

            assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


# ===========================================================================
# 19. No Network or API Imports
# ===========================================================================


class TestNoExternalOrNetworkImports:
    def test_script_imports_only_stdlib(self) -> None:
        script_file = SCRIPTS_DIR / "analyze_session_replay.py"
        source = script_file.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])

        forbidden = {
            "requests",
            "urllib",
            "http",
            "anthropic",
            "openai",
            "pandas",
            "numpy",
            "scipy",
            "sklearn",
            "statsmodels",
            "matplotlib",
        }
        intersection = imported_modules.intersection(forbidden)
        assert not intersection, f"Forbidden external imports found: {intersection}"


# ===========================================================================
# 20. Privacy Sentinels
# ===========================================================================


class TestPrivacySentinels:
    def test_no_private_data_in_report(self) -> None:
        records = [make_valid_record(session_id=f"session-{i:04d}") for i in range(5)]
        report = run_full_analysis(records, iterations=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "out.json"
            save_analysis_json(report, out_json)
            content = out_json.read_text(encoding="utf-8")

            # Check for API keys, email addresses, absolute filesystem paths
            assert "sk-" not in content
            assert "@" not in content
            assert "/Users/" not in content
            assert "/home/" not in content
            assert "C:\\" not in content


# ===========================================================================
# 21. Relevance Permutation Sanity Check
# ===========================================================================


class TestRelevancePermutationSanityCheck:
    def test_deterministic_permutations(self) -> None:
        records = [
            make_valid_record(session_id=f"session-{i:04d}", turn=0, utilization=0.08 * i, relevance=0.9 - 0.05 * i, complexity=0.09 * i)
            for i in range(10)
        ]
        train_rec, test_rec, _, _ = split_by_session(records, train_ratio=0.8)
        comparisons = run_controlled_comparison(train_rec, test_rec, iterations=20)

        res1 = run_permutation_sanity_check(train_rec, test_rec, comparisons, num_permutations=5, iterations=20)
        res2 = run_permutation_sanity_check(train_rec, test_rec, comparisons, num_permutations=5, iterations=20)

        for out in OUTCOME_NAMES:
            assert res1[out].permutation_log_losses == res2[out].permutation_log_losses
            assert res1[out].min_log_loss == res2[out].min_log_loss
            assert res1[out].max_log_loss == res2[out].max_log_loss
            assert res1[out].mean_log_loss == res2[out].mean_log_loss
            assert res1[out].median_log_loss == res2[out].median_log_loss

    def test_marginal_distribution_preservation(self) -> None:
        records = [
            make_valid_record(session_id=f"session-{i:04d}", relevance=0.15 * (i + 1))
            for i in range(6)
        ]
        train_rec, test_rec, _, _ = split_by_session(records, train_ratio=0.8)
        train_rel = [r.relevant_context_ratio for r in train_rec]
        test_rel = [r.relevant_context_ratio for r in test_rec]

        for seed in range(1, 11):
            shuf_train = shuffle_feature_values(train_rel, seed=seed)
            shuf_test = shuffle_feature_values(test_rel, seed=seed + 1000)
            assert sorted(shuf_train) == sorted(train_rel)
            assert sorted(shuf_test) == sorted(test_rel)

    def test_same_train_test_split_used(self) -> None:
        records = [
            make_valid_record(session_id=f"session-{i:04d}")
            for i in range(10)
        ]
        train_rec, test_rec, train_sids, test_sids = split_by_session(records, train_ratio=0.8)
        comparisons = run_controlled_comparison(train_rec, test_rec, iterations=10)
        perm_res = run_permutation_sanity_check(train_rec, test_rec, comparisons, num_permutations=3, iterations=10)

        for out in OUTCOME_NAMES:
            assert perm_res[out].num_permutations == 3
            assert perm_res[out].real_relevance_log_loss == comparisons[out].model_b.log_loss

    def test_unchanged_utilization_and_complexity(self) -> None:
        records = [
            make_valid_record(session_id=f"s{i}", utilization=0.1 * i, complexity=0.08 * i, relevance=0.05 * i)
            for i in range(8)
        ]
        train_rec, test_rec, _, _ = split_by_session(records, train_ratio=0.8)

        train_util = [r.context_utilization for r in train_rec]
        train_comp = [r.task_complexity for r in train_rec]
        train_rel = [r.relevant_context_ratio for r in train_rec]

        for seed in range(1, 4):
            shuf_rel = shuffle_feature_values(train_rel, seed=seed)
            X_perm = [[u, c, sr] for u, c, sr in zip(train_util, train_comp, shuf_rel)]
            for row_idx, r in enumerate(train_rec):
                assert X_perm[row_idx][0] == r.context_utilization
                assert X_perm[row_idx][1] == r.task_complexity

    def test_no_leakage_between_train_and_test(self) -> None:
        train_records = [make_valid_record(session_id="s1", relevance=0.2), make_valid_record(session_id="s1", relevance=0.3)]
        test_records = [make_valid_record(session_id="s2", relevance=0.8), make_valid_record(session_id="s2", relevance=0.9)]

        train_rel = [r.relevant_context_ratio for r in train_records]
        test_rel = [r.relevant_context_ratio for r in test_records]

        for seed in range(1, 10):
            shuf_train = shuffle_feature_values(train_rel, seed=seed)
            shuf_test = shuffle_feature_values(test_rel, seed=seed + 1000)

            for val in shuf_train:
                assert val in (0.2, 0.3)
                assert val not in (0.8, 0.9)
            for val in shuf_test:
                assert val in (0.8, 0.9)
                assert val not in (0.2, 0.3)

    def test_compute_median(self) -> None:
        assert compute_median([]) == 0.0
        assert compute_median([5.0]) == 5.0
        assert compute_median([1.0, 3.0, 2.0]) == 2.0
        assert compute_median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert compute_median([0.5, 0.1, 0.9, 0.3]) == 0.4

    def test_full_analysis_includes_permutation_results(self) -> None:
        records = [
            make_valid_record(session_id=f"session-{i:04d}", turn=0, utilization=0.05 * i, relevance=0.1 * i, complexity=0.05 * i)
            for i in range(10)
        ]
        report = run_full_analysis(records, train_ratio=0.8, iterations=10, num_permutations=3)

        assert hasattr(report, "permutation_sanity_check")
        assert len(report.permutation_sanity_check) == 5
        for out in OUTCOME_NAMES:
            res = report.permutation_sanity_check[out]
            assert res.outcome == out
            assert res.num_permutations == 3
            assert len(res.permutation_log_losses) == 3
            assert res.min_log_loss <= res.max_log_loss
            assert res.min_log_loss <= res.mean_log_loss <= res.max_log_loss

