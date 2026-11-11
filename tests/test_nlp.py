import numpy as np
import pandas as pd
import pytest

from generator_service_copilot.nlp import (
    abstention_table,
    classifier_metrics,
    rank_metrics,
    semantic_leakage_audit,
)


def test_classifier_metrics_reward_correct_confident_predictions() -> None:
    classes = np.array(["a", "b"])
    labels = np.array(["a", "b", "a", "b"])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])

    metrics = classifier_metrics(labels, probabilities, classes)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["expected_calibration_error"] < 0.25


def test_abstention_table_reports_coverage() -> None:
    classes = np.array(["a", "b"])
    labels = np.array(["a", "b"])
    probabilities = np.array([[0.95, 0.05], [0.4, 0.6]])

    table = abstention_table(labels, probabilities, classes)
    assert table[0]["coverage"] == 1.0
    assert table[-1]["coverage"] == 0.5
    assert table[-1]["accepted_accuracy"] == 1.0


def test_rank_metrics_use_first_relevant_rank() -> None:
    rankings = [["a", "b", "c"], ["c", "b", "a"]]
    metrics = rank_metrics(rankings, ["a", "b"])

    assert metrics == {"recall_at_1": 0.5, "recall_at_3": 1.0, "mrr": 0.75}


def test_semantic_leakage_audit_flags_near_duplicates() -> None:
    train = pd.DataFrame(
        {
            "case_id": ["train-1", "train-2"],
            "text": ["the cable insulation is visibly damaged", "replace the filter"],
            "intent": ["electrical", "consumables"],
        }
    )
    development = pd.DataFrame(
        {
            "case_id": ["dev-1"],
            "text": ["the cable insulation is visibly damaged"],
            "intent": ["electrical"],
        }
    )

    report = semantic_leakage_audit(train, development)
    assert report["maximum_similarity"] == pytest.approx(1.0)
    assert report["pairs_at_or_above_0_90"] == 1
