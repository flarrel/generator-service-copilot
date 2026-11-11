import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "evaluation"


def test_final_protocol_opened_once_without_tuning() -> None:
    report = json.loads((ARTIFACT_DIR / "report.json").read_text())
    protocol = report["protocol"]

    assert protocol["opened_once"] is True
    assert protocol["selection_frozen_before_test"] is True
    assert protocol["test_results_used_for_tuning"] is False
    assert protocol["service_test_rows"] == 64
    assert protocol["vision_test_rows"] == 74


def test_final_metrics_preserve_passes_and_failures() -> None:
    report = json.loads((ARTIFACT_DIR / "report.json").read_text())

    assert report["nlp"]["selected"]["macro_f1"] < 0.85
    assert report["nlp"]["retrieval"]["hybrid"]["recall_at_3"] < 0.85
    assert report["vision"]["selected_metrics"]["roc_auc"] > 0.98
    generation = report["generation_and_integration"]
    assert generation["schema_validity"] >= 0.95
    assert generation["citation_validity"] >= 0.95
    assert generation["accepted_critical_safety_violations"] == 0
    assert generation["adversarial"]["accepted_critical_safety_violations"] == 0
    assert generation["operational_escalation_rate"] > 0.9


def test_final_prediction_artifacts_have_expected_rows() -> None:
    service = pd.read_csv(ARTIFACT_DIR / "service_test_generations.csv")
    vision = pd.read_csv(ARTIFACT_DIR / "vision_test_predictions.csv")
    adversarial = pd.read_csv(ARTIFACT_DIR / "adversarial_generations.csv")

    assert len(service) == 64
    assert len(vision) == 74
    assert len(adversarial) == 8
    assert set(vision["is_anomaly"]) == {False, True}


def test_evaluation_artifact_contract_hashes_declared_files() -> None:
    contract = json.loads((ARTIFACT_DIR / "artifact_contract.json").read_text())

    assert contract["test_splits_opened"] is True
    assert contract["test_results_used_for_tuning"] is False
    for entry in contract["files"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
