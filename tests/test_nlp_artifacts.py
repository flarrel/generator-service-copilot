import json
from pathlib import Path

import joblib
import pandas as pd

from generator_service_copilot.data import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "nlp"


def test_report_keeps_sealed_test_unused() -> None:
    report = json.loads((ARTIFACT_DIR / "report.json").read_text(encoding="utf-8"))

    assert report["sealed_test_used"] is False
    assert report["data"]["test_rows_evaluated"] == 0
    assert report["selected_intent_model"] == "embedding_mlp"
    assert report["intent_candidates"]["embedding_mlp"]["macro_f1"] >= 0.85
    assert report["retrieval"]["selected"] == "hybrid"
    assert report["retrieval"]["candidates"]["hybrid"]["recall_at_3"] >= 0.85


def test_artifact_contract_matches_files() -> None:
    contract = json.loads((ARTIFACT_DIR / "artifact_contract.json").read_text(encoding="utf-8"))

    assert contract["sealed_test_used"] is False
    for record in contract["files"]:
        path = PROJECT_ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]


def test_selected_model_matches_recorded_development_predictions() -> None:
    model = joblib.load(ARTIFACT_DIR / "models" / "intent_embedding_mlp.joblib")
    predictions = pd.read_csv(ARTIFACT_DIR / "intent_development_predictions.csv")

    assert list(model.classes_) == sorted(predictions["intent"].unique())
    assert predictions["correct"].mean() >= 0.85


def test_redis_verification_passed() -> None:
    verification = json.loads(
        (ARTIFACT_DIR / "redis_verification.json").read_text(encoding="utf-8")
    )

    assert verification["redis_version"] == "8.2.3"
    assert verification["indexed_chunks"] == 24
    assert verification["embedding_dimensions"] == 384
    assert verification["all_queries_passed"] is True
