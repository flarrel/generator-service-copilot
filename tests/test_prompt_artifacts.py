import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "prompts"


def test_prompt_report_keeps_test_split_sealed() -> None:
    report = json.loads((ARTIFACT_DIR / "report.json").read_text())

    assert report["sealed_test_used"] is False
    assert report["test_cases_evaluated"] == 0
    assert report["evaluation"]["cases"] == 8
    assert report["evaluation"]["repeats_per_case"] == 3
    assert report["live_openai_call_verified"] is False


def test_selected_prompt_passes_predeclared_gate() -> None:
    report = json.loads((ARTIFACT_DIR / "report.json").read_text())
    selected = next(
        item for item in report["variants"] if item["variant_id"] == report["selected_variant"]
    )

    assert report["selection_status"] == "selected"
    assert report["selected_variant"] == "v3_few_shot"
    assert selected["passes_safety_gate"] is True
    assert selected["schema_validity"] >= 0.95
    assert selected["citation_validity"] >= 0.95
    assert selected["critical_safety_violations"] == 0


def test_generation_artifact_contains_fixed_repeated_experiment() -> None:
    generations = pd.read_csv(ARTIFACT_DIR / "generations.csv")

    assert len(generations) == 96
    assert generations["variant_id"].nunique() == 4
    assert generations["case_id"].nunique() == 8
    assert set(generations["repeat"]) == {1, 2, 3}


def test_prompt_artifact_contract_hashes_declared_files() -> None:
    contract = json.loads((ARTIFACT_DIR / "artifact_contract.json").read_text())

    assert contract["sealed_test_used"] is False
    for entry in contract["files"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
