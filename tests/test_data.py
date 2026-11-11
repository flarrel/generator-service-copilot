import json
from pathlib import Path

import pandas as pd
import pytest

from generator_service_copilot.data import (
    INTENTS,
    generate_service_cases,
    image_manifest,
    load_document_chunks,
    sha256_file,
    validate_service_cases,
)
from generator_service_copilot.download import MVTEC_REPOSITORY, MVTEC_REVISION

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_service_cases_are_balanced_and_group_isolated() -> None:
    cases = generate_service_cases()

    assert len(cases) == 320
    assert cases["group_id"].nunique() == 80
    assert set(cases["intent"]) == set(INTENTS)
    assert cases.groupby("group_id")["split"].nunique().max() == 1
    assert cases.groupby(["split", "intent"]).size().unstack().to_dict() == {
        intent: {"development": 8, "test": 8, "train": 24} for intent in INTENTS
    }


def test_validation_rejects_group_leakage() -> None:
    cases = generate_service_cases()
    group_rows = cases.index[cases["group_id"] == cases.loc[0, "group_id"]]
    cases.loc[group_rows[0], "split"] = "development"

    with pytest.raises(ValueError, match="group crosses"):
        validate_service_cases(cases)


def test_manual_chunks_have_complete_lineage() -> None:
    chunks = load_document_chunks(PROJECT_ROOT / "data" / "source" / "manuals")

    assert chunks["document_id"].nunique() == 8
    assert chunks["chunk_id"].is_unique
    assert chunks["synthetic"].all()
    assert chunks.groupby("document_id")["source_version"].nunique().max() == 1


def test_empty_image_manifest_has_stable_schema(tmp_path: Path) -> None:
    manifest = image_manifest(tmp_path)

    assert manifest.empty
    assert list(manifest.columns) == [
        "path",
        "dataset_split",
        "project_split",
        "defect_type",
        "is_anomaly",
        "mask_path",
        "sha256",
        "bytes",
    ]


def test_local_mvtec_snapshot_is_complete() -> None:
    root = PROJECT_ROOT / "data" / "external" / "mvtec_ad"
    if not (root / "cable").exists():
        pytest.skip("Licensed MVTec data is not stored in Git")

    manifest = image_manifest(root)
    anomalies = manifest.loc[manifest["is_anomaly"]]
    assert len(manifest) == 374
    assert len(anomalies) == 92
    assert (anomalies["mask_path"] != "").all()
    assert manifest["sha256"].is_unique
    assert manifest["project_split"].value_counts().to_dict() == {
        "train": 179,
        "development": 121,
        "test": 74,
    }
    assert set(manifest.loc[manifest["project_split"] == "test", "defect_type"]) == set(
        manifest.loc[manifest["dataset_split"] == "test", "defect_type"]
    )


def test_case_csv_round_trip_preserves_schema(tmp_path: Path) -> None:
    cases = generate_service_cases()
    path = tmp_path / "cases.csv"
    cases.to_csv(path, index=False)

    restored = pd.read_csv(path)
    validated = validate_service_cases(restored)
    assert len(validated) == len(cases)


def test_tracked_data_manifest_matches_files() -> None:
    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for record in manifest["files"]:
        path = PROJECT_ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]


def test_source_metadata_matches_pinned_downloader() -> None:
    sources = json.loads((PROJECT_ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    mvtec = sources["mvtec_ad_cable"]

    assert mvtec["download_repository"] == MVTEC_REPOSITORY
    assert mvtec["revision"] == MVTEC_REVISION
    assert mvtec["raw_data_committed"] is False
