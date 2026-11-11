import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "vision"


def test_report_keeps_final_test_sealed_and_records_selected_model() -> None:
    report = json.loads((ARTIFACT_DIR / "report.json").read_text())

    assert report["sealed_test_used"] is False
    assert report["data"]["test_images_evaluated"] == 0
    assert report["selected_candidate"] == "resnet_patch_top_10_percent"
    assert report["selected_metrics"]["roc_auc"] > 0.98
    assert report["selected_metrics"]["f1"] > 0.92
    assert report["localization"]["pixel_roc_auc_anomalous_images"] > 0.94


def test_artifact_contract_hashes_every_declared_file() -> None:
    contract = json.loads((ARTIFACT_DIR / "artifact_contract.json").read_text())

    assert contract["sealed_test_used"] is False
    for entry in contract["files"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_patch_memory_has_expected_inference_shape() -> None:
    memory = np.load(ARTIFACT_DIR / "models" / "patch_memory.npy")

    assert memory.shape == (5_000, 128)
    assert np.isfinite(memory).all()
