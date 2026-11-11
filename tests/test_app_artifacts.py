import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "app"


def test_live_cache_verification_passed() -> None:
    report = json.loads((ARTIFACT_DIR / "redis_cache_verification.json").read_text())

    assert report["redis_version"] == "8.2.3"
    assert report["cache_hit"] is True
    assert report["expiry_verified"] is True
    assert report["document_version_isolation"] is True
    assert report["restart_cleared_ephemeral_cache"] is True
    assert report["all_checks_passed"] is True


def test_live_workflow_used_redis_cache_and_no_final_test() -> None:
    report = json.loads((ARTIFACT_DIR / "workflow_verification.json").read_text())

    assert report["sealed_service_test_used"] is False
    assert report["sealed_vision_test_used"] is False
    assert report["question_flow"]["retrieval_mode"] == "redis_vector"
    assert report["question_flow"]["provider_status"] == "ok"
    assert report["question_flow"]["citations_valid"] is True
    assert report["question_flow"]["first_cache_status"] == "miss"
    assert report["question_flow"]["second_cache_status"] == "hit"
    assert report["vision_flow"]["source_split"] == "development"
    assert report["all_checks_passed"] is True


def test_app_artifact_contract_hashes_declared_files() -> None:
    contract = json.loads((ARTIFACT_DIR / "artifact_contract.json").read_text())

    assert contract["sealed_service_test_used"] is False
    assert contract["sealed_vision_test_used"] is False
    for entry in contract["files"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
