from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from redis import Redis

from generator_service_copilot.cache import ResponseCache
from generator_service_copilot.data import sha256_file
from generator_service_copilot.service import CopilotService

RESTART_MARKER = "gsc:verification:restart-marker"


def prepare_restart(redis_url: str) -> None:
    client = Redis.from_url(redis_url, decode_responses=False)
    client.set(RESTART_MARKER, "must-disappear")
    if client.get(RESTART_MARKER) is None:
        raise RuntimeError("Could not prepare Redis restart marker")


def verify_after_restart(project_root: Path, redis_url: str) -> dict:
    client = Redis.from_url(redis_url, decode_responses=False)
    client.ping()
    restart_cleared_cache = client.get(RESTART_MARKER) is None
    cache = ResponseCache(client, corpus_version="verification-a", ttl_seconds=1)
    payload = {"result": "cached"}
    initial = cache.get("question", "local").status
    cache.set("question", "local", payload)
    key = cache.key("question", "local")
    hit = cache.get("question", "local")
    ttl_after_write = client.ttl(key)
    time.sleep(1.2)
    expired = cache.get("question", "local").status == "miss"
    cache.set("question", "local", payload)
    invalidated = cache.invalidate_current_version()
    version_cache = ResponseCache(client, corpus_version="verification-b", ttl_seconds=10)
    version_isolated = version_cache.get("question", "local").status == "miss"
    report = {
        "redis_version": client.info("server")["redis_version"],
        "cold_cache_initial_miss": initial == "miss",
        "cache_hit": hit.status == "hit" and hit.value == payload,
        "ttl_after_write_seconds": ttl_after_write,
        "expiry_verified": expired,
        "current_version_invalidation_count": invalidated,
        "document_version_isolation": version_isolated,
        "restart_cleared_ephemeral_cache": restart_cleared_cache,
        "all_checks_passed": all(
            [
                initial == "miss",
                hit.status == "hit" and hit.value == payload,
                ttl_after_write == 1,
                expired,
                invalidated == 1,
                version_isolated,
                restart_cleared_cache,
            ]
        ),
    }
    artifact_dir = project_root / "artifacts" / "app"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "redis_cache_verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_workflow(project_root: Path, redis_url: str) -> dict:
    client = Redis.from_url(redis_url, decode_responses=False)
    service = CopilotService(project_root, redis_client=client)
    service.cache.invalidate_current_version()
    question = "The cable insulation appears damaged. What documented service path applies?"
    first = service.ask(question)
    second = service.ask(question)
    manifest = pd.read_csv(project_root / "data" / "processed" / "mvtec_cable_manifest.csv")
    development = manifest.loc[manifest["project_split"] == "development"].iloc[0]
    vision = service.inspect_image(
        project_root / "data" / "external" / "mvtec_ad" / development["path"]
    )
    report = {
        "sealed_service_test_used": False,
        "sealed_vision_test_used": False,
        "question_flow": {
            "retrieval_mode": first["retrieval_mode"],
            "provider_status": first["provider_status"],
            "citations_valid": first["citations_valid"],
            "first_cache_status": first["cache_status"],
            "second_cache_status": second["cache_status"],
            "escalate": first["response"]["escalate"],
        },
        "vision_flow": {
            "source_split": development["project_split"],
            "score": vision["score"],
            "threshold": vision["threshold"],
            "map_shape": list(vision["anomaly_map"].shape),
        },
        "health": service.health(),
    }
    report["all_checks_passed"] = bool(
        first["retrieval_mode"] == "redis_vector"
        and first["citations_valid"]
        and first["cache_status"] == "miss"
        and second["cache_status"] == "hit"
        and development["project_split"] == "development"
        and vision["anomaly_map"].shape == (128, 128)
        and not report["health"]["final_test_used"]
    )
    artifact_dir = project_root / "artifacts" / "app"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = artifact_dir / "workflow_verification.json"
    workflow_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cache_path = artifact_dir / "redis_cache_verification.json"
    paths = [cache_path, workflow_path]
    contract = {
        "schema_version": 1,
        "sealed_service_test_used": False,
        "sealed_vision_test_used": False,
        "files": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }
    (artifact_dir / "artifact_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify application Redis cache behavior")
    parser.add_argument("phase", choices=["prepare-restart", "verify", "workflow"])
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--redis-url", default="redis://localhost:6380/0")
    args = parser.parse_args()
    if args.phase == "prepare-restart":
        prepare_restart(args.redis_url)
        print(json.dumps({"restart_marker_prepared": True}))
        return
    if args.phase == "workflow":
        print(json.dumps(verify_workflow(args.project_root.resolve(), args.redis_url), indent=2))
        return
    report = verify_after_restart(args.project_root.resolve(), args.redis_url)
    report["artifact_sha256"] = sha256_file(
        args.project_root / "artifacts" / "app" / "redis_cache_verification.json"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
