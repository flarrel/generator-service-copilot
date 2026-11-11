import json
from pathlib import Path

import pandas as pd

from generator_service_copilot.prompts import GenerationResult
from generator_service_copilot.service import CopilotService

ROOT = Path(__file__).resolve().parents[1]


class GroundedProvider:
    def generate(self, variant, request, seed):
        del variant, seed
        return GenerationResult(
            text=json.dumps(
                {
                    "summary": "Follow the cited approved guidance.",
                    "actions": [request.evidence[0]["text"]],
                    "citations": [request.evidence[0]["chunk_id"]],
                    "escalate": request.safety_critical,
                    "confidence": "high",
                }
            ),
            latency_seconds=0.01,
            provider="test",
        )


class FailingProvider:
    def generate(self, variant, request, seed):
        del variant, request, seed
        raise RuntimeError("provider unavailable")


def test_question_flow_routes_retrieves_and_returns_grounded_schema() -> None:
    service = CopilotService(ROOT, generation_provider=GroundedProvider())

    result = service.ask("The cable insulation looks damaged. What should be reviewed?")

    assert result["retrieval_mode"] == "local_hybrid_fallback"
    assert len(result["evidence"]) == 3
    assert result["citations_valid"] is True
    assert result["provider_status"] == "ok"
    assert result["response"]["citations"][0] in {item["chunk_id"] for item in result["evidence"]}
    assert result["cache_status"] == "unavailable"


def test_generation_failure_returns_explicit_safe_escalation() -> None:
    service = CopilotService(ROOT, generation_provider=FailingProvider())

    result = service.ask("An unknown alarm and damaged cable appeared together.")

    assert result["provider_status"] == "error"
    assert result["response"]["escalate"] is True
    assert result["response"]["confidence"] == "low"
    assert result["citations_valid"] is True


def test_vision_flow_uses_development_image_and_saved_threshold() -> None:
    service = CopilotService(ROOT, load_generation=False)
    manifest = pd.read_csv(ROOT / "data" / "processed" / "mvtec_cable_manifest.csv")
    development = manifest.loc[manifest["project_split"] == "development"].iloc[0]
    image = ROOT / "data" / "external" / "mvtec_ad" / development["path"]

    result = service.inspect_image(image)

    assert result["score"] >= 0
    assert result["threshold"] > 0
    assert result["anomaly_map"].shape == (128, 128)
    assert "cannot declare equipment safe" in result["disclaimer"]


def test_health_does_not_claim_final_test_use() -> None:
    service = CopilotService(ROOT, load_generation=False)

    assert service.health() == {
        "status": "ok",
        "redis": "not_configured",
        "generation": "unavailable",
        "final_test_used": False,
    }
