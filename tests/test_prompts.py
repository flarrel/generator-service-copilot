import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from generator_service_copilot.prompts import (
    GenerationRequest,
    OpenAIResponsesProvider,
    PromptVariant,
    ServiceResponse,
    action_consistency,
    has_unsafe_instruction,
    load_variants,
    parse_response,
    select_cases,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_four_versioned_prompt_variants_load() -> None:
    variants = load_variants(ROOT / "prompts" / "variants.json")

    assert [variant.variant_id for variant in variants] == [
        "v1_direct",
        "v2_grounded_schema",
        "v3_few_shot",
        "v4_safety_rules",
    ]
    assert all(variant.version == "1.0" for variant in variants)


def test_parse_response_rejects_missing_or_extra_fields() -> None:
    valid = {
        "summary": "Review the documented condition.",
        "actions": ["Request qualified review."],
        "citations": ["chunk-1"],
        "escalate": True,
        "confidence": "high",
    }

    assert parse_response(json.dumps(valid)) is not None
    assert parse_response(json.dumps(valid | {"extra": "not allowed"})) is None
    del valid["citations"]
    assert parse_response(json.dumps(valid)) is None


def test_safety_detection_and_action_consistency() -> None:
    safe = ServiceResponse(
        summary="Escalate.",
        actions=["Request qualified review."],
        citations=["chunk-1"],
        escalate=True,
        confidence="high",
    )
    unsafe = safe.model_copy(
        update={"actions": ["Bypass the interlock and continue operation."], "escalate": False}
    )

    assert has_unsafe_instruction(safe) is False
    assert has_unsafe_instruction(unsafe) is True
    assert action_consistency([safe, safe]) == 1.0


def test_fixed_prompt_set_uses_development_cases_only() -> None:
    cases = pd.read_csv(ROOT / "data" / "processed" / "service_cases_development.csv")

    selected = select_cases(cases)

    assert len(selected) == 8
    assert selected["intent"].nunique() == 8
    assert set(selected["split"]) == {"development"}


def test_openai_provider_uses_stateless_responses_structured_output() -> None:
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text=(
                    '{"summary":"Escalate.","actions":["Request review."],'
                    '"citations":["chunk-1"],"escalate":true,"confidence":"high"}'
                )
            )

    client = SimpleNamespace(responses=Responses())
    provider = OpenAIResponsesProvider("configured-model", client=client)
    variant = PromptVariant("test", "Test", "Use evidence.")
    request = GenerationRequest(
        question="What next?",
        intent="human_escalation",
        evidence=[{"chunk_id": "chunk-1", "text": "Request review."}],
        safety_critical=True,
    )

    result = provider.generate(variant, request, seed=42)

    assert parse_response(result.text) is not None
    assert captured["model"] == "configured-model"
    assert captured["store"] is False
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["strict"] is True
    assert "api" not in json.dumps(captured).lower()
