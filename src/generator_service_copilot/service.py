from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from redis import Redis
from redis.exceptions import RedisError
from sklearn.metrics.pairwise import cosine_similarity

from generator_service_copilot.cache import ResponseCache
from generator_service_copilot.data import sha256_file
from generator_service_copilot.nlp import encode, load_embedder
from generator_service_copilot.prompts import (
    GenerationRequest,
    LocalTransformersProvider,
    Provider,
    ServiceResponse,
    load_variants,
    parse_response,
)
from generator_service_copilot.redis_index import create_index, replace_chunks, search
from generator_service_copilot.vision import predict_patch

ABSTENTION_THRESHOLD = 0.65
SAFETY_INTENTS = {"electrical_inspection", "gas_quality", "human_escalation"}


class CopilotService:
    def __init__(
        self,
        project_root: Path,
        redis_client: Redis | None = None,
        generation_provider: Provider | None = None,
        load_generation: bool = True,
    ) -> None:
        self.root = project_root.resolve()
        nlp_models = self.root / "artifacts" / "nlp" / "models"
        self.embedder = load_embedder(self.root)
        self.intent_model = joblib.load(nlp_models / "intent_embedding_mlp.joblib")
        self.lexical = joblib.load(nlp_models / "retrieval_lexical.joblib")
        self.document_embeddings = np.load(nlp_models / "document_embeddings.npy")
        self.chunks = pd.read_csv(self.root / "data" / "processed" / "document_chunks.csv")
        self.lexical_chunks = self.lexical.transform(self.chunks["text"])
        self.redis_client = redis_client
        if self.redis_client is not None:
            try:
                self.redis_client.ping()
                create_index(self.redis_client, self.document_embeddings.shape[1])
                replace_chunks(self.redis_client, self.chunks, self.document_embeddings)
            except RedisError:
                self.redis_client = None
        self.selected_variant = next(
            variant
            for variant in load_variants(self.root / "prompts" / "variants.json")
            if variant.variant_id == "v3_few_shot"
        )
        if generation_provider is not None:
            self.generation_provider = generation_provider
        elif load_generation:
            model_path = self.root / ".hf_cache" / "generation" / "smollm2-360m-instruct"
            self.generation_provider = LocalTransformersProvider(model_path)
        else:
            self.generation_provider = None
        corpus_version = sha256_file(self.root / "data" / "processed" / "document_chunks.csv")[:12]
        self.cache = ResponseCache(self.redis_client, corpus_version=corpus_version)

    def classify(self, question: str) -> dict[str, Any]:
        embedding = encode(self.embedder, [question])
        probabilities = self.intent_model.predict_proba(embedding)[0]
        index = int(probabilities.argmax())
        intent = str(self.intent_model.classes_[index])
        confidence = float(probabilities[index])
        return {
            "intent": intent,
            "confidence": confidence,
            "abstained": confidence < ABSTENTION_THRESHOLD,
            "embedding": embedding[0],
        }

    def retrieve(self, question: str, embedding: np.ndarray) -> tuple[list[dict[str, Any]], str]:
        if self.redis_client is not None:
            try:
                results = search(self.redis_client, embedding, top_k=3)
                if results:
                    return [asdict(result) for result in results], "redis_vector"
            except RedisError:
                pass
        lexical = cosine_similarity(self.lexical.transform([question]), self.lexical_chunks)[0]
        dense = self.document_embeddings @ embedding
        scores = (0.6 * lexical) + (0.4 * dense)
        order = np.argsort(-scores)[:3]
        records = []
        for index in order:
            row = self.chunks.iloc[index]
            records.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "distance": float(1.0 - scores[index]),
                }
            )
        return records, "local_hybrid_fallback"

    def ask(self, question: str, seed: int = 42) -> dict[str, Any]:
        question = " ".join(question.split())
        if not question:
            raise ValueError("Question must not be empty")
        cached = self.cache.get(question, "local")
        if cached.value is not None:
            return cached.value | {"cache_status": "hit"}

        classification = self.classify(question)
        evidence, retrieval_mode = self.retrieve(question, classification["embedding"])
        safety_critical = classification["intent"] in SAFETY_INTENTS
        request = GenerationRequest(
            question=question,
            intent=classification["intent"],
            evidence=[{"chunk_id": item["chunk_id"], "text": item["text"]} for item in evidence],
            safety_critical=safety_critical,
        )
        response = None
        provider_status = "unavailable"
        if self.generation_provider is not None:
            try:
                generated = self.generation_provider.generate(self.selected_variant, request, seed)
                response = parse_response(generated.text)
                provider_status = "ok" if response else "invalid_output"
            except Exception:
                provider_status = "error"
        if response is None:
            response = ServiceResponse(
                summary="A structured answer could not be generated safely.",
                actions=["Review the cited approved guidance and request qualified support."],
                citations=[evidence[0]["chunk_id"]],
                escalate=True,
                confidence="low",
            )

        allowed = {item["chunk_id"] for item in evidence}
        citations_valid = set(response.citations).issubset(allowed)
        forced_escalation = classification["abstained"] or not citations_valid
        payload = {
            "question": question,
            "intent": classification["intent"],
            "intent_confidence": classification["confidence"],
            "abstained": classification["abstained"],
            "retrieval_mode": retrieval_mode,
            "evidence": evidence,
            "response": response.model_dump()
            | {"escalate": bool(response.escalate or forced_escalation)},
            "provider_status": provider_status,
            "citations_valid": citations_valid,
            "cache_status": cached.status,
        }
        self.cache.set(question, "local", payload)
        return payload

    def inspect_image(self, path: Path) -> dict[str, Any]:
        report = json.loads((self.root / "artifacts" / "vision" / "report.json").read_text())
        result = predict_patch(
            path,
            self.root / ".torch_cache" / "hub" / "checkpoints" / "resnet18-f37072fd.pth",
            self.root / "artifacts" / "vision" / "models" / "patch_memory.npy",
        )
        threshold = float(report["selected_metrics"]["threshold"])
        return {
            "score": result["score"],
            "threshold": threshold,
            "is_anomaly": result["score"] >= threshold,
            "anomaly_map": result["anomaly_map"],
            "model": report["selected_candidate"],
            "disclaimer": "Proxy cable-image result; it cannot declare equipment safe.",
        }

    def health(self) -> dict[str, Any]:
        redis_status = "not_configured"
        if self.redis_client is not None:
            try:
                redis_status = "ok" if self.redis_client.ping() else "error"
            except RedisError:
                redis_status = "unavailable"
        return {
            "status": "ok",
            "redis": redis_status,
            "generation": "ok" if self.generation_provider is not None else "unavailable",
            "final_test_used": False,
        }
