from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from redis import Redis
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    f1_score,
    log_loss,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from generator_service_copilot.data import sha256_file
from generator_service_copilot.redis_index import (
    REDIS_IMAGE,
    REDIS_IMAGE_DIGEST,
    create_index,
    replace_chunks,
    search,
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RANDOM_STATE = 42

INTENT_TO_DOCUMENT = {
    "alarm_interpretation": "GSC-DEMO-ALARM",
    "consumables": "GSC-DEMO-CONSUMABLES",
    "electrical_inspection": "GSC-DEMO-ELECTRICAL",
    "gas_quality": "GSC-DEMO-QUALITY",
    "human_escalation": "GSC-DEMO-ESCALATION",
    "installation": "GSC-DEMO-INSTALL",
    "maintenance": "GSC-DEMO-MAINTENANCE",
    "pressure_flow": "GSC-DEMO-PRESSURE",
}


def load_embedder(project_root: Path) -> SentenceTransformer:
    cache = project_root / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(cache))
    snapshot = (
        cache
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
        / EMBEDDING_REVISION
    )
    if not snapshot.exists():
        raise FileNotFoundError(
            "Pinned embedding snapshot is unavailable; download the selected model files first"
        )
    return SentenceTransformer(
        str(snapshot),
        device="cpu",
        local_files_only=True,
    )


def encode(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return np.asarray(
        embedder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray, bins: int = 10
) -> float:
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(error)


def classifier_metrics(
    labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    predictions = classes[probabilities.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "expected_calibration_error": expected_calibration_error(probabilities, labels, classes),
    }


def abstention_table(
    labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> list[dict[str, float]]:
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    rows: list[dict[str, float]] = []
    for threshold in np.arange(0.4, 0.96, 0.05):
        accepted = confidence >= threshold
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "coverage": float(accepted.mean()),
                "accepted_accuracy": float((predictions[accepted] == labels[accepted]).mean())
                if accepted.any()
                else 0.0,
            }
        )
    return rows


def select_abstention_threshold(rows: list[dict[str, float]]) -> float:
    eligible = [row for row in rows if row["coverage"] > 0 and row["accepted_accuracy"] >= 0.9]
    if not eligible:
        return 1.0
    return float(max(eligible, key=lambda row: row["coverage"])["threshold"])


def semantic_leakage_audit(train: pd.DataFrame, development: pd.DataFrame) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    train_features = vectorizer.fit_transform(train["text"])
    development_features = vectorizer.transform(development["text"])
    similarities = cosine_similarity(development_features, train_features)
    nearest = similarities.argmax(axis=1)
    maximum = similarities.max(axis=1)
    pairs = []
    for row_index in np.argsort(maximum)[-10:][::-1]:
        train_index = nearest[row_index]
        pairs.append(
            {
                "development_case_id": development.iloc[row_index]["case_id"],
                "training_case_id": train.iloc[train_index]["case_id"],
                "similarity": float(maximum[row_index]),
                "same_intent": bool(
                    development.iloc[row_index]["intent"] == train.iloc[train_index]["intent"]
                ),
            }
        )
    return {
        "method": "character TF-IDF cosine similarity, 3-5 grams",
        "maximum_similarity": float(maximum.max()),
        "p95_similarity": float(np.percentile(maximum, 95)),
        "mean_similarity": float(maximum.mean()),
        "pairs_at_or_above_0_90": int((maximum >= 0.9).sum()),
        "top_pairs": pairs,
    }


def _tfidf_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_intent_candidates(
    train: pd.DataFrame,
    development: pd.DataFrame,
    train_embeddings: np.ndarray,
    development_embeddings: np.ndarray,
    prototype_embeddings: np.ndarray,
    prototype_labels: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels_train = train["intent"].to_numpy()
    labels_development = development["intent"].to_numpy()
    candidates: dict[str, Any] = {
        "majority": DummyClassifier(strategy="prior"),
        "tfidf_logistic": _tfidf_pipeline(),
        "embedding_mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.001,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=1_500,
            random_state=RANDOM_STATE,
        ),
    }
    reports: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    development_probabilities: dict[str, np.ndarray] = {}
    for name, model in candidates.items():
        use_embeddings = name == "embedding_mlp"
        training_input = train_embeddings if use_embeddings else train["text"]
        training_labels = labels_train
        if use_embeddings:
            training_input = np.vstack(
                [train_embeddings, prototype_embeddings, prototype_embeddings, prototype_embeddings]
            )
            training_labels = np.concatenate(
                [labels_train, prototype_labels, prototype_labels, prototype_labels]
            )
        development_input = development_embeddings if use_embeddings else development["text"]
        started = time.perf_counter()
        model.fit(training_input, training_labels)
        probabilities = model.predict_proba(development_input)
        elapsed = time.perf_counter() - started
        metrics = classifier_metrics(
            labels_development,
            probabilities,
            model.classes_,
        )
        abstention = abstention_table(
            labels_development,
            probabilities,
            model.classes_,
        )
        reports[name] = {
            **metrics,
            "fit_and_development_inference_seconds": elapsed,
            "abstention": abstention,
            "selected_abstention_threshold": select_abstention_threshold(abstention),
            "manual_prototype_rows": int(len(prototype_labels) * 3) if use_embeddings else 0,
            "per_class": classification_report(
                labels_development,
                model.classes_[probabilities.argmax(axis=1)],
                output_dict=True,
                zero_division=0,
            ),
        }
        fitted[name] = model
        development_probabilities[name] = probabilities

    selected_name = max(
        reports,
        key=lambda name: (
            reports[name]["macro_f1"],
            -reports[name]["log_loss"],
            -reports[name]["expected_calibration_error"],
        ),
    )
    return reports, {
        "name": selected_name,
        "model": fitted[selected_name],
        "all": fitted,
        "development_probabilities": development_probabilities[selected_name],
    }


def rank_metrics(ranked_documents: list[list[str]], expected: list[str]) -> dict[str, float]:
    ranks = []
    for ranking, target in zip(ranked_documents, expected, strict=True):
        ranks.append(ranking.index(target) + 1 if target in ranking else 0)
    return {
        "recall_at_1": float(np.mean([rank == 1 for rank in ranks])),
        "recall_at_3": float(np.mean([0 < rank <= 3 for rank in ranks])),
        "mrr": float(np.mean([1.0 / rank if rank else 0.0 for rank in ranks])),
    }


def evaluate_retrieval(
    development: pd.DataFrame,
    chunks: pd.DataFrame,
    query_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, TfidfVectorizer]:
    expected = development["intent"].map(INTENT_TO_DOCUMENT).tolist()
    lexical = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    lexical_chunks = lexical.fit_transform(chunks["text"])
    lexical_scores = cosine_similarity(lexical.transform(development["text"]), lexical_chunks)
    dense_scores = query_embeddings @ chunk_embeddings.T

    methods: dict[str, np.ndarray] = {
        "lexical": lexical_scores,
        "dense": dense_scores,
        "hybrid": (0.6 * lexical_scores) + (0.4 * dense_scores),
    }
    reports: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    rankings: dict[str, list[list[str]]] = {}
    for name, scores in methods.items():
        order = np.argsort(-scores, axis=1)
        rankings[name] = [chunks.iloc[indexes]["document_id"].tolist() for indexes in order]
        reports[name] = rank_metrics(rankings[name], expected)

    for index, case in development.reset_index(drop=True).iterrows():
        row: dict[str, Any] = {
            "case_id": case["case_id"],
            "intent": case["intent"],
            "expected_document_id": expected[index],
        }
        for name in methods:
            ranking = rankings[name][index]
            row[f"{name}_rank"] = ranking.index(expected[index]) + 1
            row[f"{name}_top_3"] = "|".join(ranking[:3])
        rows.append(row)
    selected = max(
        reports,
        key=lambda name: (reports[name]["mrr"], reports[name]["recall_at_3"]),
    )
    return (
        {
            "candidates": reports,
            "selected": selected,
            "hybrid_weights": {"lexical": 0.6, "dense": 0.4},
        },
        pd.DataFrame.from_records(rows),
        lexical,
    )


def cluster_cases(
    train: pd.DataFrame, embeddings: np.ndarray, artifact_dir: Path
) -> dict[str, float]:
    model = KMeans(n_clusters=8, n_init=20, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores",
            category=UserWarning,
        )
        assignments = model.fit_predict(embeddings)
    labels = train["intent"].to_numpy()
    metrics = {
        "silhouette": float(silhouette_score(embeddings, assignments, metric="cosine")),
        "adjusted_rand_index": float(adjusted_rand_score(labels, assignments)),
        "normalized_mutual_information": float(normalized_mutual_info_score(labels, assignments)),
    }
    representatives = []
    for cluster in range(8):
        member_indexes = np.flatnonzero(assignments == cluster)
        distances = np.linalg.norm(
            embeddings[member_indexes] - model.cluster_centers_[cluster], axis=1
        )
        representative_index = member_indexes[distances.argmin()]
        representatives.append(
            {
                "cluster": cluster,
                "case_id": train.iloc[representative_index]["case_id"],
                "intent": train.iloc[representative_index]["intent"],
                "text": train.iloc[representative_index]["text"],
                "members": int(len(member_indexes)),
            }
        )
    pd.DataFrame.from_records(representatives).to_csv(
        artifact_dir / "cluster_representatives.csv", index=False
    )
    pd.crosstab(
        pd.Series(assignments, name="cluster"),
        pd.Series(labels, name="intent"),
    ).to_csv(artifact_dir / "cluster_composition.csv")

    os.environ.setdefault("MPLCONFIGDIR", str(artifact_dir.parents[1] / ".mplconfig"))
    from matplotlib.figure import Figure

    projected = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(embeddings)
    figure = Figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.subplots()
    scatter = axis.scatter(projected[:, 0], projected[:, 1], c=assignments, cmap="tab10", s=28)
    axis.set_title("Training service cases clustered from sentence embeddings")
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    figure.colorbar(scatter, ax=axis, label="Cluster")
    figure.savefig(artifact_dir / "service_case_clusters.png", dpi=160)
    return metrics


def train_nlp(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    processed = project_root / "data" / "processed"
    artifact_dir = project_root / "artifacts" / "nlp"
    model_dir = artifact_dir / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(processed / "service_cases_train.csv")
    development = pd.read_csv(processed / "service_cases_development.csv")
    chunks = pd.read_csv(processed / "document_chunks.csv")
    if set(train["split"]) != {"train"} or set(development["split"]) != {"development"}:
        raise ValueError("NLP training may only use train and development splits")
    embedder = load_embedder(project_root)
    train_embeddings = encode(embedder, train["text"].tolist())
    development_embeddings = encode(embedder, development["text"].tolist())
    chunk_embeddings = encode(embedder, chunks["text"].tolist())
    document_to_intent = {document: intent for intent, document in INTENT_TO_DOCUMENT.items()}
    prototype_labels = chunks["document_id"].map(document_to_intent).to_numpy()

    leakage = semantic_leakage_audit(train, development)
    candidates, selection = evaluate_intent_candidates(
        train,
        development,
        train_embeddings,
        development_embeddings,
        chunk_embeddings,
        prototype_labels,
    )
    for name, model in selection["all"].items():
        joblib.dump(model, model_dir / f"intent_{name}.joblib")
    selected_model = selection["model"]
    selected_probabilities = selection["development_probabilities"]
    intent_predictions = development[["case_id", "intent", "text"]].copy()
    intent_predictions["predicted_intent"] = selected_model.classes_[
        selected_probabilities.argmax(axis=1)
    ]
    intent_predictions["confidence"] = selected_probabilities.max(axis=1)
    intent_predictions["correct"] = (
        intent_predictions["intent"] == intent_predictions["predicted_intent"]
    )
    intent_predictions.to_csv(artifact_dir / "intent_development_predictions.csv", index=False)
    np.save(model_dir / "document_embeddings.npy", chunk_embeddings)

    retrieval, retrieval_rows, lexical_retriever = evaluate_retrieval(
        development,
        chunks,
        development_embeddings,
        chunk_embeddings,
    )
    joblib.dump(lexical_retriever, model_dir / "retrieval_lexical.joblib")
    retrieval_rows.to_csv(artifact_dir / "retrieval_development.csv", index=False)
    clustering = cluster_cases(train, train_embeddings, artifact_dir)

    report: dict[str, Any] = {
        "sealed_test_used": False,
        "data": {
            "train_rows": len(train),
            "development_rows": len(development),
            "test_rows_evaluated": 0,
            "train_sha256": sha256_file(processed / "service_cases_train.csv"),
            "development_sha256": sha256_file(processed / "service_cases_development.csv"),
        },
        "embedding_model": {
            "model_id": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "dimensions": int(chunk_embeddings.shape[1]),
            "normalized": True,
        },
        "leakage_audit": leakage,
        "intent_candidates": candidates,
        "selected_intent_model": selection["name"],
        "retrieval": retrieval,
        "clustering": clustering,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    artifact_paths = [
        artifact_dir / "report.json",
        artifact_dir / "retrieval_development.csv",
        artifact_dir / "cluster_representatives.csv",
        artifact_dir / "cluster_composition.csv",
        artifact_dir / "intent_development_predictions.csv",
        model_dir / "document_embeddings.npy",
        model_dir / "intent_majority.joblib",
        model_dir / "intent_tfidf_logistic.joblib",
        model_dir / "intent_embedding_mlp.joblib",
        model_dir / "retrieval_lexical.joblib",
    ]
    redis_verification_path = artifact_dir / "redis_verification.json"
    if redis_verification_path.exists():
        artifact_paths.append(redis_verification_path)
    contract = {
        "schema_version": 1,
        "sealed_test_used": False,
        "selected_intent_model": selection["name"],
        "selected_retrieval": retrieval["selected"],
        "embedding_model": report["embedding_model"],
        "files": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    (artifact_dir / "artifact_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return report


def verify_redis(project_root: Path, redis_url: str) -> dict[str, Any]:
    project_root = project_root.resolve()
    artifact_dir = project_root / "artifacts" / "nlp"
    chunks = pd.read_csv(project_root / "data" / "processed" / "document_chunks.csv")
    embeddings = np.load(artifact_dir / "models" / "document_embeddings.npy")
    client = Redis.from_url(redis_url, decode_responses=False)
    client.ping()
    create_index(client, embeddings.shape[1])
    replace_chunks(client, chunks, embeddings)

    embedder = load_embedder(project_root)
    checks = [
        ("The cable harness has damaged insulation.", "GSC-DEMO-ELECTRICAL"),
        ("A suspected gas leak was reported.", "GSC-DEMO-ESCALATION"),
        ("The service record is missing a completed checklist.", "GSC-DEMO-MAINTENANCE"),
    ]
    outcomes = []
    for query, expected_document in checks:
        results = search(client, encode(embedder, [query])[0], top_k=3)
        outcomes.append(
            {
                "query": query,
                "expected_document_id": expected_document,
                "returned_document_ids": [result.document_id for result in results],
                "passed": expected_document in [result.document_id for result in results],
            }
        )
    info = client.execute_command("FT.INFO", "gsc_manuals_v1")
    verification = {
        "redis_version": client.info("server")["redis_version"],
        "container_image": REDIS_IMAGE,
        "container_image_digest": REDIS_IMAGE_DIGEST,
        "index_name": "gsc_manuals_v1",
        "indexed_chunks": len(chunks),
        "embedding_dimensions": int(embeddings.shape[1]),
        "queries": outcomes,
        "all_queries_passed": all(outcome["passed"] for outcome in outcomes),
        "index_info_fields": len(info) // 2,
    }
    (artifact_dir / "redis_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate NLP components")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--redis-url")
    args = parser.parse_args()
    report = train_nlp(args.project_root)
    if args.redis_url:
        report["redis"] = verify_redis(args.project_root, args.redis_url)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
