from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from redis import Redis
from redis.exceptions import ResponseError

INDEX_NAME = "gsc_manuals_v1"
KEY_PREFIX = "gsc:chunk:"
VECTOR_FIELD = "embedding"
REDIS_IMAGE = "redis:8.2.3-alpine"
REDIS_IMAGE_DIGEST = "sha256:08ad0b1d280850169a790dba1393ff7a90aef951fc19632cf4d3ce4f78e679ba"


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    distance: float


def create_index(client: Redis, dimension: int) -> None:
    try:
        client.execute_command("FT.INFO", INDEX_NAME)
        return
    except ResponseError as error:
        if "Unknown index name" not in str(error) and "no such index" not in str(error).lower():
            raise
    client.execute_command(
        "FT.CREATE",
        INDEX_NAME,
        "ON",
        "HASH",
        "PREFIX",
        "1",
        KEY_PREFIX,
        "SCHEMA",
        "document_id",
        "TAG",
        "safety_topic",
        "TAG",
        "section",
        "TEXT",
        "text",
        "TEXT",
        VECTOR_FIELD,
        "VECTOR",
        "FLAT",
        "6",
        "TYPE",
        "FLOAT32",
        "DIM",
        str(dimension),
        "DISTANCE_METRIC",
        "COSINE",
    )


def replace_chunks(client: Redis, chunks: pd.DataFrame, embeddings: np.ndarray) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk rows and embedding rows must align")
    existing = list(client.scan_iter(match=f"{KEY_PREFIX}*", count=100))
    if existing:
        client.delete(*existing)
    pipeline = client.pipeline(transaction=False)
    for row, embedding in zip(chunks.to_dict("records"), embeddings, strict=True):
        pipeline.hset(
            f"{KEY_PREFIX}{row['chunk_id']}",
            mapping={
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "safety_topic": row["safety_topic"],
                "section": row["section"],
                "text": row["text"],
                VECTOR_FIELD: np.asarray(embedding, dtype=np.float32).tobytes(),
            },
        )
    pipeline.execute()


def search(client: Redis, query_embedding: np.ndarray, top_k: int = 3) -> list[SearchResult]:
    query = f"(*)=>[KNN {top_k} @{VECTOR_FIELD} $vector AS distance]"
    response = client.execute_command(
        "FT.SEARCH",
        INDEX_NAME,
        query,
        "PARAMS",
        "2",
        "vector",
        np.asarray(query_embedding, dtype=np.float32).tobytes(),
        "SORTBY",
        "distance",
        "RETURN",
        "4",
        "chunk_id",
        "document_id",
        "text",
        "distance",
        "DIALECT",
        "2",
    )
    results: list[SearchResult] = []
    for offset in range(1, len(response), 2):
        fields = response[offset + 1]
        decoded = {
            fields[index].decode(): fields[index + 1].decode() for index in range(0, len(fields), 2)
        }
        results.append(
            SearchResult(
                chunk_id=decoded["chunk_id"],
                document_id=decoded["document_id"],
                text=decoded["text"],
                distance=float(decoded["distance"]),
            )
        )
    return results
