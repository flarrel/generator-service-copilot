from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from redis import Redis
from redis.exceptions import RedisError


@dataclass(frozen=True)
class CacheRead:
    value: dict[str, Any] | None
    status: str


class ResponseCache:
    def __init__(
        self,
        client: Redis | None,
        corpus_version: str,
        ttl_seconds: int = 900,
        namespace: str = "gsc:response:v1",
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("Cache TTL must be positive")
        self.client = client
        self.corpus_version = corpus_version
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace

    def key(self, question: str, provider: str) -> str:
        normalized = " ".join(question.lower().split())
        digest = hashlib.sha256(
            f"{self.corpus_version}\0{provider}\0{normalized}".encode()
        ).hexdigest()
        return f"{self.namespace}:{self.corpus_version}:{digest}"

    def get(self, question: str, provider: str) -> CacheRead:
        if self.client is None:
            return CacheRead(None, "unavailable")
        try:
            raw = self.client.get(self.key(question, provider))
        except RedisError:
            return CacheRead(None, "unavailable")
        if raw is None:
            return CacheRead(None, "miss")
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return CacheRead(None, "corrupt")
        if not isinstance(value, dict):
            return CacheRead(None, "corrupt")
        return CacheRead(value, "hit")

    def set(self, question: str, provider: str, value: dict[str, Any]) -> str:
        if self.client is None:
            return "unavailable"
        try:
            self.client.setex(
                self.key(question, provider),
                self.ttl_seconds,
                json.dumps(value, sort_keys=True),
            )
        except RedisError:
            return "unavailable"
        return "stored"

    def invalidate_current_version(self) -> int:
        if self.client is None:
            return 0
        try:
            keys = list(
                self.client.scan_iter(match=f"{self.namespace}:{self.corpus_version}:*", count=100)
            )
            return int(self.client.delete(*keys)) if keys else 0
        except RedisError:
            return 0
