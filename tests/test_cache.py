import json

from redis.exceptions import ConnectionError

from generator_service_copilot.cache import ResponseCache


class FakeRedis:
    def __init__(self) -> None:
        self.values = {}
        self.ttls = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.ttls[key] = ttl

    def scan_iter(self, match, count):
        del count
        prefix = match.removesuffix("*")
        return iter([key for key in self.values if key.startswith(prefix)])

    def delete(self, *keys):
        removed = 0
        for key in keys:
            removed += int(key in self.values)
            self.values.pop(key, None)
            self.ttls.pop(key, None)
        return removed


class BrokenRedis(FakeRedis):
    def get(self, key):
        del key
        raise ConnectionError("offline")

    def setex(self, key, ttl, value):
        del key, ttl, value
        raise ConnectionError("offline")


def test_response_cache_records_ttl_hits_and_current_version_invalidation() -> None:
    client = FakeRedis()
    cache = ResponseCache(client, corpus_version="docs-a", ttl_seconds=30)

    assert cache.get("Question", "local").status == "miss"
    assert cache.set("Question", "local", {"answer": 1}) == "stored"
    key = cache.key("Question", "local")
    assert client.ttls[key] == 30
    assert cache.get(" question ", "local").value == {"answer": 1}
    assert cache.invalidate_current_version() == 1
    assert cache.get("Question", "local").status == "miss"


def test_document_version_changes_cache_namespace() -> None:
    client = FakeRedis()
    old = ResponseCache(client, corpus_version="docs-a")
    new = ResponseCache(client, corpus_version="docs-b")
    old.set("Question", "local", {"answer": "old"})

    assert old.key("Question", "local") != new.key("Question", "local")
    assert new.get("Question", "local").status == "miss"


def test_corrupt_and_unavailable_cache_fall_back_cleanly() -> None:
    client = FakeRedis()
    cache = ResponseCache(client, corpus_version="docs-a")
    client.values[cache.key("Question", "local")] = json.dumps(["wrong shape"])

    assert cache.get("Question", "local").status == "corrupt"
    assert ResponseCache(BrokenRedis(), "docs-a").get("Question", "local").status == "unavailable"
    assert ResponseCache(BrokenRedis(), "docs-a").set("Question", "local", {}) == "unavailable"
