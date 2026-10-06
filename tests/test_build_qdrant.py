"""Tests de scripts/build_qdrant.py : hors réseau (Qdrant en mémoire)."""
import math

import build_qdrant as bq
from embedbabel.gematria import get_alphabet

FR = get_alphabet("fr")


def test_vector_shape_and_normalisation():
    v = bq.word_vector("écureuil", "e.ky.ʁœj", FR)
    assert len(v) == bq.VECTOR_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in v[:bq.HASH_DIM])), 1.0, rel_tol=1e-9)


def test_vector_deterministic_and_discriminating():
    a = bq.word_vector("chat", "ʃa", FR)
    assert a == bq.word_vector("chat", "ʃa", FR)
    assert a != bq.word_vector("chien", "ʃjɛ̃", FR)


def test_point_id_stable():
    assert bq.point_id("fr", "chat", "fr-FR") == bq.point_id("fr", "chat", "fr-FR")
    assert bq.point_id("fr", "chat", "fr-FR") != bq.point_id("en", "chat", "fr-FR")


def test_upsert_in_memory_is_idempotent():
    from qdrant_client import QdrantClient
    client = QdrantClient(":memory:")
    bq.ensure_collection(client)
    rows = [("chat", "ʃa", "fr-FR", "t"), ("chien", "ʃjɛ̃", "fr-FR", "t"), ("chat", "ʃa", "fr-FR", "t")]
    for _ in range(2):  # relancer ne duplique pas
        client.upload_points(bq.COLLECTION, bq.iter_points("fr", rows, FR))
    assert client.count(bq.COLLECTION, exact=True).count == 2
    hit = client.query_points(bq.COLLECTION, query=bq.word_vector("chat", "ʃa", FR), limit=1).points[0]
    assert hit.payload["text"] == "chat"
